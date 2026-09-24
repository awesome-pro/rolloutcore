# SPDX-License-Identifier: Apache-2.0
"""The HTTP adapter, tested against a stubbed transport and a stub driver.

No server is needed: ``HttpVLLMAdapter`` takes an injectable ``Transport`` and an
injectable :class:`~rolloutcore.weight_transfer.WeightTransferDriver`, so these
tests assert the exact method/path/body sequence and the parsing of each
response. The traps covered here:

* ``/reset_prefix_cache`` returns HTTP 200 with ``{"success": false}`` while
  blocks are still held, so the body must be read and a 200 is not success.
* ``/pause`` has no server-side timeout, so it runs on a background thread,
  :meth:`await_drain` never blocks, and a failed attempt is aborted and
  *reissued* rather than polled forever.
* Bootstrap must refuse an ``rc-*`` engine **before writing**, or it steals
  another controller's ownership on its way to refusing.
* ``{"backend": "nccl"}`` is not a real transfer-init payload, and an empty
  ``update_info`` is not a real weight update -- both are the driver's job.
"""

from __future__ import annotations

import json
import threading
import time
import unittest

from support import IDENTITY_V0, TARGET_V1

from rolloutcore import (
    AlreadyManagedEngineError,
    DrainFailedError,
    LifecycleOnlyDriver,
    WeightTransferDriver,
    WeightTransferInit,
    WeightTransferNotConfiguredError,
    WeightTransferReport,
)
from rolloutcore.adapters.http import (
    HTTPAdapterError,
    HttpVLLMAdapter,
    Response,
)

BASE = "http://engine.test"

#: A realistic NCCL worker init payload. ``NCCLWeightTransferInitInfo``
#: (``vllm/distributed/weight_transfer/nccl_common.py:71``) requires
#: ``rank_offset`` + ``world_size`` plus exactly one rendezvous mode.
NCCL_INIT_INFO = {
    "master_address": "10.0.0.1",
    "master_port": 29500,
    "rank_offset": 1,
    "world_size": 3,
    "packed": True,
}

#: Metadata for one ``/update_weights`` chunk. Upstream ships real parameter
#: names/dtypes/shapes here; ``{names: [], dtype_names: [], shapes: []}`` is what
#: a fake weight update looked like and is what review item 2 removed.
UPDATE_INFO = {
    "names": ["model.embed_tokens.weight"],
    "dtype_names": ["bfloat16"],
    "shapes": [[151936, 2048]],
}


class StubDriverError(RuntimeError):
    """A trainer-side failure: the ambiguous case that must taint."""


class RecordingTransport:
    """A transport that records calls and returns canned responses."""

    def __init__(self, responses: dict[str, Response] | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses = responses or {}
        #: path -> ordered responses, consumed once per call and then repeated.
        #: Needed because ``/weight_info`` legitimately answers differently
        #: before and after bootstrap seeds the label.
        self.sequences: dict[str, list[Response]] = {}
        self.default = Response(status=200, body="{}")
        self.block_pause: threading.Event | None = None

    def request(self, method: str, url: str, body: bytes | None, timeout: float) -> Response:
        path = "/" + url.split("/", 3)[3].split("?")[0] if url.count("/") >= 3 else url
        parsed = json.loads(body) if body else None
        self.calls.append((method, path, parsed))

        if path == "/pause" and self.block_pause is not None:
            self.block_pause.wait(timeout=timeout + 1)

        for key, seq in self.sequences.items():
            if key in url:
                return seq.pop(0) if len(seq) > 1 else seq[0]
        for key, resp in self.responses.items():
            if key in url:
                return resp
        return self.default

    def paths(self) -> list[str]:
        return [p for _m, p, _b in self.calls]

    def body_for(self, path: str) -> dict | None:
        for _m, p, b in self.calls:
            if p == path:
                return b
        return None

    def count(self, path: str) -> int:
        return self.paths().count(path)


class StubDriver:
    """A trainer-side driver double that speaks the real wire sequence.

    It owns ``/init_weight_transfer_engine`` during ``initialize()`` and
    ``/start_weight_update`` -> ``/update_weights`` -> ``/finish_weight_update``
    inside ``transfer()``, exactly like upstream's trainer engine
    (``vllm/distributed/weight_transfer/base.py:617-627``). The adapter must not
    post those itself, and must not fabricate their payloads.
    """

    name = "stub-nccl"
    moves_tensors = True

    def __init__(
        self,
        transport: RecordingTransport,
        *,
        init_info: dict | None = None,
        backend: str = "nccl",
        initialised: bool = True,
        finish_acknowledged: bool = True,
        data_plane_complete: bool = True,
        chunks: int = 1,
        observed_identity: object = None,
        raise_on: str | None = None,
        init_returns_none: bool = False,
    ) -> None:
        self.transport = transport
        self.init_info = NCCL_INIT_INFO if init_info is None else init_info
        self.backend = backend
        self.initialised = initialised
        self.finish_acknowledged = finish_acknowledged
        self.data_plane_complete = data_plane_complete
        self.chunks = chunks
        self.observed_identity = observed_identity
        self.raise_on = raise_on
        self.init_returns_none = init_returns_none

    def _post(self, path: str, body: dict | None) -> None:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        resp = self.transport.request("POST", BASE + path, payload, 10.0)
        if resp.status >= 400:
            raise StubDriverError(f"POST {path} -> HTTP {resp.status}")

    def initialize(self) -> WeightTransferInit | None:
        if self.raise_on == "initialize":
            raise StubDriverError("rendezvous failed before any request")
        if self.init_returns_none:
            return None
        self._post("/init_weight_transfer_engine", {"init_info": self.init_info})
        return WeightTransferInit(
            driver=self.name,
            backend=self.backend,
            initialised=self.initialised,
            world_size=self.init_info.get("world_size"),
        )

    def transfer(self, target) -> WeightTransferReport:
        if self.raise_on == "transfer":
            raise StubDriverError("collective timed out mid-broadcast")
        self._post("/start_weight_update", None)
        for _ in range(self.chunks):
            self._post("/update_weights", {"update_info": UPDATE_INFO})
        self._post("/finish_weight_update", {"weight_version": target.label})
        return WeightTransferReport(
            finish_acknowledged=self.finish_acknowledged,
            data_plane_complete=self.data_plane_complete,
            chunks_transferred=self.chunks,
            observed_identity=self.observed_identity,
        )

    def shutdown(self) -> None:
        return None


def adapter(transport: RecordingTransport, **kw) -> HttpVLLMAdapter:
    return HttpVLLMAdapter(
        base_url=BASE,
        transport=transport,
        weight_identity=IDENTITY_V0,
        **kw,
    )


def adapter_with_driver(transport: RecordingTransport, **driver_kw) -> HttpVLLMAdapter:
    return adapter(transport, driver=StubDriver(transport, **driver_kw))


def json_response(payload: dict, status: int = 200) -> Response:
    return Response(status=status, body=json.dumps(payload))


def wait_for_drain(ad: HttpVLLMAdapter, predicate, *, timeout: float = 2.0):
    """Poll ``await_drain`` until ``predicate`` holds; return the last evidence.

    ``await_drain`` is non-blocking by contract, so "it returned" never means
    "the drain settled" -- a reissue note is an intermediate result, not an
    outcome. Returns the :class:`DrainFailedError` instead of raising it when the
    budget is exhausted, since most callers want to assert on it.
    """
    deadline = time.monotonic() + timeout
    ev = None
    while time.monotonic() < deadline:
        try:
            ev = ad.await_drain()
        except DrainFailedError as exc:
            return exc
        if predicate(ev):
            return ev
        time.sleep(0.001)
    return ev


def completed(ev) -> bool:
    return bool(getattr(ev, "engine_drain_completed", False))


class TestBootstrap(unittest.TestCase):
    def test_driver_owns_the_transfer_init_call(self):
        t = RecordingTransport(
            {
                "/update_weight_version": json_response({"success": True}),
                "/get_world_size": json_response({"world_size": 3}),
            }
        )
        t.sequences["/weight_info"] = [
            json_response({"weight_version": "default"}),
            json_response({"weight_version": "rc-0"}),
        ]
        ad = adapter_with_driver(t)

        ev = ad.bootstrap()

        self.assertEqual(
            t.paths(),
            [
                "/weight_info",
                "/init_weight_transfer_engine",
                "/update_weight_version",
                "/weight_info",
                "/get_world_size",
            ],
        )
        # The payload is a real rendezvous description, not {"backend": "nccl"}.
        self.assertEqual(t.body_for("/init_weight_transfer_engine"), {"init_info": NCCL_INIT_INFO})
        self.assertEqual(t.body_for("/update_weight_version"), {"new_version": "rc-0"})
        self.assertEqual(ev.observed_engine_label, "rc-0")
        self.assertEqual(ev.weight_transfer_driver, "stub-nccl")
        self.assertTrue(ev.weight_transfer_initialised)
        self.assertEqual(ev.backend, "nccl")
        self.assertEqual(ev.world_size, 3)
        self.assertIsNone(ev.failure_reason())

    def test_control_plane_only_bootstrap_skips_the_transfer_engine(self):
        """No driver -> no fabricated NCCL init. The evidence says so."""
        t = RecordingTransport()
        t.sequences["/weight_info"] = [
            json_response({"weight_version": "default"}),
            json_response({"weight_version": "rc-0"}),
        ]
        ev = adapter(t).bootstrap()

        self.assertNotIn("/init_weight_transfer_engine", t.paths())
        self.assertFalse(ev.weight_transfer_initialised)
        self.assertIsNone(ev.weight_transfer_driver)
        self.assertEqual(ev.backend, "none")
        self.assertIsNone(ev.world_size)
        self.assertIsNone(ev.failure_reason())

    def test_managed_engine_is_refused_before_any_write(self):
        """Review item 1: the refusal must not corrupt the other controller."""
        t = RecordingTransport({"/weight_info": json_response({"weight_version": "rc-7"})})
        ad = adapter_with_driver(t)

        with self.assertRaises(AlreadyManagedEngineError) as ctx:
            ad.bootstrap()

        self.assertEqual(ctx.exception.observed_label, "rc-7")
        self.assertIn("lease/recovery", str(ctx.exception))
        # The whole point: a read, and nothing else. Not even the driver ran.
        self.assertEqual(t.paths(), ["/weight_info"])
        self.assertNotIn("/update_weight_version", t.paths())
        self.assertNotIn("/init_weight_transfer_engine", t.paths())

    def test_refusal_happens_before_the_driver_rendezvous(self):
        """A refused adoption must not have opened a transfer session either."""
        t = RecordingTransport({"/weight_info": json_response({"weight_version": "rc-1"})})
        ad = adapter(t, driver=StubDriver(t))

        with self.assertRaises(AlreadyManagedEngineError):
            ad.bootstrap()
        self.assertEqual(t.paths(), ["/weight_info"])

    def test_pre_seed_label_is_captured_before_seeding(self):
        """The adapter must read the label *before* writing rc-0."""
        seen: list[str] = []

        class Ordered(RecordingTransport):
            def request(self, method, url, body, timeout):
                if "/weight_info" in url:
                    seen.append("read")
                if "/update_weight_version" in url:
                    seen.append("write")
                return super().request(method, url, body, timeout)

        t = Ordered()
        t.sequences["/weight_info"] = [
            json_response({"weight_version": "default"}),
            json_response({"weight_version": "rc-0"}),
        ]
        ev = adapter_with_driver(t).bootstrap()
        self.assertEqual(ev.pre_seed_label, "default")
        self.assertEqual(seen[0], "read")
        self.assertIn("write", seen)

    def test_missing_weight_identity_is_refused_before_any_write(self):
        t = RecordingTransport({"/weight_info": json_response({"weight_version": "default"})})
        ad = HttpVLLMAdapter(base_url=BASE, transport=t)
        with self.assertRaises(WeightTransferNotConfiguredError) as ctx:
            ad.bootstrap()
        self.assertIn("weight_identity", str(ctx.exception))
        self.assertEqual(t.paths(), [], "a config error must not touch the engine")

    def test_missing_weight_version_field_is_refused(self):
        t = RecordingTransport({"/weight_info": json_response({})})
        with self.assertRaises(HTTPAdapterError):
            adapter(t).bootstrap()

    def test_driver_init_failure_is_reported_not_hidden(self):
        """A failed rendezvous must not be reported as an initialised engine."""
        t = RecordingTransport()
        t.sequences["/weight_info"] = [
            json_response({"weight_version": "default"}),
            json_response({"weight_version": "rc-0"}),
        ]
        ad = adapter_with_driver(t, initialised=False)
        ev = ad.bootstrap()
        self.assertFalse(ev.weight_transfer_initialised)
        self.assertIsNotNone(ev.failure_reason())


class TestDrain(unittest.TestCase):
    def test_completed_drain(self):
        t = RecordingTransport({"/pause": Response(status=200, body="")})
        ad = adapter(t, drain_timeout=2.0)
        ad.begin_drain()
        ev = wait_for_drain(ad, completed)

        self.assertTrue(ev.engine_drain_completed)
        self.assertIsNone(ev.failure_reason())
        self.assertEqual(t.body_for("/pause"), None)  # query params, no body
        self.assertEqual(t.count("/pause"), 1, "a completed drain must not be reissued")

    def test_pause_query_uses_wait_mode(self):
        recorded: list[str] = []

        class UrlSpy(RecordingTransport):
            def request(self, method, url, body, timeout):
                recorded.append(url)
                return super().request(method, url, body, timeout)

        t = UrlSpy({"/pause": Response(status=200, body="")})
        ad = adapter(t, drain_timeout=2.0)
        ad.begin_drain()
        wait_for_drain(ad, completed)
        pause_urls = [u for u in recorded if "/pause" in u]
        self.assertTrue(pause_urls)
        self.assertIn("mode=wait", pause_urls[0])
        self.assertNotIn("mode=keep", pause_urls[0])

    def test_await_drain_never_blocks_on_a_stalled_attempt(self):
        """Review item 5: the per-attempt timeout must not become ours."""
        t = RecordingTransport({"/pause": Response(status=200, body="")})
        t.block_pause = threading.Event()  # never set: the pause hangs
        ad = adapter(t, drain_timeout=30.0)

        ad.begin_drain()
        started = time.monotonic()
        ev = ad.await_drain()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5, "await_drain blocked on the stalled attempt")
        self.assertFalse(ev.engine_drain_completed)
        self.assertIn("in flight", ev.note)
        # No abort: the attempt has not failed, it is still running.
        self.assertNotIn("/abort_requests", t.paths())
        t.block_pause.set()  # let the daemon thread finish

    def test_failed_attempt_is_aborted_then_reissued(self):
        t = RecordingTransport(
            {
                "/pause": json_response({"error": "boom"}, status=500),
                "/abort_requests": json_response({}),
            }
        )
        ad = adapter(t, drain_reissues=2)

        ad.begin_drain()
        ev = wait_for_drain(ad, lambda e: "reissued" in e.note)

        self.assertFalse(ev.engine_drain_completed)
        self.assertIn("reissued", ev.note)
        self.assertIn("/abort_requests", t.paths())
        self.assertEqual(t.count("/pause"), 2, "the failed attempt must be reissued")
        self.assertEqual(t.body_for("/abort_requests"), {})

    def test_reissue_budget_is_bounded_and_then_raises(self):
        """No silent 600-poll spin against an attempt that will never finish."""
        t = RecordingTransport(
            {
                "/pause": json_response({"error": "boom"}, status=500),
                "/abort_requests": json_response({}),
            }
        )
        ad = adapter(t, drain_reissues=2)

        ad.begin_drain()
        self.assertIn("reissued", ad.await_drain().note)  # attempt 2
        self.assertIn("reissued", ad.await_drain().note)  # attempt 3
        with self.assertRaises(DrainFailedError) as ctx:
            ad.await_drain()  # budget exhausted

        self.assertEqual(ctx.exception.attempts, 3)
        self.assertIn("remains paused", str(ctx.exception))
        self.assertEqual(t.count("/pause"), 3)

    def test_transient_transport_error_is_retried(self):
        class Flaky(RecordingTransport):
            def __init__(self):
                super().__init__({"/pause": Response(status=200, body="")})
                self.failures = 1
                self.attempts = 0

            def request(self, method, url, body, timeout):
                if "/pause" in url:
                    self.attempts += 1
                    if self.failures > 0:
                        self.failures -= 1
                        raise HTTPAdapterError("POST", "/pause", "connection reset")
                return super().request(method, url, body, timeout)

        t = Flaky()
        ad = adapter(t, drain_reissues=3)
        ad.begin_drain()
        ev = wait_for_drain(ad, completed)
        self.assertTrue(ev.engine_drain_completed)
        self.assertEqual(t.attempts, 2, "the failed attempt must be reissued")

    def test_abort_retries_are_bounded(self):
        t = RecordingTransport(
            {
                "/pause": json_response({"error": "boom"}, status=500),
                "/abort_requests": json_response({"error": "no"}, status=500),
            }
        )
        ad = adapter(t, drain_reissues=5, abort_retries=2)
        ad.begin_drain()
        ad.await_drain()
        self.assertEqual(t.count("/abort_requests"), 2)

    def test_await_drain_before_begin_drain_is_not_completed(self):
        ev = adapter(RecordingTransport()).await_drain()
        self.assertFalse(ev.engine_drain_completed)
        self.assertIn("never started", ev.note)


class TestWeightUpdate(unittest.TestCase):
    def test_driver_drives_the_round_trip(self):
        t = RecordingTransport()
        ad = adapter_with_driver(t)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)

        self.assertTrue(ev.weights_loaded)
        self.assertTrue(ev.finish_acknowledged)
        self.assertEqual(ev.target, TARGET_V1)
        self.assertIsNone(ev.failure_reason())
        update_paths = [p for p in t.paths() if "weight" in p]
        self.assertEqual(
            update_paths,
            ["/start_weight_update", "/update_weights", "/finish_weight_update"],
        )
        self.assertEqual(t.body_for("/finish_weight_update"), {"weight_version": TARGET_V1.label})
        # Real metadata, not an empty list dressed up as a transfer.
        self.assertEqual(t.body_for("/update_weights"), {"update_info": UPDATE_INFO})

    def test_adapter_does_not_post_start_itself(self):
        """Posting it here as well would open the engine's session twice."""
        t = RecordingTransport()
        ad = adapter_with_driver(t)
        ad.start_weight_update(TARGET_V1)
        self.assertEqual(t.paths(), [])
        ad.complete_weight_update(TARGET_V1)
        self.assertEqual(t.count("/start_weight_update"), 1)

    def test_lifecycle_only_driver_refuses_to_install_weights(self):
        """Review item 2: no driver means no weight update, loudly."""
        t = RecordingTransport()
        ad = adapter(t)  # default LifecycleOnlyDriver
        with self.assertRaises(WeightTransferNotConfiguredError) as ctx:
            ad.start_weight_update(TARGET_V1)
        self.assertIn("lifecycle-only", str(ctx.exception))
        self.assertEqual(t.paths(), [], "nothing may be sent for a fake update")

    def test_complete_without_start_is_refused(self):
        t = RecordingTransport()
        ad = adapter_with_driver(t)
        with self.assertRaises(WeightTransferNotConfiguredError):
            ad.complete_weight_update(TARGET_V1)
        self.assertEqual(t.paths(), [])

    def test_driver_exception_propagates(self):
        """Ambiguous mutating failures are the runner's to taint, not ours to hide."""
        t = RecordingTransport()
        ad = adapter_with_driver(t, raise_on="transfer")
        ad.start_weight_update(TARGET_V1)
        with self.assertRaises(StubDriverError):
            ad.complete_weight_update(TARGET_V1)

    def test_honest_partial_failure_becomes_failed_evidence(self):
        t = RecordingTransport()
        ad = adapter_with_driver(t, data_plane_complete=False)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)
        self.assertFalse(ev.weights_loaded)
        self.assertFalse(ev.data_plane_complete)
        self.assertIsNotNone(ev.failure_reason())

    def test_observed_identity_mismatch_is_evidence(self):
        t = RecordingTransport()
        ad = adapter_with_driver(t, observed_identity=IDENTITY_V0)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)
        self.assertIn("staged weight source", ev.failure_reason())

    def test_unacknowledged_finish_is_evidence(self):
        t = RecordingTransport()
        ad = adapter_with_driver(t, finish_acknowledged=False)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)
        self.assertIn("not acknowledged", ev.failure_reason())


class TestInvalidateCaches(unittest.TestCase):
    def test_all_three_called_and_success_read_from_the_body(self):
        t = RecordingTransport({"/reset_prefix_cache": json_response({"success": True})})
        ad = adapter(t)
        ev = ad.invalidate_caches(TARGET_V1)

        self.assertIsNone(ev.failure_reason())
        self.assertEqual(
            [p for p in t.paths() if p.startswith("/reset")],
            ["/reset_prefix_cache", "/reset_encoder_cache", "/reset_mm_cache"],
        )

    def test_http_200_with_success_false_is_a_failure(self):
        """The trap: 200 does not mean the cache was reset."""
        t = RecordingTransport({"/reset_prefix_cache": json_response({"success": False})})
        ev = adapter(t).invalidate_caches(TARGET_V1)

        self.assertFalse(ev.prefix_cache_reset)
        self.assertIn("prefix cache", ev.failure_reason())

    def test_reset_external_query_param_is_sent(self):
        urls: list[str] = []

        class UrlSpy(RecordingTransport):
            def request(self, method, url, body, timeout):
                urls.append(url)
                return super().request(method, url, body, timeout)

        t = UrlSpy({"/reset_prefix_cache": json_response({"success": True})})
        adapter(t).invalidate_caches(TARGET_V1)
        reset_url = next(u for u in urls if "/reset_prefix_cache" in u)
        # The HTTP param is `reset_external`; it fills the scheduler's
        # `reset_connector` slot positionally.
        self.assertIn("reset_external=true", reset_url)
        self.assertIn("reset_running_requests=true", reset_url)

    def test_encoder_reset_failure_is_reported_separately(self):
        t = RecordingTransport(
            {
                "/reset_prefix_cache": json_response({"success": True}),
                "/reset_encoder_cache": json_response({}, status=500),
            }
        )
        ev = adapter(t).invalidate_caches(TARGET_V1)
        self.assertTrue(ev.prefix_cache_reset)
        self.assertFalse(ev.encoder_cache_reset)
        self.assertIn("encoder cache", ev.failure_reason())


class TestValidatePreResume(unittest.TestCase):
    def test_never_calls_resume(self):
        """Amendment 1: validation must not resume the engine."""
        t = RecordingTransport(
            {
                "/weight_info": json_response({"weight_version": "rc-1"}),
                "/is_paused": json_response({"is_paused": True}),
            }
        )
        ev = adapter(t).validate_pre_resume(TARGET_V1)

        self.assertNotIn("/resume", t.paths())
        self.assertTrue(ev.is_paused)
        self.assertIsNone(ev.failure_reason())

    def test_reports_a_version_mismatch(self):
        t = RecordingTransport(
            {
                "/weight_info": json_response({"weight_version": "rc-9"}),
                "/is_paused": json_response({"is_paused": True}),
            }
        )
        ev = adapter(t).validate_pre_resume(TARGET_V1)
        self.assertIsNotNone(ev.failure_reason())
        self.assertIn("refusing to reconcile", ev.failure_reason())

    def test_reports_an_unpaused_engine(self):
        t = RecordingTransport(
            {
                "/weight_info": json_response({"weight_version": "rc-1"}),
                "/is_paused": json_response({"is_paused": False}),
            }
        )
        ev = adapter(t).validate_pre_resume(TARGET_V1)
        self.assertFalse(ev.is_paused)
        self.assertIsNotNone(ev.failure_reason())


class TestResume(unittest.TestCase):
    def test_happy_path(self):
        t = RecordingTransport(
            {
                "/weight_info": json_response({"weight_version": "rc-1"}),
                "/is_paused": json_response({"is_paused": False}),
            }
        )
        ev = adapter(t).resume(TARGET_V1)
        self.assertIn("/resume", t.paths())
        self.assertIsNone(ev.failure_reason())

    def test_failed_resume_is_reported(self):
        t = RecordingTransport(
            {
                "/resume": json_response({}, status=500),
                "/is_paused": json_response({"is_paused": True}),
                "/weight_info": json_response({"weight_version": "rc-1"}),
            }
        )
        ev = adapter(t).resume(TARGET_V1)
        self.assertFalse(ev.resume_acknowledged)
        self.assertIsNotNone(ev.failure_reason())

    def test_still_paused_after_resume_is_reported(self):
        t = RecordingTransport(
            {
                "/is_paused": json_response({"is_paused": True}),
                "/weight_info": json_response({"weight_version": "rc-1"}),
            }
        )
        ev = adapter(t).resume(TARGET_V1)
        self.assertTrue(ev.is_paused)
        self.assertIsNotNone(ev.failure_reason())


class TestErrorHandling(unittest.TestCase):
    def test_http_error_raises_with_context(self):
        t = RecordingTransport({"/weight_info": json_response({}, status=503)})
        with self.assertRaises(HTTPAdapterError) as ctx:
            adapter(t).validate_pre_resume(TARGET_V1)
        self.assertEqual(ctx.exception.path, "/weight_info")

    def test_malformed_json_raises(self):
        t = RecordingTransport({"/weight_info": Response(status=200, body="not json")})
        with self.assertRaises(HTTPAdapterError):
            adapter(t).validate_pre_resume(TARGET_V1)


class TestAdapterSatisfiesThePort(unittest.TestCase):
    def test_isinstance_check(self):
        from rolloutcore import LifecycleAdapter

        self.assertIsInstance(adapter(RecordingTransport()), LifecycleAdapter)

    def test_fake_adapter_satisfies_the_port(self):
        from rolloutcore import LifecycleAdapter
        from rolloutcore.adapters import FakeVLLMAdapter, seeded_engine

        self.assertIsInstance(FakeVLLMAdapter(seeded_engine(IDENTITY_V0)), LifecycleAdapter)

    def test_driver_doubles_satisfy_the_driver_protocol(self):
        self.assertIsInstance(StubDriver(RecordingTransport()), WeightTransferDriver)
        self.assertIsInstance(LifecycleOnlyDriver(), WeightTransferDriver)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
