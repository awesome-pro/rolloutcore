# SPDX-License-Identifier: Apache-2.0
"""The HTTP adapter, tested against a stubbed transport.

No server is needed: ``HttpVLLMAdapter`` takes an injectable ``Transport``, so
these tests assert the exact method/path/body sequence and the parsing of each
response -- including the two traps the adapter exists to work around:

* ``/reset_prefix_cache`` returns HTTP 200 with ``{"success": false}`` while
  blocks are still held, so the body must be read and a 200 is not success.
* ``/pause`` has no server-side timeout, so it runs on a background thread and
  a stalled attempt triggers ``/abort_requests``.
"""

from __future__ import annotations

import json
import threading
import unittest

from support import IDENTITY_V0, TARGET_V1

from rolloutcore.adapters.http import (
    HTTPAdapterError,
    HttpVLLMAdapter,
    Response,
)


class RecordingTransport:
    """A transport that records calls and returns canned responses."""

    def __init__(self, responses: dict[str, Response] | None = None) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses = responses or {}
        self.default = Response(status=200, body="{}")
        self.block_pause: threading.Event | None = None

    def request(self, method: str, url: str, body: bytes | None, timeout: float) -> Response:
        path = "/" + url.split("/", 3)[3].split("?")[0] if url.count("/") >= 3 else url
        parsed = json.loads(body) if body else None
        self.calls.append((method, path, parsed))

        if path == "/pause" and self.block_pause is not None:
            self.block_pause.wait(timeout=timeout + 1)

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


def adapter(transport: RecordingTransport, **kw) -> HttpVLLMAdapter:
    return HttpVLLMAdapter(
        base_url="http://engine.test",
        transport=transport,
        weight_identity=IDENTITY_V0,
        **kw,
    )


def json_response(payload: dict, status: int = 200) -> Response:
    return Response(status=status, body=json.dumps(payload))


class TestBootstrap(unittest.TestCase):
    def test_call_sequence_and_evidence(self):
        t = RecordingTransport(
            {
                "/init_weight_transfer_engine": json_response({"message": "ok"}),
                "/update_weight_version": json_response({"success": True}),
                "/get_world_size": json_response({"world_size": 2}),
            }
        )
        # /weight_info is read twice; give it the post-seed answer.
        t.responses["/weight_info"] = json_response({"weight_version": "rc-0"})
        ad = adapter(t)

        ev = ad.bootstrap()

        self.assertIn("/init_weight_transfer_engine", t.paths())
        self.assertIn("/update_weight_version", t.paths())
        self.assertEqual(t.body_for("/update_weight_version"), {"new_version": "rc-0"})
        self.assertEqual(ev.observed_engine_label, "rc-0")
        self.assertEqual(ev.weight_identity, IDENTITY_V0)
        self.assertEqual(ev.world_size, 2)

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

        t = Ordered({"/weight_info": json_response({"weight_version": "default"})})
        ev = adapter(t).bootstrap()
        self.assertEqual(ev.pre_seed_label, "default")
        self.assertEqual(seen[0], "read")
        self.assertIn("write", seen)

    def test_missing_weight_identity_is_refused(self):
        t = RecordingTransport({"/weight_info": json_response({"weight_version": "rc-0"})})
        ad = HttpVLLMAdapter(base_url="http://engine.test", transport=t)
        with self.assertRaises(HTTPAdapterError) as ctx:
            ad.bootstrap()
        self.assertIn("weight_identity", str(ctx.exception))

    def test_missing_weight_version_field_is_refused(self):
        t = RecordingTransport({"/weight_info": json_response({})})
        with self.assertRaises(HTTPAdapterError):
            adapter(t).bootstrap()


class TestDrain(unittest.TestCase):
    def test_completed_drain(self):
        t = RecordingTransport({"/pause": Response(status=200, body="")})
        ad = adapter(t, drain_timeout=2.0)
        ad.begin_drain()
        ev = ad.await_drain()
        self.assertTrue(ev.engine_drain_completed)
        self.assertIsNone(ev.failure_reason())
        self.assertEqual(t.body_for("/pause"), None)  # query params, no body
        self.assertIn("/pause", t.paths())

    def test_pause_query_uses_wait_mode(self):
        recorded: list[str] = []

        class UrlSpy(RecordingTransport):
            def request(self, method, url, body, timeout):
                recorded.append(url)
                return super().request(method, url, body, timeout)

        t = UrlSpy({"/pause": Response(status=200, body="")})
        ad = adapter(t, drain_timeout=2.0)
        ad.begin_drain()
        ad.await_drain()
        pause_urls = [u for u in recorded if "/pause" in u]
        self.assertTrue(pause_urls)
        self.assertIn("mode=wait", pause_urls[0])
        self.assertNotIn("mode=keep", pause_urls[0])

    def test_stalled_pause_aborts_and_reports_not_completed(self):
        """The trap: /pause can block forever, so we bound it and abort."""
        t = RecordingTransport({"/pause": Response(status=200, body="")})
        t.block_pause = threading.Event()  # never set -> pause blocks
        ad = adapter(t, drain_timeout=0.05, drain_retries=1)

        ad.begin_drain()
        ev = ad.await_drain()

        self.assertFalse(ev.engine_drain_completed)
        self.assertIn("aborted", ev.note)
        self.assertIn("/abort_requests", t.paths())
        self.assertEqual(t.body_for("/abort_requests"), {})
        t.block_pause.set()  # let the daemon thread finish

    def test_failed_pause_reports_not_completed(self):
        t = RecordingTransport({"/pause": json_response({"error": "boom"}, status=500)})
        ad = adapter(t, drain_timeout=1.0)
        ad.begin_drain()
        ev = ad.await_drain()
        self.assertFalse(ev.engine_drain_completed)

    def test_abort_retries_are_bounded(self):
        t = RecordingTransport(
            {
                "/pause": Response(status=200, body=""),
                "/abort_requests": json_response({"error": "no"}, status=500),
            }
        )
        t.block_pause = threading.Event()
        ad = adapter(t, drain_timeout=0.05, drain_retries=2)
        ad.begin_drain()
        ad.await_drain()
        self.assertEqual(t.paths().count("/abort_requests"), 2)
        t.block_pause.set()


class TestWeightUpdate(unittest.TestCase):
    def test_start_then_transfer_then_finish(self):
        t = RecordingTransport()
        ad = adapter(t)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)

        self.assertTrue(ev.weights_loaded)
        self.assertTrue(ev.finish_acknowledged)
        self.assertEqual(ev.target, TARGET_V1)
        self.assertIsNone(ev.failure_reason())
        self.assertIn("/start_weight_update", t.paths())
        self.assertIn("/update_weights", t.paths())
        self.assertEqual(t.body_for("/finish_weight_update"), {"weight_version": TARGET_V1.label})
        # The version label is the generation, not the identity digest.
        self.assertEqual(t.body_for("/finish_weight_update")["weight_version"], "rc-1")

    def test_failed_finish_reports_failure_without_raising(self):
        t = RecordingTransport({"/finish_weight_update": json_response({}, status=500)})
        ad = adapter(t)
        ad.start_weight_update(TARGET_V1)
        ev = ad.complete_weight_update(TARGET_V1)
        self.assertFalse(ev.weights_loaded)
        self.assertFalse(ev.finish_acknowledged)
        self.assertIsNotNone(ev.failure_reason())

    def test_failed_update_weights_reports_failure(self):
        t = RecordingTransport({"/update_weights": json_response({}, status=503)})
        ad = adapter(t)
        ev = ad.complete_weight_update(TARGET_V1)
        self.assertFalse(ev.weights_loaded)
        self.assertNotIn("/finish_weight_update", t.paths())


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
        ad = adapter(t)
        ev = ad.invalidate_caches(TARGET_V1)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
