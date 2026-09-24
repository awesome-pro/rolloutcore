# SPDX-License-Identifier: Apache-2.0
"""Thin HTTP adapter over vLLM's dev endpoints.

Transport is stdlib ``urllib`` so the package keeps zero runtime dependencies,
and it is injectable (``Transport``) so the adapter can be tested against a
stubbed socket without a server.

**Scope.** This adapter implements the lifecycle *control-plane* surface:
bootstrap, drain, cache invalidation, pre-resume validation and resume. Real
weight transfer is **not** implemented here -- the tensor data plane belongs to
the trainer side and reaches vLLM through
:class:`~rolloutcore.weight_transfer.WeightTransferDriver`, whose only shipped
implementation (``LifecycleOnlyDriver``) deliberately moves nothing. Do not
expect ``HttpVLLMAdapter`` to install new weights until a real driver exists
(Phase 3B).

All endpoints live in ``vllm/entrypoints/serve/dev/`` and require
``VLLM_SERVER_DEV_MODE=1`` on the server:

* ``dev/rlhf/api_router.py`` -- ``/pause`` ``:29``, ``/resume`` ``:76``,
  ``/init_weight_transfer_engine`` ``:156``, ``/start_weight_update`` ``:174``,
  ``/update_weights`` ``:186``, ``/finish_weight_update`` ``:204``,
  ``/update_weight_version`` ``:213``, ``/weight_info`` ``:222``,
  ``/is_paused`` ``:139``, ``/get_world_size`` ``:228``
* ``dev/cache/api_router.py`` -- ``/reset_prefix_cache`` ``:20``,
  ``/reset_encoder_cache`` ``:57``, ``/reset_mm_cache`` ``:47``

Three documented traps this adapter works around rather than inheriting:

1. ``/pause`` is a *blocking* call with **no server-side timeout**
   (``EngineCoreProc`` defers the utility response until the pause future
   resolves, ``vllm/v1/engine/core.py:1664-1668``; the client awaits with no
   deadline, ``vllm/v1/engine/core_client.py:1236-1248``). So the drain is issued
   from a background thread, :meth:`HttpVLLMAdapter.await_drain` never blocks,
   and a failed attempt is aborted and *reissued* rather than polled forever.
2. ``/reset_prefix_cache`` returns ``{"success": false}`` while blocks are still
   held (``BlockPool.reset_prefix_cache``, ``vllm/v1/core/block_pool.py:831-838``).
   A ``200`` is therefore *not* success; the body must be read.
3. Bootstrap must refuse an already-managed engine **before writing**. Reading
   ``/weight_info`` first and only then writing ``rc-0`` is not enough: if the
   engine already reports ``rc-7``, that write steals another controller's
   ownership before anyone can refuse. See :meth:`HttpVLLMAdapter.bootstrap`.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import (
    AlreadyManagedEngineError,
    DrainFailedError,
    WeightTransferNotConfiguredError,
)
from ..evidence import (
    BootstrapEvidence,
    DrainEvidence,
    InvalidateEvidence,
    ResumeEvidence,
    UpdateEvidence,
    ValidateEvidence,
)
from ..versions import INITIAL_VERSION, UpdateTarget, WeightIdentity, WeightVersion
from ..weight_transfer import (
    LifecycleOnlyDriver,
    WeightTransferDriver,
    WeightTransferInit,
    WeightTransferReport,
)

#: Drain attempt states, explicit rather than inferred from a thread's liveness.
DRAIN_IDLE = "IDLE"
DRAIN_IN_FLIGHT = "IN_FLIGHT"
DRAIN_FAILED = "FAILED"
DRAIN_COMPLETED = "COMPLETED"


class HTTPAdapterError(RuntimeError):
    """A control-plane call failed or returned an unusable body."""

    def __init__(self, method: str, path: str, detail: str) -> None:
        self.method = method
        self.path = path
        self.detail = detail
        super().__init__(f"{method} {path}: {detail}")


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: str

    def json(self) -> Any:
        try:
            return json.loads(self.body) if self.body else None
        except json.JSONDecodeError as exc:
            raise HTTPAdapterError("<transport>", "<response>", f"bad JSON: {exc}") from exc


class Transport(Protocol):
    """Minimal HTTP surface, so the adapter can be tested without a server."""

    def request(self, method: str, url: str, body: bytes | None, timeout: float) -> Response: ...


class UrllibTransport:
    """Default transport: stdlib only."""

    def request(self, method: str, url: str, body: bytes | None, timeout: float) -> Response:
        req = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return Response(status=resp.status, body=resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return Response(status=exc.code, body=exc.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise HTTPAdapterError(method, url, f"transport error: {exc.reason}") from exc


@dataclass
class HttpVLLMAdapter:
    """``LifecycleAdapter`` over a live ``vllm serve``.

    ``weight_identity`` is RolloutCore-side metadata that the engine does not
    report (it only knows an opaque version string), so the adapter is told it at
    bootstrap. A real integration computes it from the trainer's
    ``WeightSource.metadata()`` -- ideally with a
    :class:`~rolloutcore.versions.WeightSource`, since a manifest alone cannot
    distinguish two training steps of the same architecture.

    ``driver`` supplies the trainer side of a weight update. It defaults to
    :class:`~rolloutcore.weight_transfer.LifecycleOnlyDriver`, which moves no
    tensors and therefore makes any attempt to install weights fail loudly rather
    than send an empty ``update_info`` and call it a transfer.
    """

    base_url: str = "http://127.0.0.1:8000"
    #: Timeout for fast read-back calls.
    timeout: float = 10.0
    #: How long ONE ``/pause`` attempt may block before it is considered failed.
    #: :meth:`await_drain` never waits this long -- the attempt runs on its own
    #: thread and the caller polls.
    drain_timeout: float = 60.0
    #: How many times a failed ``/pause`` attempt is aborted and reissued before
    #: :class:`~rolloutcore.errors.DrainFailedError` is raised.
    drain_reissues: int = 2
    #: ``/abort_requests`` retries per reissue.
    abort_retries: int = 2
    transport: Transport = None  # type: ignore[assignment]
    weight_identity: WeightIdentity | None = None
    #: Trainer-side half of a weight update. The default moves no tensors, so a
    #: controller bootstrapped with it can never silently install weights.
    driver: WeightTransferDriver = field(default_factory=LifecycleOnlyDriver)
    #: Set once ``bootstrap`` has run; the target a transfer is expected for.
    _inflight: UpdateTarget | None = field(default=None, init=False, repr=False)
    _drain_lock: threading.Lock = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _drain_state: str = field(default=DRAIN_IDLE, init=False, repr=False)
    _drain_error: str | None = field(default=None, init=False, repr=False)
    _drain_attempts: int = field(default=0, init=False, repr=False)
    _drain_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = UrllibTransport()
        self._drain_lock = threading.Lock()

    # ------------------------------------------------------------- HTTP verbs

    def _url(self, path: str, query: str = "") -> str:
        return f"{self.base_url.rstrip('/')}{path}{query}"

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: str = "",
        timeout: float | None = None,
    ) -> Response:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        resp = self.transport.request(
            method, self._url(path, query), payload, timeout or self.timeout
        )
        if resp.status >= 400:
            raise HTTPAdapterError(method, path, f"HTTP {resp.status}: {resp.body[:400]}")
        return resp

    def _get_json(self, path: str, query: str = "") -> Any:
        return self._call("GET", path, query=query).json()

    # ------------------------------------------------------------- the port

    def bootstrap(self) -> BootstrapEvidence:
        """Bring a fresh, unmanaged engine under control.

        Order matters, and it is the review's item 1:

        1. Refuse if the adapter has no ``weight_identity`` -- a config error
           must not be discovered after a label write.
        2. Read ``GET /weight_info``.
        3. **Refuse if that label parses as ``rc-*``**, before any write. The
           controller-side check in ``BootstrapEvidence`` happens after this
           method returns, which is too late to protect the *other* controller's
           ownership of the engine.
        4. Ask the driver to initialize the transfer engine (it posts
           ``/init_weight_transfer_engine`` itself; see
           :mod:`rolloutcore.weight_transfer`).
        5. Write ``rc-0`` -- the one sanctioned label write.
        6. Read the label back and report it.
        """
        if self.weight_identity is None:
            raise WeightTransferNotConfiguredError(
                "bootstrap",
                "the adapter has no weight_identity; the engine cannot report one "
                "(weight_version is an opaque string) so it must be supplied",
            )

        pre_seed = self._weight_info()
        try:
            managed = WeightVersion.parse_label(pre_seed)
        except ValueError:
            managed = None
        if managed is not None:
            # Refuse *before* writing. Writing rc-0 here would silently steal an
            # engine another controller is driving, which is the exact
            # single-writer violation bootstrap exists to prevent.
            raise AlreadyManagedEngineError(pre_seed)

        init: WeightTransferInit | None = self.driver.initialize()

        self._call(
            "POST",
            "/update_weight_version",
            body={"new_version": WeightVersion(INITIAL_VERSION).label},
        )
        observed = self._weight_info()
        world_size = self._get_json("/get_world_size", "?include_dp=true").get("world_size")

        return BootstrapEvidence(
            observed_engine_label=observed,
            weight_transfer_initialised=bool(init and init.initialised),
            pre_seed_label=pre_seed,
            weight_identity=self.weight_identity,
            backend=init.backend if init else "none",
            world_size=world_size if init else None,
            weight_transfer_driver=init.driver if init else None,
        )

    def begin_drain(self) -> None:
        """Issue ``/pause?mode=wait`` on a background thread.

        Cannot block: the call may legitimately take minutes, and the engine
        offers no timeout. :meth:`await_drain` polls the attempt's *state*, which
        is tracked explicitly (``IN_FLIGHT``/``FAILED``/``COMPLETED``) rather
        than inferred from the thread object.
        """
        with self._drain_lock:
            self._drain_attempts = 0
        self._start_attempt()

    def await_drain(self) -> DrainEvidence:
        """Report drain progress. **Never blocks.**

        Three outcomes, and the third is the one that was missing:

        * ``COMPLETED`` -- report a completed drain.
        * ``IN_FLIGHT`` -- report not-completed immediately.
        * ``FAILED`` -- abort stragglers and issue a *fresh* pause attempt, up to
          ``drain_reissues``; past that, raise
          :class:`~rolloutcore.errors.DrainFailedError` so the runner stops
          polling instead of spinning 600 times against a dead attempt.

        The per-attempt socket timeout (``drain_timeout``) and the caller's
        deadline are therefore independent: this method returns promptly
        regardless of how long the attempt itself is allowed to hang.
        """
        with self._drain_lock:
            state, error, attempts = self._drain_state, self._drain_error, self._drain_attempts

        if state == DRAIN_COMPLETED:
            return DrainEvidence(
                engine_drain_completed=True,
                engine_active_requests=None,
                note="pause(mode=wait) returned",
            )
        if state == DRAIN_IN_FLIGHT:
            return DrainEvidence(
                engine_drain_completed=False,
                note="pause(mode=wait) still in flight",
            )
        if state == DRAIN_IDLE:
            return DrainEvidence(engine_drain_completed=False, note="drain was never started")

        # FAILED. Abort stragglers, then reissue (bounded).
        self._abort_all()
        if attempts > self.drain_reissues:
            raise DrainFailedError(attempts, error)
        self._start_attempt()
        return DrainEvidence(
            engine_drain_completed=False,
            note=(
                f"pause attempt {attempts} failed ({error}); "
                f"aborted stragglers and reissued attempt {attempts + 1}"
            ),
        )

    def start_weight_update(self, target: UpdateTarget) -> None:
        """Open the update. With a driver, the *driver* posts the session opener.

        Upstream's trainer engine drives ``/start_weight_update`` from inside
        ``send_weights()`` (``vllm/distributed/weight_transfer/base.py:617-627``),
        concurrently with the collective, so posting it here as well would open
        the session twice. This method therefore validates that a driver can
        install weights at all and records the target; it sends nothing.
        """
        if not self._driver_can_transfer():
            raise WeightTransferNotConfiguredError(
                "start_weight_update",
                f"driver {self.driver.name!r} moves no tensors. Nothing was sent, so "
                "the engine is still paused and the controller stays in UPDATING: "
                "configure a driving WeightTransferDriver, or taint the controller "
                "deliberately so the situation is explicit.",
            )
        self._inflight = target

    def complete_weight_update(self, target: UpdateTarget) -> UpdateEvidence:
        """Delegate the round trip to the driver and turn its report into evidence.

        The driver owns ``/start_weight_update`` -> ``/update_weights`` ->
        ``/finish_weight_update`` plus the tensor broadcast, because the metadata
        POST blocks while the workers receive and only returns once the
        broadcast is underway. Any exception here propagates to the runner, which
        treats it as an ambiguous mutating failure and taints.
        """
        if self._inflight != target:
            raise WeightTransferNotConfiguredError(
                "complete_weight_update",
                f"{target.describe()} was never opened via start_weight_update",
            )
        report: WeightTransferReport = self.driver.transfer(target)
        self._inflight = None
        return UpdateEvidence(
            target=target,
            weights_loaded=report.data_plane_complete,
            finish_acknowledged=report.finish_acknowledged,
            chunks_transferred=report.chunks_transferred,
            data_plane_complete=report.data_plane_complete,
            observed_identity=report.observed_identity,
        )

    def invalidate_caches(self, target: UpdateTarget) -> InvalidateEvidence:
        """Reset all three caches. The prefix result is read from the body."""
        prefix_body = self._call(
            "POST",
            "/reset_prefix_cache",
            query="?reset_running_requests=true&reset_external=true",
        ).json()
        prefix_ok = bool(prefix_body and prefix_body.get("success"))

        encoder_ok = True
        try:
            self._call("POST", "/reset_encoder_cache")
        except HTTPAdapterError:
            encoder_ok = False

        mm_ok = True
        try:
            self._call("POST", "/reset_mm_cache")
        except HTTPAdapterError:
            mm_ok = False

        return InvalidateEvidence(
            prefix_cache_reset=prefix_ok,
            encoder_cache_reset=encoder_ok,
            mm_cache_reset=mm_ok,
        )

    def validate_pre_resume(self, target: UpdateTarget) -> ValidateEvidence:
        """Read version and pause state. Deliberately never calls ``/resume``."""
        return ValidateEvidence(
            target=target,
            observed_engine_label=self._weight_info(),
            is_paused=bool(self._get_json("/is_paused").get("is_paused")),
        )

    def resume(self, target: UpdateTarget) -> ResumeEvidence:
        acknowledged = True
        try:
            self._call("POST", "/resume")
        except HTTPAdapterError:
            acknowledged = False
        return ResumeEvidence(
            target=target,
            resume_acknowledged=acknowledged,
            is_paused=bool(self._get_json("/is_paused").get("is_paused")),
            observed_engine_label=self._weight_info(),
        )

    # --------------------------------------------------------------- helpers

    def _weight_info(self) -> str:
        body = self._get_json("/weight_info")
        label = body.get("weight_version")
        if not isinstance(label, str):
            raise HTTPAdapterError("GET", "/weight_info", f"missing weight_version: {body!r}")
        return label

    def _driver_can_transfer(self) -> bool:
        """Whether the configured driver implements a real transfer.

        The lifecycle-only driver is a legitimate bootstrap mode but must never
        be mistaken for a weight path, so its ``transfer`` raises
        :class:`~rolloutcore.errors.WeightTransferNotConfiguredError` rather than
        reporting success for zero chunks.
        """
        return self.driver.moves_tensors

    def _start_attempt(self) -> None:
        """Kick off one ``/pause`` attempt and mark the drain IN_FLIGHT.

        Attempts are numbered so a superseded thread cannot write state after a
        reissue has already started a newer one.
        """
        with self._drain_lock:
            self._drain_attempts += 1
            attempt = self._drain_attempts
            self._drain_state = DRAIN_IN_FLIGHT
        self._drain_thread = threading.Thread(
            target=self._pause_attempt, args=(attempt,), name="rolloutcore-drain", daemon=True
        )
        self._drain_thread.start()

    def _pause_attempt(self, attempt: int) -> None:
        try:
            self._call(
                "POST",
                "/pause",
                query="?mode=wait&clear_cache=true",
                timeout=self.drain_timeout,
            )
        except Exception as exc:
            ok: bool = False
            error: str | None = str(exc)
        else:
            ok = True
            error = None
        with self._drain_lock:
            if attempt != self._drain_attempts:
                return  # superseded by a reissue; its verdict is stale
            self._drain_state = DRAIN_COMPLETED if ok else DRAIN_FAILED
            self._drain_error = error

    def _abort_all(self) -> None:
        """``POST /abort_requests`` with an empty body aborts everything tracked."""
        for _ in range(self.abort_retries):
            try:
                self._call("POST", "/abort_requests", body={})
                return
            except HTTPAdapterError:
                continue
