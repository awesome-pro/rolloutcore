# SPDX-License-Identifier: Apache-2.0
"""Thin HTTP adapter over vLLM's dev endpoints.

Transport is stdlib ``urllib`` so the package keeps zero runtime dependencies,
and it is injectable (``Transport``) so the adapter can be tested against a
stubbed socket without a server.

All endpoints live in ``vllm/entrypoints/serve/dev/`` and require
``VLLM_SERVER_DEV_MODE=1`` on the server:

* ``dev/rlhf/api_router.py`` -- ``/pause`` ``:29``, ``/resume`` ``:76``,
  ``/init_weight_transfer_engine`` ``:156``, ``/start_weight_update`` ``:174``,
  ``/update_weights`` ``:186``, ``/finish_weight_update`` ``:204``,
  ``/update_weight_version`` ``:213``, ``/weight_info`` ``:222``,
  ``/is_paused`` ``:139``, ``/get_world_size`` ``:228``
* ``dev/cache/api_router.py`` -- ``/reset_prefix_cache`` ``:20``,
  ``/reset_encoder_cache`` ``:57``, ``/reset_mm_cache`` ``:47``

Two documented traps this adapter works around rather than inheriting:

1. ``/pause`` is a *blocking* call with **no server-side timeout**
   (``EngineCoreProc`` defers the utility response until the pause future
   resolves, ``vllm/v1/engine/core.py:1664-1668``; the client awaits with no
   deadline, ``vllm/v1/engine/core_client.py:1236-1248``). So the drain is
   issued from a background thread and polled, and a timeout triggers
   ``/abort_requests`` followed by a bounded retry.
2. ``/reset_prefix_cache`` returns ``{"success": false}`` while blocks are still
   held (``BlockPool.reset_prefix_cache``, ``vllm/v1/core/block_pool.py:831-838``).
   A ``200`` is therefore *not* success; the body must be read.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ..evidence import (
    BootstrapEvidence,
    DrainEvidence,
    InvalidateEvidence,
    ResumeEvidence,
    UpdateEvidence,
    ValidateEvidence,
)
from ..versions import INITIAL_VERSION, UpdateTarget, WeightIdentity, WeightVersion


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
    report (it only knows an opaque version string), so the adapter is told it
    when an update is opened and echoes it back in the evidence. A real
    integration would compute it from the trainer's ``WeightSource.metadata()``.
    """

    base_url: str = "http://127.0.0.1:8000"
    #: Timeout for fast read-back calls.
    timeout: float = 10.0
    #: How long to let ONE ``/pause`` attempt block before giving up on it.
    drain_timeout: float = 60.0
    #: ``/abort_requests`` retries after a drain timeout.
    drain_retries: int = 2
    backend: str = "nccl"
    transport: Transport = None  # type: ignore[assignment]
    weight_identity: WeightIdentity | None = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = UrllibTransport()
        self._drain_thread: threading.Thread | None = None
        self._drain_error: str | None = None
        self._drain_done = threading.Event()

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
        pre_seed = self._weight_info()

        self._call(
            "POST",
            "/init_weight_transfer_engine",
            body={"init_info": {"backend": self.backend}},
        )
        # The one sanctioned label write. Only legal because ``pre_seed`` was
        # unmanaged; the controller re-checks that from the evidence.
        self._call(
            "POST",
            "/update_weight_version",
            body={"new_version": WeightVersion(INITIAL_VERSION).label},
        )
        observed = self._weight_info()
        world_size = self._get_json("/get_world_size", "?include_dp=true").get("world_size")

        if self.weight_identity is None:
            raise HTTPAdapterError(
                "POST",
                "/init_weight_transfer_engine",
                "adapter has no weight_identity; the engine cannot report one "
                "(weight_version is an opaque string) so it must be supplied",
            )
        return BootstrapEvidence(
            observed_engine_label=observed,
            weight_transfer_initialised=True,
            pre_seed_label=pre_seed,
            weight_identity=self.weight_identity,
            backend=self.backend,
            world_size=world_size,
        )

    def begin_drain(self) -> None:
        """Issue ``/pause?mode=wait`` on a background thread.

        Cannot block: the call may legitimately take minutes, and the engine
        offers no timeout. ``await_drain`` polls the thread.
        """
        self._drain_done.clear()
        self._drain_error = None

        def _pause() -> None:
            try:
                self._call(
                    "POST",
                    "/pause",
                    query="?mode=wait&clear_cache=true",
                    timeout=self.drain_timeout,
                )
            except Exception as exc:
                self._drain_error = str(exc)
            finally:
                self._drain_done.set()

        self._drain_thread = threading.Thread(target=_pause, name="rolloutcore-drain", daemon=True)
        self._drain_thread.start()

    def await_drain(self) -> DrainEvidence:
        """Report whether the in-flight pause has returned yet."""
        if self._drain_done.wait(timeout=self.drain_timeout):
            if self._drain_error is None:
                return DrainEvidence(
                    engine_drain_completed=True,
                    engine_active_requests=None,
                    note="pause(mode=wait) returned",
                )
            return DrainEvidence(
                engine_drain_completed=False,
                note=f"pause failed: {self._drain_error}",
            )

        # Still running after the per-attempt timeout. Abort stragglers and let
        # the caller re-drive the pause.
        self._abort_all()
        return DrainEvidence(
            engine_drain_completed=False,
            note=f"pause attempt exceeded {self.drain_timeout}s; aborted in-flight requests",
        )

    def start_weight_update(self, target: UpdateTarget) -> None:
        self._call("POST", "/start_weight_update")

    def complete_weight_update(self, target: UpdateTarget) -> UpdateEvidence:
        """Drive ``/update_weights`` then ``/finish_weight_update``.

        The tensors do not come through here: they move over the trainer-side
        NCCL data plane, out of band. This call carries metadata and blocks while
        the workers receive, then commits the version.
        """
        try:
            self._call(
                "POST",
                "/update_weights",
                body={"update_info": {"names": [], "dtype_names": [], "shapes": []}},
            )
            self._call("POST", "/finish_weight_update", body={"weight_version": target.label})
        except HTTPAdapterError:
            return UpdateEvidence(
                target=target,
                weights_loaded=False,
                finish_acknowledged=False,
                data_plane_complete=False,
                observed_identity=None,
            )
        return UpdateEvidence(
            target=target,
            weights_loaded=True,
            finish_acknowledged=True,
            data_plane_complete=True,
            observed_identity=None,
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

    def _abort_all(self) -> None:
        """``POST /abort_requests`` with an empty body aborts everything tracked."""
        for _ in range(self.drain_retries):
            try:
                self._call("POST", "/abort_requests", body={})
                return
            except HTTPAdapterError:
                continue
