# SPDX-License-Identifier: Apache-2.0
"""An in-memory model of the vLLM dev-endpoint surface.

Enough fidelity to exercise the real cycle without a GPU:

* ``weight_version`` starts as the literal ``"default"`` and is only ever
  written by an explicit call, mirroring ``EngineCore._weight_version``
  (``vllm/v1/engine/core.py:137``, ``:1043``).
* ``pause(mode="wait")`` refuses to complete while requests are active -- the
  drain proof. ``mode="keep"`` is modelled too, so tests can show why it is
  unusable.
* ``finish_weight_update`` writes the version but invalidates **no cache**,
  mirroring ``AsyncLLM.finish_weight_update``
  (``vllm/v1/engine/async_llm.py:1284-1288``) and ``Worker.finish_weight_update``
  (``vllm/v1/worker/gpu_worker.py:1488-1505``).
* ``reset_prefix_cache`` can fail while blocks are still held, mirroring
  ``BlockPool.reset_prefix_cache`` (``vllm/v1/core/block_pool.py:831-838``).

Faults are injectable per operation so failure paths can be tested without a
real server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..versions import ENGINE_LABEL_PREFIX, WeightIdentity


class FakeEngineError(RuntimeError):
    """The simulated engine rejected an operation."""


@dataclass
class FakeWeightUpdate:
    """A weight transfer the fake engine can apply."""

    label: str
    identity: WeightIdentity


@dataclass
class FakeVLLMEngine:
    """In-memory stand-in for a ``vllm serve`` process with DEV_MODE endpoints."""

    #: Mirrors a fresh engine: the literal "default", not "rc-0".
    weight_version: str = "default"
    weight_identity: WeightIdentity | None = None
    #: Mirrors ``PauseState`` (``vllm/v1/core/sched/interface.py:24-35``).
    #: ``PAUSED_NEW`` blocks new admissions but keeps stepping running work --
    #: which is exactly what ``mode="wait"`` sets while draining.
    pause_state: str = "UNPAUSED"
    pause_mode: str | None = None
    active_requests: int = 0
    world_size: int = 1
    backend: str = "nccl"

    prefix_cache_dirty: bool = False
    encoder_cache_dirty: bool = False
    mm_cache_dirty: bool = False

    #: Mirrors BlockPool.reset_prefix_cache failing while blocks are held.
    prefix_reset_succeeds: bool = True

    weight_transfer_initialised: bool = False
    update_session_open: bool = False
    update_session_draft: bool = False

    #: Fault injection: operation name -> exception message.
    faults: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    @property
    def paused(self) -> bool:
        """Mirrors ``GET /is_paused``: True for both PAUSED_NEW and PAUSED_ALL."""
        return self.pause_state != "UNPAUSED"

    @property
    def fully_quiesced(self) -> bool:
        """Only true once the drain has actually completed."""
        return self.pause_state in ("PAUSED_NEW", "PAUSED_ALL") and self.active_requests == 0

    # ------------------------------------------------------------- endpoints

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name in self.faults:
            raise FakeEngineError(self.faults[name])

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._call("init_weight_transfer_engine")
        if not self.weight_transfer_initialised:
            self.weight_transfer_initialised = True

    def get_world_size(self) -> int:
        self._call("get_world_size")
        return self.world_size

    def get_weight_info(self) -> str:
        self._call("get_weight_info")
        return self.weight_version

    def update_weight_version(self, new_version: str) -> None:
        """``POST /update_weight_version``. RolloutCore only calls this at bootstrap."""
        self._call("update_weight_version")
        self.weight_version = new_version

    def is_paused(self) -> bool:
        self._call("is_paused")
        return self.paused

    def pause(self, mode: str = "abort", clear_cache: bool = True) -> None:
        """``POST /pause``.

        Faithful to ``EngineCoreProc.pause_scheduler``
        (``vllm/v1/engine/core.py:1984-2026``) in the one respect that matters:
        the pause state is set **first**, so new admissions are blocked
        immediately, and only then does the caller wait for running work to
        finish. ``mode="wait"`` therefore reports "not yet complete" while
        requests are active rather than leaving the engine unpaused.
        """
        self._call("pause")
        self.pause_mode = mode

        if mode == "keep":
            # PAUSED_ALL: freeze in place. Requests stay active; nothing drains.
            self.pause_state = "PAUSED_ALL"
            return

        # PAUSED_NEW for both "abort" and "wait".
        self.pause_state = "PAUSED_NEW"

        if mode == "abort":
            self.active_requests = 0
        elif mode == "wait" and self.active_requests > 0:
            raise FakeEngineError(
                f"pause(wait) not complete: {self.active_requests} request(s) active"
            )

        if clear_cache:
            # Mirrors _finish_pause -> _reset_caches (core.py:877-882, :861-875):
            # the pause resets all three caches as a side effect.
            self.prefix_cache_dirty = False
            self.encoder_cache_dirty = False
            self.mm_cache_dirty = False

    def resume(self) -> None:
        self._call("resume")
        self.pause_state = "UNPAUSED"
        self.pause_mode = None

    def start_weight_update(self) -> None:
        self._call("start_weight_update")
        if self.update_session_open:
            raise FakeEngineError("start_weight_update while a session is already open")
        self.update_session_open = True

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Control-plane metadata only; the data plane is out of band."""
        self._call("update_weights")
        if not self.update_session_open:
            raise FakeEngineError("update_weights without start_weight_update")

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finalize and commit. Invalidates nothing, exactly like vLLM."""
        self._call("finish_weight_update")
        if not self.update_session_open:
            raise FakeEngineError("finish_weight_update without start_weight_update")
        self.update_session_open = False
        self.weight_version = weight_version or self.weight_version

    def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        self._call("reset_prefix_cache")
        if not self.prefix_reset_succeeds:
            return False
        self.prefix_cache_dirty = False
        return True

    def reset_encoder_cache(self) -> None:
        self._call("reset_encoder_cache")
        self.encoder_cache_dirty = False

    def reset_mm_cache(self) -> None:
        self._call("reset_mm_cache")
        self.mm_cache_dirty = False

    # --------------------------------------------------------------- helpers

    def seed_fresh_with(self, identity: WeightIdentity) -> None:
        """Simulate a `vllm serve` that has just loaded weights."""
        self.weight_version = "default"
        self.weight_identity = identity
        self.prefix_cache_dirty = True
        self.encoder_cache_dirty = True
        self.mm_cache_dirty = True

    def is_managed(self) -> bool:
        return self.weight_version.startswith(ENGINE_LABEL_PREFIX)
