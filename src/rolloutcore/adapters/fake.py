# SPDX-License-Identifier: Apache-2.0
"""A :class:`LifecycleAdapter` over the in-memory fake engine.

This is the Phase 2 reference implementation of the port, and the one the
no-GPU cycle test and demo run against. ``adapters/http.py`` implements the same
protocol against a real ``vllm serve``.
"""

from __future__ import annotations

from ..errors import AlreadyManagedEngineError
from ..evidence import (
    BootstrapEvidence,
    DrainEvidence,
    InvalidateEvidence,
    ResumeEvidence,
    UpdateEvidence,
    ValidateEvidence,
)
from ..versions import INITIAL_VERSION, ParamSpec, UpdateTarget, WeightIdentity, WeightVersion
from .fake_engine import FakeEngineError, FakeVLLMEngine


class FakeVLLMAdapter:
    """Implements ``rolloutcore.port.LifecycleAdapter`` over ``FakeVLLMEngine``.

    Every method mirrors the control-plane call the HTTP adapter would make, and
    every piece of returned evidence is derived from observed engine state
    rather than assumed. That is what lets the tests use this adapter to check
    the *controller*, while the HTTP adapter is checked separately against a
    stubbed transport.
    """

    def __init__(
        self,
        engine: FakeVLLMEngine | None = None,
        *,
        chunks_per_update: int = 4,
    ) -> None:
        self.engine = engine if engine is not None else FakeVLLMEngine()
        self._chunks = chunks_per_update
        self._drain_completed = False

    #: Reported in ``BootstrapEvidence.weight_transfer_driver``. The fake adapter
    #: moves "tensors" in-process, which is why it can be used to test the whole
    #: cycle while the HTTP adapter cannot yet.
    driver_name = "fake-in-process"

    # ------------------------------------------------------------- the port

    def bootstrap(self) -> BootstrapEvidence:
        """Bring a fresh, unmanaged engine under control.

        Mirrors the HTTP adapter's ordering: the already-managed check happens
        **before** any write, so a refused adoption leaves the engine exactly as
        it was (review item 1). A lazy check would let this adapter seed ``rc-0``
        over another controller's ``rc-7`` and only then refuse.
        """
        identity = self.engine.weight_identity
        if identity is None:
            raise RuntimeError(
                "fake engine has no weight identity; call engine.seed_fresh_with(...) first"
            )
        pre_seed = self.engine.get_weight_info()
        if self.engine.is_managed():
            raise AlreadyManagedEngineError(pre_seed)

        self.engine.init_weight_transfer_engine({"backend": self.engine.backend})
        # Bootstrap is the one sanctioned label write: seed rc-0 over an
        # unmanaged label.
        self.engine.update_weight_version(WeightVersion(INITIAL_VERSION).label)

        return BootstrapEvidence(
            observed_engine_label=self.engine.get_weight_info(),
            weight_transfer_initialised=self.engine.weight_transfer_initialised,
            pre_seed_label=pre_seed,
            weight_identity=identity,
            backend=self.engine.backend,
            world_size=self.engine.get_world_size(),
            weight_transfer_driver=self.driver_name,
        )

    def begin_drain(self) -> None:
        """Issue the pause without blocking on completion."""
        self._drain_completed = False
        self._try_pause()

    def await_drain(self) -> DrainEvidence:
        """Report drain progress. Never blocks indefinitely."""
        if not self._drain_completed and not self._try_pause():
            return DrainEvidence(
                engine_drain_completed=False,
                engine_active_requests=self.engine.active_requests,
                note="drain still in flight",
            )
        return DrainEvidence(
            engine_drain_completed=True,
            engine_active_requests=self.engine.active_requests,
            note="pause(wait) returned",
        )

    def start_weight_update(self, target: UpdateTarget) -> None:
        self.engine.start_weight_update()

    def complete_weight_update(self, target: UpdateTarget) -> UpdateEvidence:
        """Transfer and finalize, reporting precisely which stage failed."""
        chunks = 0
        try:
            for _ in range(self._chunks):
                self.engine.update_weights({"names": [], "dtype_names": [], "shapes": []})
                chunks += 1
        except FakeEngineError:
            # The data plane failed: nothing was fully loaded.
            return UpdateEvidence(
                target=target,
                weights_loaded=False,
                finish_acknowledged=False,
                data_plane_complete=False,
                chunks_transferred=chunks,
                observed_identity=None,
            )

        # Every chunk loaded. The engine now holds the target's weights even if
        # the finalize call fails.
        self.engine.weight_identity = target.identity
        try:
            self.engine.finish_weight_update(target.label)
        except FakeEngineError:
            # Faithful to vLLM: ``AsyncLLM.finish_weight_update`` runs the
            # worker RPC and only writes the version *after* it returns
            # (vllm/v1/engine/async_llm.py:1284-1288), so a failed finalize
            # leaves the label unwritten.
            return UpdateEvidence(
                target=target,
                weights_loaded=True,
                finish_acknowledged=False,
                data_plane_complete=True,
                chunks_transferred=chunks,
                observed_identity=self.engine.weight_identity,
            )

        return UpdateEvidence(
            target=target,
            weights_loaded=True,
            finish_acknowledged=True,
            chunks_transferred=chunks,
            data_plane_complete=True,
            observed_identity=self.engine.weight_identity,
        )

    def invalidate_caches(self, target: UpdateTarget) -> InvalidateEvidence:
        prefix_ok = self.engine.reset_prefix_cache(reset_running_requests=True)
        self.engine.reset_encoder_cache()
        self.engine.reset_mm_cache()
        return InvalidateEvidence(
            prefix_cache_reset=prefix_ok,
            encoder_cache_reset=not self.engine.encoder_cache_dirty,
            mm_cache_reset=not self.engine.mm_cache_dirty,
        )

    def validate_pre_resume(self, target: UpdateTarget) -> ValidateEvidence:
        """Read back version and pause state. Deliberately does not resume."""
        return ValidateEvidence(
            target=target,
            observed_engine_label=self.engine.get_weight_info(),
            is_paused=self.engine.is_paused(),
            caches_still_clean=not (
                self.engine.prefix_cache_dirty
                or self.engine.encoder_cache_dirty
                or self.engine.mm_cache_dirty
            ),
        )

    def resume(self, target: UpdateTarget) -> ResumeEvidence:
        acknowledged = True
        try:
            self.engine.resume()
        except FakeEngineError:
            acknowledged = False
        return ResumeEvidence(
            target=target,
            resume_acknowledged=acknowledged,
            is_paused=self.engine.is_paused(),
            observed_engine_label=self.engine.get_weight_info(),
        )

    # --------------------------------------------------------------- helpers

    def _try_pause(self) -> bool:
        """Attempt ``pause(mode="wait")``; return whether it completed.

        ``clear_cache=False`` mirrors :class:`HttpVLLMAdapter`: the drain does
        not clear, because the clear belongs after the mutation (INVALIDATING).
        """
        try:
            self.engine.pause(mode="wait", clear_cache=False)
        except FakeEngineError:
            return False
        self._drain_completed = True
        return True


def seeded_engine(identity: WeightIdentity | None = None, **kwargs: object) -> FakeVLLMEngine:
    """A fresh engine with weights loaded, ready for ``bootstrap()``."""
    engine = FakeVLLMEngine(**kwargs)  # type: ignore[arg-type]
    engine.seed_fresh_with(identity or manifest_identity())
    return engine


def manifest_identity(seed: str = "v0") -> WeightIdentity:
    """A deterministic, distinct-per-seed identity for tests and the demo.

    Two different seeds model the plan's two distinguishable checkpoints
    (Model A / Model B), which is what the cache-coherence and replay tests need
    in order to tell "the update happened" from "it silently did not".
    """
    return WeightIdentity.from_param_specs(
        [
            ParamSpec("model.embed_tokens.weight", "bfloat16", (151936, 2048)),
            ParamSpec("model.layers.0.self_attn.q_proj.weight", "bfloat16", (2048, 2048)),
            ParamSpec(f"model.layers.0.mlp.{seed}.weight", "bfloat16", (2048, 11008)),
        ]
    )
