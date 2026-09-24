# SPDX-License-Identifier: Apache-2.0
"""Every transition is a typed call that either happens or raises.

The controller never performs I/O. It receives *evidence* describing what some
adapter observed, checks the evidence against the state's postconditions, and
either advances or taints. Keeping evidence as explicit objects (rather than
booleans threaded through the controller) is what makes the "proven zero active
work" and "cache invalidation actually happened" claims testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .versions import WeightVersion


class Evidence:
    """Base class. Subclasses report what an external system observed.

    ``failure_reason()`` returns ``None`` when the evidence satisfies the
    postcondition, or a human-readable string when it does not. A non-``None``
    reason always taints the engine: see ``docs/state-machine.md`` for why local
    precondition errors raise instead.
    """

    def failure_reason(self) -> str | None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BootstrapEvidence(Evidence):
    """Result of bringing an engine under RolloutCore control.

    Phase 2 adapter produces this from, in order:
    ``POST /init_weight_transfer_engine`` -> (optionally seed the label) ->
    ``POST /update_weight_version`` -> ``GET /weight_info`` -> ``GET /get_world_size``.
    All are vLLM dev endpoints (``vllm/entrypoints/serve/dev/rlhf/api_router.py:156``,
    ``:213``, ``:222``, ``:228``) and all require ``VLLM_SERVER_DEV_MODE=1``.

    **Bootstrap is the one sanctioned exception to "never write the version
    yourself".** A fresh engine reports the literal ``"default"``
    (``vllm/v1/engine/core.py:137``), so without seeding, no engine could ever
    enter service. The exception is narrowly scoped by ``seeded``/
    ``pre_seed_label``: the adapter may overwrite the label **only** when the
    pre-seed label is not already RolloutCore-managed. Clobbering an ``rc-*``
    label would mean two writers, which breaks "RolloutCore is the single
    lifecycle/weight writer".
    """

    #: Raw ``weight_version`` string reported by ``GET /weight_info`` *after*
    #: bootstrap has finished. Must be ``"rc-0"``.
    observed_engine_label: str
    #: Did ``POST /init_weight_transfer_engine`` return success?
    weight_transfer_initialised: bool
    #: Did the adapter have to write the initial label itself?
    seeded: bool = False
    #: What ``GET /weight_info`` reported *before* seeding, if ``seeded``.
    pre_seed_label: str | None = None
    #: ``world_size_across_dp`` from ``GET /get_world_size``; None if unknown.
    world_size: int | None = None
    #: Backend the engine was launched with, e.g. ``"nccl"``.
    backend: str = "nccl"

    def failure_reason(self) -> str | None:
        if not self.weight_transfer_initialised:
            return "weight transfer engine was not initialised"

        if not self.seeded:
            if self.pre_seed_label is not None:
                return "pre_seed_label was supplied but seeded is False"
            return None

        if self.pre_seed_label is None:
            return "adapter seeded the version label but did not report the pre-seed label"

        try:
            WeightVersion.parse_label(self.pre_seed_label)
        except ValueError:
            # Expected: the pre-seed label was unmanaged ("default"), so
            # overwriting it is legitimate bootstrap, not reconciliation.
            return None
        return (
            f"refusing to seed over pre-existing managed label {self.pre_seed_label!r}; "
            "another RolloutCore writer may own this engine"
        )


@dataclass(frozen=True, slots=True)
class DrainEvidence(Evidence):
    """Proof that the engine has no active work and is quiesced.

    Phase 2 adapter produces this from ``POST /pause?mode=wait&clear_cache=true``
    (``dev/rlhf/api_router.py:29``). That call returns only once the
    ``EngineCoreProc.pause_scheduler`` future resolves, which happens when
    ``Scheduler.get_num_unfinished_requests()`` reports zero
    (``vllm/v1/core/scheduler.py:2668-2669`` feeding ``has_work()`` at
    ``vllm/v1/engine/core.py:1457-1463``). That is the drain proof.

    ``engine_pause_confirmed`` must be True. If the pause RPC failed or timed
    out, the engine's state is unknown and the controller taints.
    """

    #: Active rollouts as RolloutCore counts them. Must be zero.
    active_rollouts: int
    #: Did the drain call actually return successfully?
    engine_pause_confirmed: bool
    #: Optional human note, e.g. "aborted 3 stragglers then re-paused".
    note: str = ""

    def failure_reason(self) -> str | None:
        if not self.engine_pause_confirmed:
            return "engine did not confirm a completed drain (pause not confirmed)"
        if self.active_rollouts != 0:
            # Unreachable via the public API: the controller checks its own
            # counter first and raises. Kept as defence in depth.
            return f"drain evidence claims zero work but {self.active_rollouts} rollouts are active"
        return None


@dataclass(frozen=True, slots=True)
class UpdateEvidence(Evidence):
    """Result of transferring and loading the new weights.

    Phase 2 adapter produces this from ``POST /finish_weight_update``
    (``dev/rlhf/api_router.py:204``) returning successfully. The engine's own
    commit point is ``AsyncLLM.finish_weight_update``
    (``vllm/v1/engine/async_llm.py:1284-1288``): it runs the worker-side
    ``finish_weight_update`` collective RPC and *then* writes the version.

    Critically, that RPC is the only thing vLLM does -- it invalidates no cache
    (``vllm/v1/worker/gpu_worker.py:1488-1505``; only ``reset_lora_state()``).
    Hence the separate INVALIDATING state.
    """

    #: The version this update was supposed to install.
    target_version: WeightVersion
    #: Did every chunk of ``POST /update_weights`` report success?
    weights_loaded: bool
    #: Did ``POST /finish_weight_update`` return success?
    finish_acknowledged: bool
    #: How many weight-transfer chunks the data plane carried. Diagnostic only.
    chunks_transferred: int | None = None
    #: Did the trainer-side NCCL/IPC data plane report completion?
    data_plane_complete: bool = True

    def failure_reason(self) -> str | None:
        if not self.weights_loaded:
            return "not all weight chunks were loaded"
        if not self.data_plane_complete:
            return "trainer-side weight data plane did not report completion"
        if not self.finish_acknowledged:
            return "finish_weight_update was not acknowledged by the engine"
        return None


@dataclass(frozen=True, slots=True)
class InvalidateEvidence(Evidence):
    """Proof that every cache holding version-N content has been dropped.

    All three flags are required. This is deliberately stricter than the
    ``clear_cache=true`` pause flag, because:

    * ``POST /pause?mode=wait&clear_cache=true`` already resets all three caches
      as a side effect (``vllm/v1/engine/core.py:877-882`` -> ``:861-875``), so
      re-asserting them here is cheap;
    * the ``clear_cache`` flag's documented semantics contradict the code
      (``dev/rlhf/api_router.py:46-47`` claims it is ignored for ``keep``, while
      ``core.py:2017-2021`` honours it), so relying on it implicitly is fragile;
    * it is the natural place to add a cache that pause does not know about.

    ``prefix_cache_reset`` comes from ``POST /reset_prefix_cache`` returning
    ``{"success": true}`` (``dev/cache/api_router.py:44``). Note the naming trap:
    the HTTP query parameter is ``reset_external`` but it fills the scheduler's
    ``reset_connector`` slot positionally.

    ``encoder_cache_reset`` and ``mm_cache_reset`` are **not** optional and are
    **not** covered by ``cache_salt``, which only reaches the prefix-cache block
    hash (``vllm/v1/core/kv_cache_utils.py:632-634``). This is the cache that
    ``finish_weight_update`` forgets: PR #48762 ("Invalidate encoder cache on
    finish_weight_update") was closed unmerged.
    """

    prefix_cache_reset: bool
    encoder_cache_reset: bool
    mm_cache_reset: bool

    def failure_reason(self) -> str | None:
        missing = [
            name
            for name, ok in (
                ("prefix cache", self.prefix_cache_reset),
                ("encoder cache", self.encoder_cache_reset),
                ("multimodal cache", self.mm_cache_reset),
            )
            if not ok
        ]
        if missing:
            return f"cache invalidation incomplete: {', '.join(missing)} not reset"
        return None


@dataclass(frozen=True, slots=True)
class ValidateEvidence(Evidence):
    """Proof that the engine is serving exactly the target version.

    Phase 2 adapter produces this from ``POST /resume`` followed by
    ``GET /weight_info`` (``dev/rlhf/api_router.py:76``, ``:222``) and
    ``GET /is_paused`` (``:139``).

    ``observed_engine_label`` is compared for **exact equality** with the
    target's label. A mismatch is a hard failure that taints: RolloutCore never
    "fixes" it by calling ``POST /update_weight_version``
    (``dev/rlhf/api_router.py:213``). Silently reconciling would mean publishing
    a version whose weights we cannot prove are installed -- the exact
    ``SERVING with partially updated weights`` state the old plan forbids.
    """

    target_version: WeightVersion
    #: Raw ``weight_version`` string reported by ``GET /weight_info``.
    observed_engine_label: str
    #: Did ``POST /resume`` return success?
    resume_acknowledged: bool
    #: ``GET /is_paused`` -- must be False.
    is_paused: bool

    def failure_reason(self) -> str | None:
        if not self.resume_acknowledged:
            return "resume was not acknowledged by the engine"
        if self.is_paused:
            return "engine still reports paused after resume"
        if self.observed_engine_label != self.target_version.label:
            return (
                f"engine reports weight_version {self.observed_engine_label!r} but "
                f"{self.target_version.label!r} was expected; refusing to reconcile"
            )
        return None


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One entry in the lifecycle journal.

    Exists so the Phase 4 metrics work (drain latency, update latency, total
    downtime) has a substrate, and so tests can assert on the exact path taken.
    """

    seq: int
    from_state: str
    to_state: str
    event: str
    version: WeightVersion | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CyclePlan:
    """The intents emitted when a cycle advances.

    Returned to the adapter so it knows exactly what I/O to perform next
    without the controller knowing anything about HTTP.
    """

    target_version: WeightVersion
    steps: tuple[str, ...] = field(default_factory=tuple)
