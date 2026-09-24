# SPDX-License-Identifier: Apache-2.0
"""Evidence: what an adapter observed, checked against a state's postcondition.

The controller never performs I/O. It receives *evidence*, checks it, and either
advances, raises (recoverable), or taints (terminal). Keeping evidence as
explicit frozen objects rather than booleans threaded through the controller is
what makes claims like "the drain was proven" and "the encoder cache was
actually dropped" testable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .versions import UpdateTarget, WeightIdentity, WeightVersion


class Evidence:
    """Base class.

    ``failure_reason()`` returns ``None`` when the evidence satisfies the
    postcondition, or a human-readable string when it does not. A non-``None``
    reason always taints; ``EvidenceNotReady`` is raised instead when the
    observation is merely incomplete (see ``lifecycle.py`` for the split).
    """

    def failure_reason(self) -> str | None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BootstrapEvidence(Evidence):
    """Result of bringing a fresh engine under RolloutCore control.

    The adapter must refuse **before writing anything** if the engine already
    reports an ``rc-*`` label: it reads ``GET /weight_info`` first and raises
    :class:`~rolloutcore.errors.AlreadyManagedEngineError` if the label parses as
    a RolloutCore version. The check in :meth:`failure_reason` below is the
    controller-side backstop for that contract, not the primary defence -- by the
    time the controller sees this object, a careless adapter would already have
    overwritten another controller's label.

    ``/init_weight_transfer_engine`` is posted by the **driver**, not the
    adapter: upstream's ``trainer_init`` opens the trainer endpoint and
    initializes the workers concurrently, so the two cannot be separated
    (``vllm/distributed/weight_transfer/factory.py:167``). ``weight_transfer_driver``
    records which driver did it, or ``None`` for the control-plane-only path.

    **V1 bootstrap is single-owner.** A new controller may only claim an engine
    whose label is *unmanaged* (``"default"`` on a fresh server). Adopting an
    engine already carrying an ``rc-*`` label would mean two controllers can
    write lifecycle state for one engine, which breaks "RolloutCore is the
    single lifecycle/weight writer". Recovering such an engine requires an
    explicit lease/recovery protocol that V1 does not implement, so it is
    refused rather than guessed at.
    """

    #: Raw ``weight_version`` string reported by ``GET /weight_info`` *after*
    #: bootstrap. Must be ``"rc-0"``.
    observed_engine_label: str
    #: Did ``POST /init_weight_transfer_engine`` return success?
    weight_transfer_initialised: bool
    #: What ``GET /weight_info`` reported *before* seeding. Required, and must be
    #: unmanaged -- this is the single-owner check.
    pre_seed_label: str | None
    #: Identity of the weights the engine currently holds.
    weight_identity: WeightIdentity
    #: Engine-side backend key the driver reported, e.g. ``"nccl"``; ``"none"``
    #: on the control-plane-only path.
    backend: str = "nccl"
    #: ``world_size_across_dp`` from ``GET /get_world_size``; None if unknown.
    world_size: int | None = None
    #: Which :class:`~rolloutcore.weight_transfer.WeightTransferDriver`
    #: initialized the transfer engine, or ``None`` when bootstrap ran
    #: control-plane-only. A controller with no driver can never legally install
    #: new weights, so this is recorded rather than inferred.
    weight_transfer_driver: str | None = None

    def failure_reason(self) -> str | None:
        if not self.weight_transfer_initialised and self.weight_transfer_driver is not None:
            return (
                f"weight transfer driver {self.weight_transfer_driver!r} reported an "
                "uninitialised transfer engine"
            )
        if self.weight_transfer_initialised and self.weight_transfer_driver is None:
            return (
                "bootstrap reports an initialised weight transfer engine but no "
                "driver to drive it; a bootstrapped controller must be able to "
                "account for how updates reach the engine"
            )

        if self.pre_seed_label is None:
            return (
                "bootstrap must report the pre-seed label; V1 is single-owner and "
                "does not adopt an engine it did not seed"
            )

        try:
            WeightVersion.parse_label(self.pre_seed_label)
        except ValueError:
            # Expected: the pre-seed label was unmanaged, so overwriting it is
            # legitimate bootstrap rather than reconciliation.
            return None
        return (
            f"refusing to claim engine already labelled {self.pre_seed_label!r}: "
            "another RolloutCore controller may own it (a lease/recovery protocol "
            "is required and V1 does not implement one)"
        )


@dataclass(frozen=True, slots=True)
class DrainEvidence(Evidence):
    """The engine's response to ``POST /pause?mode=wait&clear_cache=false``.

    Two distinct outcomes, deliberately separated:

    * ``engine_drain_completed is False`` -- the drain is still in flight. Not a
      failure, just early: the controller raises ``EvidenceNotReady`` and stays
      in DRAINING. Non-zero local active rollouts are entirely normal here.
    * ``engine_drain_completed is True`` -- the engine asserts it is quiesced, so
      the local active-rollout count *must* be zero. Any disagreement means one
      of the two bookkeeping systems is wrong, and we can no longer tell which,
      so the controller **taints** rather than retrying.

    Reference for what "completed" means on the engine side: the
    ``EngineCoreProc.pause_scheduler`` future resolves only when
    ``Scheduler.get_num_unfinished_requests()`` reports zero
    (``vllm/v1/core/sched/scheduler.py:2668-2669`` feeding ``has_work()`` at
    ``vllm/v1/engine/core.py:1457-1463``).
    """

    #: Did the drain call actually return successfully?
    engine_drain_completed: bool
    #: Optional engine-side active-request count, if the adapter can read one.
    #: A non-zero value alongside ``engine_drain_completed`` is self-contradictory.
    engine_active_requests: int | None = None
    #: Optional human note, e.g. "aborted 3 stragglers then re-paused".
    note: str = ""

    def failure_reason(self) -> str | None:
        if not self.engine_drain_completed:
            return "engine has not confirmed a completed drain"
        if self.engine_active_requests not in (None, 0):
            return (
                f"engine reports drain complete but also {self.engine_active_requests} "
                "active request(s); engine evidence is self-contradictory"
            )
        return None


@dataclass(frozen=True, slots=True)
class UpdateEvidence(Evidence):
    """Result of transferring and loading the new weights.

    Produced from the driver's
    :class:`~rolloutcore.weight_transfer.WeightTransferReport`. The driver owns
    ``/start_weight_update`` -> ``/update_weights`` -> ``/finish_weight_update``
    because they interleave with the trainer-side collective; RolloutCore
    assembles the evidence and checks it here.

    The engine's own commit point is ``AsyncLLM.finish_weight_update``
    (``vllm/v1/engine/async_llm.py:1284-1288``): it runs the worker-side
    ``finish_weight_update`` collective RPC and *then* writes the version.

    Critically, that RPC is the only thing vLLM does -- it invalidates no cache
    (``vllm/v1/worker/gpu_worker.py:1488-1505``; only ``reset_lora_state()``).
    Hence the separate INVALIDATING state.
    """

    #: What this update was supposed to install: generation *and* identity.
    target: UpdateTarget
    #: Did every chunk of ``POST /update_weights`` report success?
    weights_loaded: bool
    #: Did ``POST /finish_weight_update`` return success?
    finish_acknowledged: bool
    #: How many weight-transfer chunks the data plane carried. Diagnostic only.
    chunks_transferred: int | None = None
    #: Did the trainer-side NCCL/IPC data plane report completion?
    data_plane_complete: bool = True
    #: Manifest identity of the tensors the driver *actually staged*, computed
    #: from its own source metadata -- not from the target it was asked for. A
    #: mismatch against ``target.identity`` means the wrong checkpoint was
    #: pushed. ``None`` when the driver cannot compute one; the engine reports
    #: only an opaque version string, so there is no engine-side identity to read.
    observed_identity: WeightIdentity | None = None

    def failure_reason(self) -> str | None:
        if not self.weights_loaded:
            return "not all weight chunks were loaded"
        if not self.data_plane_complete:
            return "trainer-side weight data plane did not report completion"
        if not self.finish_acknowledged:
            return "finish_weight_update was not acknowledged by the engine"
        if self.observed_identity is not None and self.observed_identity != self.target.identity:
            return (
                f"driver staged weight source {self.observed_identity.describe()} but "
                f"{self.target.identity.describe()} was the target"
            )
        return None


@dataclass(frozen=True, slots=True)
class InvalidateEvidence(Evidence):
    """Proof that every cache holding previous-generation content is dropped.

    All three flags are required, and this is the **only** place a cycle drops
    caches. The drain deliberately pauses with ``clear_cache=false``: a clear
    before the mutation is not a correctness boundary -- it discards KV that is
    still valid at that instant -- and it would hide a failure here. The order
    (pause, mutate, invalidate, validate, resume) means nothing can be served
    between the mutation and this evidence, while the engine is still paused, so
    a cache that survives it is exactly the hazard Phase 4C measured: 32
    prefix-cache hits served across a weight change, producing different tokens.

    ``prefix_cache_reset`` comes from ``POST /reset_prefix_cache`` returning
    ``{"success": true}`` (``dev/cache/api_router.py:44``). Note the naming trap:
    the HTTP query parameter is ``reset_external`` but it fills the scheduler's
    ``reset_connector`` slot positionally.

    ``encoder_cache_reset`` and ``mm_cache_reset`` are **not** optional and are
    **not** covered by ``cache_salt``, which only reaches the prefix-cache block
    hash (``vllm/v1/core/kv_cache_utils.py:632-634``). This is the cache that
    ``finish_weight_update`` forgets: its only worker-side cleanup is
    ``reset_lora_state()`` and PR #48762 ("Invalidate encoder cache on
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
    """Pre-resume correctness evidence. **The engine is still paused here.**

    Phase 2 adapter produces this from ``GET /weight_info`` and
    ``GET /is_paused`` (``dev/rlhf/api_router.py:222``, ``:139``) *without*
    calling ``/resume``. Resume happens only in RESUMING, after this passes.

    Two reasons for the ordering:

    1. Fail-closed. If validation fails, the engine has never been resumed, so
       it cannot serve a rollout under weights we could not verify.
    2. A resume is itself an observable act. Doing it inside VALIDATING would
       mean VALIDATING sometimes leaves the engine serving and sometimes not,
       depending on where it failed.

    ``observed_engine_label`` is compared for **exact equality** with the
    target's label. A mismatch is a hard failure that taints: RolloutCore never
    "fixes" it by calling ``POST /update_weight_version``
    (``dev/rlhf/api_router.py:213``). Silently reconciling would mean publishing
    a version whose weights we cannot prove are installed -- the exact
    ``READY with partially updated weights`` state the plan forbids.
    """

    target: UpdateTarget
    #: Raw ``weight_version`` string reported by ``GET /weight_info``.
    observed_engine_label: str
    #: ``GET /is_paused``. Must be **True**: we have not resumed yet, so a
    #: not-paused engine means something else resumed it behind our back.
    is_paused: bool
    #: Did ``POST /reset_prefix_cache`` (or an equivalent) leave the caches clean
    #: *and still clean* at validation time? Adapter-supplied, optional.
    caches_still_clean: bool = True

    def failure_reason(self) -> str | None:
        if self.observed_engine_label != self.target.label:
            return (
                f"engine reports weight_version {self.observed_engine_label!r} but "
                f"{self.target.label!r} was expected; refusing to reconcile"
            )
        if not self.is_paused:
            return (
                "engine is not paused during validation; it was resumed outside "
                "RolloutCore's control"
            )
        if not self.caches_still_clean:
            return "caches were repopulated between invalidation and validation"
        return None


@dataclass(frozen=True, slots=True)
class ResumeEvidence(Evidence):
    """Result of issuing the resume, after validation has already passed.

    Phase 2 adapter produces this from ``POST /resume`` followed by
    ``GET /is_paused`` (``dev/rlhf/api_router.py:76``, ``:139``).
    """

    target: UpdateTarget
    #: Did ``POST /resume`` return success?
    resume_acknowledged: bool
    #: ``GET /is_paused`` after the resume. Must be False.
    is_paused: bool
    #: Re-read of ``GET /weight_info`` after resume, if the adapter re-checks.
    #: A post-resume version drift taints.
    observed_engine_label: str | None = None

    def failure_reason(self) -> str | None:
        if not self.resume_acknowledged:
            return "resume was not acknowledged by the engine"
        if self.is_paused:
            return "engine still reports paused after resume"
        if (
            self.observed_engine_label is not None
            and self.observed_engine_label != self.target.label
        ):
            return (
                f"engine reports weight_version {self.observed_engine_label!r} after "
                f"resume but {self.target.label!r} was committed"
            )
        return None


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One entry in the lifecycle journal.

    Exists so the benchmark work (drain latency, update latency, total downtime)
    has a substrate, and so tests can assert on the exact path taken.
    """

    seq: int
    from_state: str
    to_state: str
    event: str
    version: WeightVersion | None = None
    identity: WeightIdentity | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OrphanedRollout:
    """A rollout binding retained for forensics after the engine was tainted.

    Amendment 5: tainting must not silently discard active bindings. Whatever
    was in flight when the engine became untrustworthy is exactly the evidence
    an operator needs to decide what to do with those trajectories, so it is
    preserved (and marked unresolved) rather than cleared.
    """

    binding: Any  # RolloutBinding; Any avoids a circular import
    orphaned_in_state: str
    reason: str
    seq: int

    @property
    def request_id(self) -> str:
        return str(self.binding.request_id)

    @property
    def version(self) -> WeightVersion:
        return self.binding.version  # type: ignore[no-any-return]

    @property
    def weight_identity(self) -> WeightIdentity:
        return self.binding.weight_identity  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class CyclePlan:
    """Narration of what a transition expects the adapter to do next.

    ``steps`` is **human-readable documentation only**. It is written for logs,
    error messages and test failure output. Phase 2 adapters must call typed
    adapter methods (see ``rolloutcore.port.LifecycleAdapter``); parsing these
    strings would make the HTTP surface part of the controller's contract, which
    is precisely what the controller/adapter split exists to prevent.
    """

    target: UpdateTarget
    steps: tuple[str, ...] = field(default_factory=tuple)
    #: Machine-readable tags for logging/assertions. Never a dispatch mechanism.
    tags: tuple[str, ...] = field(default_factory=tuple)
