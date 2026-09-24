# SPDX-License-Identifier: Apache-2.0
"""The RolloutCore lifecycle state machine.

Pure Python, no I/O, no vLLM import. The controller owns *state*; an adapter
owns *effects*. Every transition is a method that either advances the state,
raises (recoverable), or taints (terminal).

Design rationale and the invariant-to-transition mapping live in
``docs/state-machine.md``.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Final

from .errors import (
    DrainDisagreementError,
    EngineTaintedError,
    EvidenceNotReady,
    IllegalTransitionError,
    InvariantViolation,
    NotServingError,
    UnknownRolloutError,
    VersionMismatchError,
)
from .evidence import (
    BootstrapEvidence,
    CyclePlan,
    DrainEvidence,
    Evidence,
    InvalidateEvidence,
    OrphanedRollout,
    ResumeEvidence,
    TransitionRecord,
    UpdateEvidence,
    ValidateEvidence,
)
from .versions import INITIAL_VERSION, UpdateTarget, WeightIdentity, WeightVersion


class LifecycleState(enum.StrEnum):
    """The nine lifecycle states.

    UNINITIALIZED -- constructed, not yet bound to a proven engine.
    READY         -- serving; rollouts may be admitted. The only admitting state.
    DRAINING      -- drain requested; active rollouts must reach zero.
    QUIESCED      -- drain proven: zero active work, engine paused.
    UPDATING      -- weights in flight over the native data plane.
    INVALIDATING  -- dropping every cache that may hold previous-version content.
    VALIDATING    -- proving the target version while **still paused**.
    RESUMING      -- resume issued; confirming the engine is actually serving.
    TAINTED       -- terminal in V1. Never resumed, never rolled back.
    """

    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    DRAINING = "DRAINING"
    QUIESCED = "QUIESCED"
    UPDATING = "UPDATING"
    INVALIDATING = "INVALIDATING"
    VALIDATING = "VALIDATING"
    RESUMING = "RESUMING"
    TAINTED = "TAINTED"


class Event(enum.StrEnum):
    """Every state-mutating event. Used to enumerate the transition matrix."""

    INITIALIZE = "initialize"
    BEGIN_DRAIN = "begin_drain"
    CONFIRM_DRAINED = "confirm_drained"
    BEGIN_UPDATE = "begin_update"
    CONFIRM_UPDATED = "confirm_updated"
    CONFIRM_INVALIDATED = "confirm_invalidated"
    CONFIRM_VALIDATED = "confirm_validated"
    CONFIRM_RESUMED = "confirm_resumed"
    TAINT = "taint"


#: The complete happy-path transition table. Every (state, event) pair not listed
#: here and not ``TAINT`` is illegal, and the test suite asserts exactly that for
#: the full cartesian product of states x events.
#:
#: Note the ordering guarantee: VALIDATING happens while the engine is still
#: paused, and the resume is issued only on entry to RESUMING. That is what makes
#: a validation failure fail *closed* -- at the point we discover something is
#: wrong, the engine has never served the new weights.
LEGAL_TRANSITIONS: Final[dict[tuple[LifecycleState, Event], LifecycleState]] = {
    (LifecycleState.UNINITIALIZED, Event.INITIALIZE): LifecycleState.READY,
    (LifecycleState.READY, Event.BEGIN_DRAIN): LifecycleState.DRAINING,
    (LifecycleState.DRAINING, Event.CONFIRM_DRAINED): LifecycleState.QUIESCED,
    (LifecycleState.QUIESCED, Event.BEGIN_UPDATE): LifecycleState.UPDATING,
    (LifecycleState.UPDATING, Event.CONFIRM_UPDATED): LifecycleState.INVALIDATING,
    (LifecycleState.INVALIDATING, Event.CONFIRM_INVALIDATED): LifecycleState.VALIDATING,
    (LifecycleState.VALIDATING, Event.CONFIRM_VALIDATED): LifecycleState.RESUMING,
    (LifecycleState.RESUMING, Event.CONFIRM_RESUMED): LifecycleState.READY,
}

#: States from which rollouts may be admitted. Exactly one, by design.
ADMITTING_STATES: Final[frozenset[LifecycleState]] = frozenset({LifecycleState.READY})

#: States in which a rollout may complete. DRAINING is included because that is
#: precisely what draining means; QUIESCED is not, because reaching it required
#: proving there were none left.
ROLLOUT_COMPLETION_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {LifecycleState.READY, LifecycleState.DRAINING}
)

#: States in which the engine is expected to be paused. Asserted by tests and
#: used by the Phase 2 adapter to reason about ordering.
PAUSED_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.DRAINING,
        LifecycleState.QUIESCED,
        LifecycleState.UPDATING,
        LifecycleState.INVALIDATING,
        LifecycleState.VALIDATING,
    }
)


@dataclass(frozen=True, slots=True)
class RolloutBinding:
    """A rollout bound to one generation *and* one weight identity, at admission.

    Frozen because invariant I2 says the binding cannot change afterwards. The
    engine will not enforce this for us: PR #49040 removed per-request version
    binding, so RolloutCore is the only place the binding exists.
    """

    request_id: str
    version: WeightVersion
    weight_identity: WeightIdentity
    cache_salt: str
    admitted_seq: int

    @property
    def target(self) -> UpdateTarget:
        return UpdateTarget(version=self.version, identity=self.weight_identity)


class LifecycleController:
    """Owns the lifecycle. The single lifecycle/weight writer.

    Deliberately exposes no setters for ``state``, ``current_target``,
    ``current_version``, ``current_identity``, ``pending_target``,
    ``active_rollouts`` or the journal: the only way to change them is to call a
    transition method. That is how "RolloutCore is the single lifecycle/weight
    writer" is enforced structurally rather than by convention.
    """

    __slots__ = (
        "_active",
        "_current_target",
        "_journal",
        "_orphaned",
        "_pending_target",
        "_seq",
        "_state",
        "_taint_reason",
        "_tainted_in_state",
    )

    def __init__(self) -> None:
        self._state: LifecycleState = LifecycleState.UNINITIALIZED
        self._current_target: UpdateTarget | None = None
        self._pending_target: UpdateTarget | None = None
        self._active: dict[str, RolloutBinding] = {}
        self._orphaned: list[OrphanedRollout] = []
        self._journal: list[TransitionRecord] = []
        self._seq: int = 0
        self._taint_reason: str | None = None
        self._tainted_in_state: str | None = None

    # ------------------------------------------------------------------ views

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def current_target(self) -> UpdateTarget | None:
        """The last *committed* target. ``None`` before successful bootstrap."""
        return self._current_target

    @property
    def current_version(self) -> WeightVersion | None:
        return self._current_target.version if self._current_target else None

    @property
    def current_identity(self) -> WeightIdentity | None:
        return self._current_target.identity if self._current_target else None

    @property
    def pending_target(self) -> UpdateTarget | None:
        """The in-flight target. Never published until RESUMING succeeds."""
        return self._pending_target

    @property
    def pending_version(self) -> WeightVersion | None:
        return self._pending_target.version if self._pending_target else None

    @property
    def active_rollouts(self) -> tuple[RolloutBinding, ...]:
        return tuple(self._active.values())

    @property
    def active_rollout_count(self) -> int:
        return len(self._active)

    @property
    def orphaned_rollouts(self) -> tuple[OrphanedRollout, ...]:
        """Bindings retained for forensics after a taint. Never discarded."""
        return tuple(self._orphaned)

    @property
    def journal(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._journal)

    @property
    def taint_reason(self) -> str | None:
        return self._taint_reason

    @property
    def tainted_in_state(self) -> str | None:
        """Which state the controller was in when it was tainted."""
        return self._tainted_in_state

    @property
    def is_tainted(self) -> bool:
        return self._state is LifecycleState.TAINTED

    @property
    def is_serving(self) -> bool:
        return self._state in ADMITTING_STATES

    # ------------------------------------------------------- transition core

    def _require_legal(self, event: Event) -> None:
        """Assert ``event`` is legal from the current state.

        Called at the top of every transition method, *before* any precondition
        or argument check. Legality is the more fundamental question: from an
        illegal state, "no pending target" is a consequence of the illegal
        state, not an independent error, and reporting it would send the caller
        looking in the wrong place.
        """
        if (self._state, event) not in LEGAL_TRANSITIONS:
            raise IllegalTransitionError(self._state.value, event.value)

    def _advance(
        self,
        event: Event,
        detail: str = "",
        target: UpdateTarget | None = None,
    ) -> None:
        """Apply a table-defined transition, or raise ``IllegalTransitionError``.

        The legality check happens *before* any state mutation, so a rejected
        event leaves the controller byte-for-byte unchanged.
        """
        next_state = LEGAL_TRANSITIONS.get((self._state, event))
        if next_state is None:
            raise IllegalTransitionError(self._state.value, event.value, detail)
        self._record(self._state, next_state, event, detail, target=target)
        self._state = next_state

    def _record(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        event: Event,
        detail: str = "",
        target: UpdateTarget | None = None,
    ) -> None:
        self._seq += 1
        effective = target or self._pending_target or self._current_target
        self._journal.append(
            TransitionRecord(
                seq=self._seq,
                from_state=from_state.value,
                to_state=to_state.value,
                event=event.value,
                version=effective.version if effective else None,
                identity=effective.identity if effective else None,
                detail=detail,
            )
        )

    def _taint(self, reason: str) -> EngineTaintedError:
        """Move to TAINTED, retaining active bindings as forensic state."""
        return self._taint_with(reason, None)

    def _taint_with(self, reason: str, error: EngineTaintedError | None) -> EngineTaintedError:
        """Taint, preserving forensic state, and return a specific error instance.

        Idempotent: a second taint keeps the *first* reason, because later ones
        are cascades of the root cause.
        """
        if self._state is LifecycleState.TAINTED:
            existing = EngineTaintedError(self._state.value, self._taint_reason or reason)
            return existing

        from_state = self._state
        self._taint_reason = reason
        self._tainted_in_state = from_state.value

        # Amendment 5: do NOT discard active bindings. Whatever was in flight
        # when the engine became untrustworthy is exactly the diagnostic
        # evidence an operator needs. Move them to the orphan list, unresolved.
        for binding in self._active.values():
            self._seq += 1
            self._orphaned.append(
                OrphanedRollout(
                    binding=binding,
                    orphaned_in_state=from_state.value,
                    reason=reason,
                    seq=self._seq,
                )
            )
        self._active.clear()

        self._record(from_state, LifecycleState.TAINTED, Event.TAINT, reason)
        self._state = LifecycleState.TAINTED
        self._pending_target = None

        if error is not None:
            return error
        return EngineTaintedError(from_state.value, reason)

    def _require(self, condition: bool, invariant: str, detail: str) -> None:
        if not condition:
            raise InvariantViolation(invariant, detail)

    def _check_evidence(self, evidence: Evidence) -> None:
        """Taint on any engine-side postcondition failure."""
        reason = evidence.failure_reason()
        if reason is not None:
            raise self._taint(reason)

    # ----------------------------------------------------------- transitions

    def initialize(self, evidence: BootstrapEvidence) -> None:
        """UNINITIALIZED -> READY. Binds the controller to a proven fresh engine.

        V1 bootstrap is **single-owner**: the engine must have carried an
        unmanaged label (``"default"``) which the adapter then seeded to
        ``rc-0``. Adopting an engine that already reports ``rc-*`` is refused --
        see ``BootstrapEvidence``.
        """
        self._require_legal(Event.INITIALIZE)
        self._check_evidence(evidence)

        try:
            observed = WeightVersion.parse_label(evidence.observed_engine_label)
        except ValueError as exc:
            raise self._taint(
                f"engine does not report a RolloutCore-managed weight_version: {exc}"
            ) from exc

        if observed != WeightVersion(INITIAL_VERSION):
            raise self._taint(
                f"engine reports {observed.label!r} at bootstrap; a fresh engine must "
                f"report {WeightVersion(INITIAL_VERSION).label!r}"
            )

        self._advance(Event.INITIALIZE, detail=f"backend={evidence.backend}")
        self._current_target = UpdateTarget(version=observed, identity=evidence.weight_identity)

    def begin_drain(self) -> CyclePlan:
        """READY -> DRAINING. Requests the quiesce.

        Maps to ``POST /pause?mode=wait&clear_cache=true``. ``mode="keep"`` is
        deliberately unreachable from this API: it freezes requests in place and
        lets a single response span two weight versions, violating I2.
        """
        self._require_legal(Event.BEGIN_DRAIN)
        self._advance(Event.BEGIN_DRAIN)
        assert self._current_target is not None
        return CyclePlan(
            target=self._current_target,
            steps=("POST /pause?mode=wait&clear_cache=true",),
            tags=("pause", "drain"),
        )

    def confirm_drained(self, evidence: DrainEvidence) -> None:
        """DRAINING -> QUIESCED. Requires proof of zero active work.

        Amendment 2, precisely:

        * ``engine_drain_completed is False`` -> ``EvidenceNotReady``
          (retryable). Non-zero local active rollouts are normal here; the drain
          has simply not finished.
        * ``engine_drain_completed is True`` while local active rollouts remain
          -> **taint**. The engine asserts quiescence while RolloutCore counts
          live work; one of them is wrong and we cannot tell which.
        """
        self._require_legal(Event.CONFIRM_DRAINED)

        if not evidence.engine_drain_completed:
            raise EvidenceNotReady(
                "I5-DRAIN",
                "engine drain is still in flight; "
                f"{len(self._active)} local rollout(s) outstanding is expected here",
            )

        # The engine asserts it is drained, so local bookkeeping must agree.
        if self._active:
            ids = sorted(self._active)
            raise self._taint_with(
                f"engine reported a completed drain but RolloutCore still counts "
                f"{len(ids)} active rollout(s) ({', '.join(ids)})",
                DrainDisagreementError(self._state.value, len(ids), ids),
            )

        self._check_evidence(evidence)
        self._advance(Event.CONFIRM_DRAINED, detail=evidence.note)

    def begin_update(self, target: UpdateTarget) -> CyclePlan:
        """QUIESCED -> UPDATING. Opens a weight-transfer session.

        Maps to ``POST /start_weight_update``. ``target.version`` must be
        strictly greater than the committed generation; the identity rides along
        so a trajectory can later claim exact weights, not merely "generation N".
        """
        self._require_legal(Event.BEGIN_UPDATE)
        self._require(
            self._current_target is not None,
            "I1-ATOMICITY",
            "cannot begin an update before a target has been committed",
        )
        assert self._current_target is not None  # narrowed for type checkers
        self._require(
            target.version > self._current_target.version,
            "I1-MONOTONIC",
            f"target version {target.label!r} must be greater than committed "
            f"{self._current_target.version.label!r}",
        )
        # Advance first: a rejected event must not leave pending_target set.
        self._advance(Event.BEGIN_UPDATE, detail=f"target={target.describe()}", target=target)
        self._pending_target = target
        return CyclePlan(
            target=target,
            steps=(
                "POST /start_weight_update",
                "POST /update_weights (xN; the data plane runs out of band)",
            ),
            tags=("start_weight_update", "update_weights"),
        )

    def confirm_updated(self, evidence: UpdateEvidence) -> CyclePlan:
        """UPDATING -> INVALIDATING. The weights are installed and finalized.

        Maps to ``POST /finish_weight_update {"weight_version": "<target>"}``.

        The engine's ``finish_weight_update`` RPC invalidates **nothing**
        (``gpu_worker.py:1488-1505``), which is why this transition leads to
        INVALIDATING rather than back to READY.
        """
        self._require_legal(Event.CONFIRM_UPDATED)
        pending = self._pending_target
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending target; begin_update must precede confirm_updated",
        )
        assert pending is not None
        if evidence.target != pending:
            # We can no longer say which target the engine holds -> unknown update.
            raise self._taint(
                f"update evidence targets {evidence.target.describe()} but "
                f"{pending.describe()} is pending; engine version is now ambiguous"
            )
        self._check_evidence(evidence)
        self._advance(
            Event.CONFIRM_UPDATED,
            detail=f"chunks={evidence.chunks_transferred}",
            target=pending,
        )
        return CyclePlan(
            target=pending,
            steps=(
                "POST /reset_prefix_cache?reset_running_requests=true&reset_external=true",
                "POST /reset_encoder_cache",
                "POST /reset_mm_cache",
            ),
            tags=("reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"),
        )

    def confirm_invalidated(self, evidence: InvalidateEvidence) -> CyclePlan:
        """INVALIDATING -> VALIDATING. Every previous-generation cache is gone.

        Maps to the ``/reset_*`` triple. All three flags are mandatory: prefix
        KV, encoder cache and multimodal cache are independent structures, and
        ``cache_salt`` only isolates the first of them.

        The returned plan still does **not** resume: validation runs against a
        paused engine.
        """
        self._require_legal(Event.CONFIRM_INVALIDATED)
        pending = self._pending_target
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending target; cannot invalidate",
        )
        assert pending is not None
        self._check_evidence(evidence)
        self._advance(Event.CONFIRM_INVALIDATED, target=pending)
        return CyclePlan(
            target=pending,
            steps=(
                "GET /weight_info  (expect the target label)",
                "GET /is_paused    (expect still paused -- do NOT resume yet)",
            ),
            tags=("get_weight_info", "get_is_paused"),
        )

    def confirm_validated(self, evidence: ValidateEvidence) -> CyclePlan:
        """VALIDATING -> RESUMING. Correctness proven **while still paused**.

        This is the fail-closed gate. If anything here fails, the engine has
        never been resumed and therefore cannot serve a rollout under weights we
        could not verify. The resume is issued only on entry to RESUMING.

        A version mismatch is a hard failure that taints. RolloutCore does not
        call ``POST /update_weight_version`` to reconcile: doing so would publish
        a version whose weights we cannot prove are installed.
        """
        self._require_legal(Event.CONFIRM_VALIDATED)
        pending = self._pending_target
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending target; cannot validate",
        )
        assert pending is not None

        if evidence.target != pending:
            raise self._taint(
                f"validation evidence targets {evidence.target.describe()} but "
                f"{pending.describe()} is pending"
            )

        reason = evidence.failure_reason()
        if reason is not None:
            if evidence.observed_engine_label != pending.label:
                raise self._taint_with(
                    reason,
                    VersionMismatchError(
                        self._state.value, pending.version, evidence.observed_engine_label
                    ),
                )
            raise self._taint(reason)

        self._advance(
            Event.CONFIRM_VALIDATED,
            detail=f"validated while paused: {pending.describe()}",
            target=pending,
        )
        return CyclePlan(
            target=pending,
            steps=("POST /resume", "GET /is_paused  (expect false)"),
            tags=("resume", "get_is_paused"),
        )

    def confirm_resumed(self, evidence: ResumeEvidence) -> None:
        """RESUMING -> READY. The engine is serving. **The commit point.**

        This is the only place ``current_target`` changes (invariant I1).
        Requires resume acknowledgement *and* ``is_paused == false``.
        """
        self._require_legal(Event.CONFIRM_RESUMED)
        pending = self._pending_target
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending target; cannot confirm resume",
        )
        assert pending is not None

        if evidence.target != pending:
            raise self._taint(
                f"resume evidence targets {evidence.target.describe()} but "
                f"{pending.describe()} is pending"
            )

        reason = evidence.failure_reason()
        if reason is not None:
            if (
                evidence.observed_engine_label is not None
                and evidence.observed_engine_label != pending.label
            ):
                raise self._taint_with(
                    reason,
                    VersionMismatchError(
                        self._state.value, pending.version, evidence.observed_engine_label
                    ),
                )
            raise self._taint(reason)

        self._advance(
            Event.CONFIRM_RESUMED, detail=f"published={pending.describe()}", target=pending
        )
        self._current_target = pending
        self._pending_target = None

    def taint(self, reason: str) -> None:
        """Move to TAINTED from anywhere. Used for operator aborts and faults.

        Idempotent. TAINTED is absorbing: V1 never rolls back and never resumes.
        Active bindings are retained as orphans, not discarded.
        """
        self._taint(reason)

    # ------------------------------------------------------ rollout admission

    def next_target(self, identity: WeightIdentity | None = None) -> UpdateTarget:
        """Build the target for the next generation.

        ``identity`` defaults to the current identity, which is only appropriate
        for a no-op or fake update. A real update must pass the identity of the
        weights it actually transfers, or I4 would claim provenance it cannot
        support.
        """
        self._require(
            self._current_target is not None,
            "I1-ATOMICITY",
            "cannot compute a next target before one has been committed",
        )
        assert self._current_target is not None
        return UpdateTarget(
            version=self._current_target.version.next(),
            identity=identity or self._current_target.identity,
        )

    def admit_rollout(self, request_id: str) -> RolloutBinding:
        """Bind a new rollout to the current committed target.

        Only legal in READY. During any transition the request is **rejected**
        rather than queued: the engine's own queue would schedule it after
        resume under the *new* version while RolloutCore had bound it to the old
        one, producing exactly the mixed-version response I2 forbids.
        """
        if self._state not in ADMITTING_STATES:
            raise NotServingError(self._state.value)
        self._require(
            self._current_target is not None,
            "I2-BINDING",
            "cannot admit a rollout before a target has been committed",
        )
        self._require(
            request_id not in self._active,
            "I2-BINDING",
            f"rollout {request_id!r} is already active",
        )
        assert self._current_target is not None
        current = self._current_target
        self._seq += 1
        binding = RolloutBinding(
            request_id=request_id,
            version=current.version,
            weight_identity=current.identity,
            cache_salt=current.version.cache_salt,
            admitted_seq=self._seq,
        )
        self._active[request_id] = binding
        self._journal.append(
            TransitionRecord(
                seq=self._seq,
                from_state=self._state.value,
                to_state=self._state.value,
                event="admit_rollout",
                version=binding.version,
                identity=binding.weight_identity,
                detail=request_id,
            )
        )
        return binding

    def finish_rollout(self, request_id: str) -> RolloutBinding:
        """Release a completed rollout. Legal in READY and DRAINING."""
        return self._release_rollout(request_id, event="finish_rollout")

    def abort_rollout(self, request_id: str) -> RolloutBinding:
        """Release an aborted rollout. Legal in READY and DRAINING."""
        return self._release_rollout(request_id, event="abort_rollout")

    def _release_rollout(self, request_id: str, *, event: str) -> RolloutBinding:
        if self._state not in ROLLOUT_COMPLETION_STATES:
            raise IllegalTransitionError(
                self._state.value,
                event,
                f"rollouts may only complete in "
                f"{sorted(s.value for s in ROLLOUT_COMPLETION_STATES)}",
            )
        binding = self._active.pop(request_id, None)
        if binding is None:
            raise UnknownRolloutError(request_id)
        self._seq += 1
        self._journal.append(
            TransitionRecord(
                seq=self._seq,
                from_state=self._state.value,
                to_state=self._state.value,
                event=event,
                version=binding.version,
                identity=binding.weight_identity,
                detail=request_id,
            )
        )
        return binding


def enumerate_transition_matrix() -> list[tuple[LifecycleState, Event]]:
    """Every (state, event) pair. The illegal-transition tests iterate this."""
    return list(itertools.product(LifecycleState, Event))
