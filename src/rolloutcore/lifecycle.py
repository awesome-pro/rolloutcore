# SPDX-License-Identifier: Apache-2.0
"""The RolloutCore lifecycle state machine.

Pure Python, no I/O, no vLLM import. The controller owns *state*; an adapter
(Phase 2) owns *effects*. Every transition is a method that either advances the
state, raises ``InvariantViolation`` (caller was early; retryable), or raises
``EngineTaintedError`` (engine state no longer provable; terminal).

Design rationale and the invariant-to-transition mapping live in
``docs/state-machine.md``.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass
from typing import Final

from .errors import (
    EngineTaintedError,
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
    TransitionRecord,
    UpdateEvidence,
    ValidateEvidence,
)
from .versions import INITIAL_VERSION, WeightVersion


class LifecycleState(enum.StrEnum):
    """The eight lifecycle states.

    UNINITIALIZED -- constructed, not yet bound to a proven engine.
    READY         -- serving; rollouts may be admitted. The only admitting state.
    DRAINING      -- drain requested; active rollouts must reach zero.
    QUIESCED      -- drain proven: zero active work, engine paused.
    UPDATING      -- weights in flight over the native data plane.
    INVALIDATING  -- dropping every cache that may hold previous-version content.
    VALIDATING    -- proving the engine serves the target version before publishing.
    TAINTED       -- terminal in V1. Never resumed, never rolled back.
    """

    UNINITIALIZED = "UNINITIALIZED"
    READY = "READY"
    DRAINING = "DRAINING"
    QUIESCED = "QUIESCED"
    UPDATING = "UPDATING"
    INVALIDATING = "INVALIDATING"
    VALIDATING = "VALIDATING"
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
    TAINT = "taint"


#: The complete happy-path transition table. Every (state, event) pair not listed
#: here and not ``TAINT`` is illegal, and the test suite asserts exactly that for
#: the full cartesian product of states x events.
LEGAL_TRANSITIONS: Final[dict[tuple[LifecycleState, Event], LifecycleState]] = {
    (LifecycleState.UNINITIALIZED, Event.INITIALIZE): LifecycleState.READY,
    (LifecycleState.READY, Event.BEGIN_DRAIN): LifecycleState.DRAINING,
    (LifecycleState.DRAINING, Event.CONFIRM_DRAINED): LifecycleState.QUIESCED,
    (LifecycleState.QUIESCED, Event.BEGIN_UPDATE): LifecycleState.UPDATING,
    (LifecycleState.UPDATING, Event.CONFIRM_UPDATED): LifecycleState.INVALIDATING,
    (LifecycleState.INVALIDATING, Event.CONFIRM_INVALIDATED): LifecycleState.VALIDATING,
    (LifecycleState.VALIDATING, Event.CONFIRM_VALIDATED): LifecycleState.READY,
}

#: States from which rollouts may be admitted. Exactly one, by design.
ADMITTING_STATES: Final[frozenset[LifecycleState]] = frozenset({LifecycleState.READY})

#: States in which a rollout may complete. DRAINING is included because that is
#: precisely what draining means; QUIESCED is not, because reaching it required
#: proving there were none left.
ROLLOUT_COMPLETION_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {LifecycleState.READY, LifecycleState.DRAINING}
)


@dataclass(frozen=True, slots=True)
class RolloutBinding:
    """A rollout bound to exactly one committed version, fixed at admission.

    Frozen because invariant I2 says the binding cannot change afterwards. The
    engine will not enforce this for us: PR #49040 removed per-request version
    binding, so RolloutCore is the only place the binding exists.
    """

    request_id: str
    version: WeightVersion
    cache_salt: str
    admitted_seq: int


class LifecycleController:
    """Owns the lifecycle. The single lifecycle/weight writer.

    Deliberately exposes no setters for ``state``, ``current_version``,
    ``pending_version`` or ``active_rollouts``: the only way to change them is to
    call a transition method. That is how "RolloutCore is the single
    lifecycle/weight writer" is enforced structurally rather than by convention.
    """

    __slots__ = (
        "_state",
        "_current_version",
        "_pending_version",
        "_active",
        "_journal",
        "_seq",
        "_taint_reason",
    )

    def __init__(self) -> None:
        self._state: LifecycleState = LifecycleState.UNINITIALIZED
        self._current_version: WeightVersion | None = None
        self._pending_version: WeightVersion | None = None
        self._active: dict[str, RolloutBinding] = {}
        self._journal: list[TransitionRecord] = []
        self._seq: int = 0
        self._taint_reason: str | None = None

    # ------------------------------------------------------------------ views

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def current_version(self) -> WeightVersion | None:
        """The last *committed* version. ``None`` before successful bootstrap."""
        return self._current_version

    @property
    def pending_version(self) -> WeightVersion | None:
        """The in-flight target. Never published until VALIDATING succeeds."""
        return self._pending_version

    @property
    def active_rollouts(self) -> tuple[RolloutBinding, ...]:
        return tuple(self._active.values())

    @property
    def active_rollout_count(self) -> int:
        return len(self._active)

    @property
    def journal(self) -> tuple[TransitionRecord, ...]:
        return tuple(self._journal)

    @property
    def taint_reason(self) -> str | None:
        return self._taint_reason

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
        illegal state, "no pending version" is a consequence of the illegal
        state, not an independent error, and reporting it would send the caller
        looking in the wrong place.
        """
        if (self._state, event) not in LEGAL_TRANSITIONS:
            raise IllegalTransitionError(self._state.value, event.value)

    def _advance(
        self, event: Event, detail: str = "", version: WeightVersion | None = None
    ) -> None:
        """Apply a table-defined transition, or raise ``IllegalTransitionError``.

        The legality check happens *before* any state mutation, so a rejected
        event leaves the controller byte-for-byte unchanged.
        """
        target = LEGAL_TRANSITIONS.get((self._state, event))
        if target is None:
            raise IllegalTransitionError(self._state.value, event.value, detail)
        self._record(self._state, target, event, detail, version=version)
        self._state = target

    def _record(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        event: Event,
        detail: str = "",
        version: WeightVersion | None = None,
    ) -> None:
        self._seq += 1
        self._journal.append(
            TransitionRecord(
                seq=self._seq,
                from_state=from_state.value,
                to_state=to_state.value,
                event=event.value,
                version=version or self._pending_version or self._current_version,
                detail=detail,
            )
        )

    def _taint(self, reason: str) -> EngineTaintedError:
        """Move to TAINTED and return the error to raise.

        Idempotent: a second taint keeps the *first* reason, because later ones
        are cascades of the root cause.
        """
        if self._state is LifecycleState.TAINTED:
            return EngineTaintedError(self._state.value, self._taint_reason or reason)
        from_state = self._state
        self._taint_reason = reason
        self._record(from_state, LifecycleState.TAINTED, Event.TAINT, reason)
        self._state = LifecycleState.TAINTED
        self._pending_version = None
        self._active.clear()
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
        """UNINITIALIZED -> READY. Binds the controller to a proven engine.

        Bootstrap is the **only** place RolloutCore may write a version label
        without an update having occurred, and only over an *unmanaged* label.
        After this call, a version mismatch is always a hard failure.
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
        self._current_version = observed

    def begin_drain(self) -> CyclePlan:
        """READY -> DRAINING. Requests the quiesce.

        Maps to ``POST /pause?mode=wait&clear_cache=true``. ``mode="keep"`` is
        deliberately not reachable from this API: it freezes requests in place
        and lets a single response span two weight versions, which violates I2.
        """
        self._require_legal(Event.BEGIN_DRAIN)
        self._advance(Event.BEGIN_DRAIN)
        return CyclePlan(
            target_version=(self._current_version or WeightVersion(0)).next(),
            steps=("POST /pause?mode=wait&clear_cache=true",),
        )

    def confirm_drained(self, evidence: DrainEvidence) -> None:
        """DRAINING -> QUIESCED. Requires *proof* of zero active work.

        Two independent checks, with deliberately different failure modes:

        1. Local bookkeeping must show zero active rollouts. A non-zero count
           means the drain simply has not finished -- the engine is fine and the
           caller should wait, so this raises ``InvariantViolation`` and stays in
           DRAINING.
        2. The evidence must confirm the engine-side pause completed. If it did
           not, the engine's state is unprovable, so this taints.
        """
        self._require_legal(Event.CONFIRM_DRAINED)
        self._require(
            not self._active,
            "I5-DRAIN",
            f"cannot confirm drain with {len(self._active)} rollout(s) still active: "
            f"{sorted(self._active)}",
        )
        self._check_evidence(evidence)
        self._advance(Event.CONFIRM_DRAINED, detail=evidence.note)

    def begin_update(self, target: WeightVersion) -> CyclePlan:
        """QUIESCED -> UPDATING. Opens a weight-transfer session.

        Maps to ``POST /start_weight_update``. ``target`` must be strictly
        greater than the committed version: re-publishing the current version is
        not an update, and downgrades are out of scope for V1.
        """
        self._require_legal(Event.BEGIN_UPDATE)
        self._require(
            self._current_version is not None,
            "I1-ATOMICITY",
            "cannot begin an update before a version has been committed",
        )
        assert self._current_version is not None  # narrowed for type checkers
        self._require(
            target > self._current_version,
            "I1-MONOTONIC",
            f"target version {target.label!r} must be greater than committed "
            f"{self._current_version.label!r}",
        )
        # Advance first: a rejected event must not leave pending_version set.
        self._advance(Event.BEGIN_UPDATE, detail=f"target={target.label}", version=target)
        self._pending_version = target
        return CyclePlan(
            target_version=target,
            steps=(
                "POST /start_weight_update",
                "POST /update_weights (xN, data plane runs out of band)",
            ),
        )

    def confirm_updated(self, evidence: UpdateEvidence) -> CyclePlan:
        """UPDATING -> INVALIDATING. The weights are installed and finalized.

        Maps to ``POST /finish_weight_update {"weight_version": "<target>"}``.

        The engine's ``finish_weight_update`` RPC invalidates **nothing**
        (``gpu_worker.py:1488-1505``), which is why this transition must lead to
        INVALIDATING rather than back to READY.
        """
        self._require_legal(Event.CONFIRM_UPDATED)
        pending = self._pending_version
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending version; begin_update must precede confirm_updated",
        )
        assert pending is not None
        if evidence.target_version != pending:
            # We can no longer say which version the engine holds -> unknown update.
            raise self._taint(
                f"update evidence targets {evidence.target_version.label!r} but "
                f"{pending.label!r} is pending; engine version is now ambiguous"
            )
        self._check_evidence(evidence)
        self._advance(Event.CONFIRM_UPDATED, detail=f"chunks={evidence.chunks_transferred}")
        return CyclePlan(
            target_version=pending,
            steps=(
                "POST /reset_prefix_cache?reset_running_requests=true&reset_external=true",
                "POST /reset_encoder_cache",
                "POST /reset_mm_cache",
            ),
        )

    def confirm_invalidated(self, evidence: InvalidateEvidence) -> CyclePlan:
        """INVALIDATING -> VALIDATING. Every version-N cache is provably gone.

        Maps to the ``/reset_*`` triple. All three flags are mandatory: prefix
        KV, encoder cache and multimodal cache are independent structures, and
        ``cache_salt`` only isolates the first of them.
        """
        self._require_legal(Event.CONFIRM_INVALIDATED)
        pending = self._pending_version
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending version; cannot invalidate",
        )
        assert pending is not None
        self._check_evidence(evidence)
        self._advance(Event.CONFIRM_INVALIDATED)
        return CyclePlan(
            target_version=pending,
            steps=(
                "POST /resume",
                "GET /weight_info",
                "GET /is_paused",
            ),
        )

    def confirm_validated(self, evidence: ValidateEvidence) -> None:
        """VALIDATING -> READY. Publishes the target version. The commit point.

        This is the *only* place ``current_version`` changes (invariant I1).

        A version mismatch is a hard failure that taints. RolloutCore does not
        call ``POST /update_weight_version`` to reconcile: doing so would publish
        a version whose weights we cannot prove are installed.
        """
        self._require_legal(Event.CONFIRM_VALIDATED)
        pending = self._pending_version
        self._require(
            pending is not None,
            "I1-ATOMICITY",
            "no pending version; cannot validate",
        )
        assert pending is not None

        if evidence.target_version != pending:
            raise self._taint(
                f"validation evidence targets {evidence.target_version.label!r} but "
                f"{pending.label!r} is pending"
            )

        reason = evidence.failure_reason()
        if reason is not None:
            if evidence.observed_engine_label != pending.label:
                raise self._taint_version_mismatch(pending, evidence.observed_engine_label)
            raise self._taint(reason)

        self._advance(Event.CONFIRM_VALIDATED, detail=f"published={pending.label}")
        self._current_version = pending
        self._pending_version = None

    def _taint_version_mismatch(
        self, expected: WeightVersion, observed: str
    ) -> EngineTaintedError:
        """Taint with the more specific ``VersionMismatchError``."""
        if self._state is LifecycleState.TAINTED:
            return EngineTaintedError(self._state.value, self._taint_reason or "mismatch")
        from_state = self._state
        self._taint_reason = (
            f"expected engine weight_version {expected.label!r}, observed {observed!r}; "
            "automatic reconciliation is disabled"
        )
        self._record(from_state, LifecycleState.TAINTED, Event.TAINT, self._taint_reason)
        self._state = LifecycleState.TAINTED
        self._pending_version = None
        self._active.clear()
        return VersionMismatchError(from_state.value, expected, observed)

    def taint(self, reason: str) -> None:
        """Move to TAINTED from anywhere. Used for operator aborts and faults.

        Idempotent. TAINTED is absorbing: V1 never rolls back and never resumes.
        """
        self._taint(reason)

    # ------------------------------------------------------ rollout admission

    def admit_rollout(self, request_id: str) -> RolloutBinding:
        """Bind a new rollout to the current committed version.

        Only legal in READY. During any transition the request is **rejected**
        rather than queued: the engine's own queue would schedule it after
        resume under the *new* version while RolloutCore had bound it to the old
        one, producing exactly the mixed-version response I2 forbids.
        """
        if self._state not in ADMITTING_STATES:
            raise NotServingError(self._state.value)
        self._require(
            self._current_version is not None,
            "I2-BINDING",
            "cannot admit a rollout before a version has been committed",
        )
        self._require(
            request_id not in self._active,
            "I2-BINDING",
            f"rollout {request_id!r} is already active",
        )
        assert self._current_version is not None
        version = self._current_version
        self._seq += 1
        binding = RolloutBinding(
            request_id=request_id,
            version=version,
            cache_salt=version.cache_salt,
            admitted_seq=self._seq,
        )
        self._active[request_id] = binding
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
                f"rollouts may only complete in {sorted(s.value for s in ROLLOUT_COMPLETION_STATES)}",
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
                detail=request_id,
            )
        )
        return binding


def enumerate_transition_matrix() -> list[tuple[LifecycleState, Event]]:
    """Every (state, event) pair. The illegal-transition tests iterate this."""
    return list(itertools.product(LifecycleState, Event))
