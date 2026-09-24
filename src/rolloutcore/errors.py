# SPDX-License-Identifier: Apache-2.0
"""Errors, split by *recoverability*.

This split is load-bearing, not cosmetic:

* ``InvariantViolation`` -- RolloutCore's own bookkeeping says the transition was
  attempted too early. Nothing external is known to be wrong, so the state is
  left unchanged and the caller can retry once the precondition holds.
  Example: confirming a drain while rollouts are still active.

* ``EngineTaintedError`` -- the engine reported something that contradicts a
  required postcondition, so its state is no longer trustworthy. The controller
  moves to TAINTED, which is absorbing; V1 never rolls back or resumes from it.

Conflating the two would either (a) taint a perfectly healthy engine because a
caller was early, or (b) continue serving from an engine whose state we can no
longer prove. Both are worse than two exception types.
"""

from __future__ import annotations

from .versions import WeightVersion


class RolloutCoreError(Exception):
    """Base class for every error raised by RolloutCore."""


class InvariantViolation(RolloutCoreError):
    """A precondition failed. State is unchanged and the call may be retried."""

    def __init__(self, invariant: str, detail: str) -> None:
        self.invariant = invariant
        self.detail = detail
        super().__init__(f"[{invariant}] {detail}")


class EvidenceNotReady(InvariantViolation):
    """The observation is simply incomplete — the caller was early.

    Distinct from both ``InvariantViolation`` (which usually means RolloutCore's
    own bookkeeping disagrees with itself) and ``EngineTaintedError`` (which
    means the engine can no longer be trusted).

    The canonical case is confirming a drain while ``/pause?mode=wait`` is still
    in flight. That is normal operation: the caller should wait and re-confirm,
    not abandon the engine. Once the engine *does* report a completed drain,
    however, an outstanding local count is a genuine disagreement and taints --
    see ``docs/state-machine.md`` section 4.
    """

    def __init__(self, invariant: str, detail: str) -> None:
        super().__init__(invariant, f"not ready: {detail}")


class IllegalTransitionError(InvariantViolation):
    """The requested event does not exist from the current state."""

    def __init__(self, state: str, event: str, detail: str = "") -> None:
        self.state = state
        self.event = event
        message = f"event {event!r} is not legal from state {state}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__("T-ILLEGAL", message)


class EngineTaintedError(RolloutCoreError):
    """The engine entered TAINTED. Terminal in V1."""

    def __init__(self, state: str, reason: str) -> None:
        self.state = state
        self.reason = reason
        super().__init__(
            f"engine tainted in state {state}: {reason}. "
            "TAINTED is terminal in V1: restart the engine and construct a new controller."
        )


class NotServingError(InvariantViolation):
    """Rollout admission was attempted while not READY.

    This is the mechanism behind "no new request can accidentally run through
    half-updated weights". Note it is an ``InvariantViolation``, not a taint:
    rejecting a request during a lifecycle transition is normal operation, not
    an engine fault.
    """

    def __init__(self, state: str) -> None:
        super().__init__(
            "I2-ADMISSION",
            f"rollouts may only be admitted in READY; controller is {state}",
        )


class UnknownRolloutError(InvariantViolation):
    """A rollout id was completed that was never admitted, or was already completed."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(
            "I2-BINDING",
            f"no active rollout with id {request_id!r}",
        )


class VersionMismatchError(EngineTaintedError):
    """The engine's reported version is not the one RolloutCore published.

    Hard failure by design: RolloutCore never auto-reconciles by calling
    ``POST /update_weight_version``.
    """

    def __init__(self, state: str, expected: WeightVersion, observed: str) -> None:
        self.expected = expected
        self.observed = observed
        super().__init__(
            state,
            f"expected engine weight_version {expected.label!r}, observed {observed!r}; "
            "automatic reconciliation is disabled",
        )


class DrainDisagreementError(EngineTaintedError):
    """The engine reported a completed drain while rollouts were still active.

    Amendment 2: this is a disagreement between two bookkeeping systems, not a
    "wait a bit longer" situation. One of them is wrong and we cannot tell
    which, so the engine is tainted rather than retried.
    """

    def __init__(self, state: str, active_rollouts: int, request_ids: list[str]) -> None:
        self.active_rollouts = active_rollouts
        self.request_ids = request_ids
        super().__init__(
            state,
            f"engine reported a completed drain but RolloutCore still counts "
            f"{active_rollouts} active rollout(s) ({', '.join(request_ids)}); "
            "engine and controller bookkeeping disagree",
        )
