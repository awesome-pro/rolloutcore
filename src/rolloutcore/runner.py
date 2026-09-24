# SPDX-License-Identifier: Apache-2.0
"""Drives a controller through a cycle using typed adapter methods.

The runner is the only place that knows the *order* of adapter calls. The
controller knows which orders are legal; the adapter knows how to perform each
step; the runner joins them.

It never inspects ``CyclePlan.steps``.

Failure classification (review item 3)
--------------------------------------

An external effect can fail *ambiguously*: once ``POST /start_weight_update``
has been attempted, ``worker 0 initialized reload / worker 1 failed`` is
indistinguishable from ``nothing happened``. The controller would otherwise sit
in ``UPDATING`` believing it knows something it does not, which is the one thing
RolloutCore exists to prevent. So every call is classified on purpose:

============================  ==========================================
``adapter.bootstrap``         refuse-before-write = retryable; anything else taints
``adapter.begin_drain``       mutating (it issues the pause) -> taints
``adapter.await_drain``       observation -> propagates, retryable
``adapter.start_weight_update`` mutating -> taints
``adapter.complete_weight_update`` mutating, interleaved with a collective -> taints
``adapter.invalidate_caches``  mutating -> taints
``adapter.validate_pre_resume`` observation, engine still paused -> propagates
``adapter.resume``            mutating -> taints
============================  ==========================================

"Observation" does not mean "harmless": a raised :class:`RolloutCoreError` is
re-raised untouched from either path, because RolloutCore's own errors already
say whether the engine was tainted. Only *foreign* exceptions (a transport
error, a driver fault) are ambiguous, and only mutating calls translate those
into a taint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from .errors import AlreadyManagedEngineError, EvidenceNotReady, RolloutCoreError
from .evidence import DrainEvidence
from .lifecycle import LifecycleController, LifecycleState
from .port import LifecycleAdapter
from .versions import UpdateTarget, WeightIdentity

#: How many times to re-poll a drain before giving up. The engine's own drain
#: call has no timeout (``vllm/v1/engine/core_client.py:996-1002``), so a bound
#: has to live here.
DEFAULT_DRAIN_POLLS = 600

#: Seconds between drain polls.
DEFAULT_DRAIN_INTERVAL = 0.5

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CycleResult:
    """What one cycle did, for logs, tests and the eventual benchmark."""

    committed: UpdateTarget
    states_visited: tuple[LifecycleState, ...]
    drain_polls: int
    duration_s: float
    detail: dict[str, float] = field(default_factory=dict)


class LifecycleRunner:
    """Runs bootstrap and update cycles over a :class:`LifecycleAdapter`."""

    def __init__(
        self,
        controller: LifecycleController,
        adapter: LifecycleAdapter,
        *,
        drain_polls: int = DEFAULT_DRAIN_POLLS,
        drain_interval: float = DEFAULT_DRAIN_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._ctrl = controller
        self._adapter = adapter
        self._drain_polls = drain_polls
        self._drain_interval = drain_interval
        self._sleep = sleep

    @property
    def controller(self) -> LifecycleController:
        return self._ctrl

    # --------------------------------------------------------------- bootstrap

    def bootstrap(self) -> None:
        """Bring the engine under control. Leaves the controller in READY.

        A refused adoption (:class:`~rolloutcore.errors.AlreadyManagedEngineError`)
        leaves the controller UNINITIALIZED on purpose: nothing was written, the
        engine is healthy, and it simply is not ours. Tainting would demand a
        restart of an engine another controller is using correctly.
        """
        try:
            evidence = self._adapter.bootstrap()
        except (AlreadyManagedEngineError, RolloutCoreError):
            raise
        except Exception as exc:
            raise self._ctrl.taint(
                f"bootstrap failed with an unknown engine outcome: {exc!r}"
            ) from exc
        self._ctrl.initialize(evidence)

    # ------------------------------------------------------------------- cycle

    def run_cycle(self, target: UpdateTarget) -> CycleResult:
        """One full ``READY -> ... -> READY`` cycle installing ``target``.

        Every step is fail-closed: a mutating adapter failure taints rather than
        leaving the controller in a state it cannot justify, and the adapter is
        never asked to resume a target that failed validation.
        """
        visited: list[LifecycleState] = [self._ctrl.state]
        started = time.monotonic()

        # READY -> DRAINING
        self._ctrl.begin_drain()
        visited.append(self._ctrl.state)
        self._effect("begin_drain", lambda: self._adapter.begin_drain())

        # DRAINING -> QUIESCED, polling until the engine confirms.
        polls = self._drain_until_quiesced()
        visited.append(self._ctrl.state)

        # QUIESCED -> UPDATING
        self._ctrl.begin_update(target)
        visited.append(self._ctrl.state)
        self._effect("start_weight_update", lambda: self._adapter.start_weight_update(target))

        # UPDATING -> INVALIDATING
        update_evidence = self._effect(
            "complete_weight_update", lambda: self._adapter.complete_weight_update(target)
        )
        self._ctrl.confirm_updated(update_evidence)
        visited.append(self._ctrl.state)

        # INVALIDATING -> VALIDATING
        invalidate_evidence = self._effect(
            "invalidate_caches", lambda: self._adapter.invalidate_caches(target)
        )
        self._ctrl.confirm_invalidated(invalidate_evidence)
        visited.append(self._ctrl.state)

        # VALIDATING -> RESUMING. Still paused here.
        validate_evidence = self._observe(
            "validate_pre_resume", lambda: self._adapter.validate_pre_resume(target)
        )
        self._ctrl.confirm_validated(validate_evidence)
        visited.append(self._ctrl.state)

        # RESUMING -> READY. The commit point.
        resume_evidence = self._effect("resume", lambda: self._adapter.resume(target))
        self._ctrl.confirm_resumed(resume_evidence)
        visited.append(self._ctrl.state)

        committed = self._ctrl.current_target
        assert committed is not None
        return CycleResult(
            committed=committed,
            states_visited=tuple(visited),
            drain_polls=polls,
            duration_s=time.monotonic() - started,
        )

    def install_next(self, identity: WeightIdentity | None = None) -> CycleResult:
        """Advance one generation beyond the current one.

        ``identity`` defaults to the current identity, which is only appropriate
        for a no-op or fake update. A real update must pass the identity of the
        weights it actually transfers, or I4 would claim a weight source it
        cannot support.
        """
        return self.run_cycle(self._ctrl.next_target(identity))

    # ----------------------------------------------------------------- internals

    def _effect(self, what: str, fn: Callable[[], T]) -> T:
        """Run a mutating adapter call, tainting on an unknown outcome.

        RolloutCore's own errors pass through untouched -- they already encode
        recoverability, and ``EngineTaintedError`` from the controller is
        idempotent with the taint raised here. A *foreign* exception from a
        mutating call is ambiguous by construction, so it becomes a taint.
        """
        try:
            return fn()
        except RolloutCoreError:
            raise
        except Exception as exc:
            raise self._ctrl.taint(
                f"{what} failed with an unknown engine outcome: {exc!r}"
            ) from exc

    def _observe(self, what: str, fn: Callable[[], T]) -> T:
        """Run an observational adapter call. Failures stay retryable.

        The engine is paused during every state these are called from
        (DRAINING and VALIDATING), so a failed read leaves it unable to serve
        anything new -- safe, and recoverable without declaring the engine
        untrustworthy. ``what`` is kept for symmetry with :meth:`_effect` and for
        the eventual logging hook.
        """
        del what  # reserved for logging; reads are deliberately not tainted
        return fn()

    def _drain_until_quiesced(self) -> int:
        """Poll ``await_drain`` until the engine confirms, then confirm locally.

        ``EvidenceNotReady`` means "still draining", which is normal; anything
        else propagates. ``await_drain`` is required to return promptly, so the
        poll budget is also a wall-clock deadline: ``drain_polls`` x
        ``drain_interval`` seconds, checked against ``time.monotonic`` in case an
        adapter ever blocks longer than it promises.
        """
        deadline = time.monotonic() + self._drain_polls * self._drain_interval
        for poll in range(1, self._drain_polls + 1):
            evidence: DrainEvidence = self._observe("await_drain", self._adapter.await_drain)
            try:
                self._ctrl.confirm_drained(evidence)
                return poll
            except EvidenceNotReady:
                if poll < self._drain_polls and time.monotonic() < deadline:
                    self._sleep(self._drain_interval)
                continue
        raise RolloutCoreError(
            f"drain did not complete after {self._drain_polls} polls "
            f"({self._drain_polls * self._drain_interval:.0f}s); "
            "the engine is still paused and has not been resumed"
        )
