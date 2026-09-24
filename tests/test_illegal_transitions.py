# SPDX-License-Identifier: Apache-2.0
"""Every illegal transition, enumerated as the full cartesian product.

``LifecycleState`` has 8 members and ``Event`` has 8 members, so there are 64
(state, event) pairs. Seven are forward edges, ``taint`` is legal from every
state (absorbing at TAINTED), and the remaining 49 are illegal. Rather than
hand-listing them, this module derives the expectation from the table and
asserts it for all 64 -- so adding a state or an event without updating the
design immediately shows up as failures here rather than as silent gaps.

Illegality is also asserted *strongly*: a rejected event must leave the
controller observably untouched (state, versions, active rollouts, journal).
"""

from __future__ import annotations

import unittest

from rolloutcore import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    ROLLOUT_COMPLETION_STATES,
    Event,
    IllegalTransitionError,
    LifecycleState,
    NotServingError,
    UnknownRolloutError,
    enumerate_transition_matrix,
)
from support import GOOD_DRAIN, assert_state_unchanged, drive_to, invoke


def _is_legal(state: LifecycleState, event: Event) -> bool:
    """Expected legality, derived independently of the controller's own check."""
    if event is Event.TAINT:
        return True  # legal everywhere; absorbing no-op at TAINTED
    return (state, event) in LEGAL_TRANSITIONS


class TestIllegalTransitionMatrix(unittest.TestCase):
    def test_matrix_is_complete(self):
        """8 states x 8 events, with no duplicates."""
        matrix = enumerate_transition_matrix()
        self.assertEqual(len(matrix), len(LifecycleState) * len(Event))
        self.assertEqual(len(set(matrix)), len(matrix))

    def test_expected_legal_and_illegal_counts(self):
        pairs = enumerate_transition_matrix()
        legal = [p for p in pairs if _is_legal(*p)]
        illegal = [p for p in pairs if not _is_legal(*p)]
        # 7 forward edges + taint legal from all 8 states (a no-op at TAINTED).
        self.assertEqual(len(legal), 7 + len(LifecycleState))
        self.assertEqual(len(illegal), 64 - len(legal))
        self.assertEqual(len(illegal), 49)

    def test_every_pair_behaves_as_specified(self):
        for state, event in enumerate_transition_matrix():
            with self.subTest(state=state.value, event=event.value):
                ctrl = drive_to(state)
                if _is_legal(state, event):
                    invoke(ctrl, event)  # must not raise
                else:
                    exc = assert_state_unchanged(
                        self, ctrl, lambda c=ctrl, e=event: invoke(c, e)
                    )
                    self.assertIsInstance(
                        exc,
                        IllegalTransitionError,
                        f"{state.value}+{event.value} raised {type(exc).__name__} "
                        f"but an illegal transition must raise IllegalTransitionError",
                    )
                    self.assertEqual(exc.state, state.value)
                    self.assertEqual(exc.event, event.value)

    def test_illegal_transition_message_is_actionable(self):
        """The error names both the state and the event, for debuggability."""
        ctrl = drive_to(LifecycleState.READY)
        exc = assert_state_unchanged(
            self, ctrl, lambda: ctrl.confirm_drained(GOOD_DRAIN)
        )
        self.assertIsInstance(exc, IllegalTransitionError)
        self.assertEqual(exc.invariant, "T-ILLEGAL")
        self.assertEqual(exc.state, LifecycleState.READY.value)
        self.assertEqual(exc.event, Event.CONFIRM_DRAINED.value)
        self.assertIn("confirm_drained", str(exc))
        self.assertIn("READY", str(exc))

    def test_tainted_is_absorbing_for_every_forward_event(self):
        """No forward edge exists out of TAINTED."""
        forward = [e for e in Event if e is not Event.TAINT]
        for event in forward:
            with self.subTest(event=event.value):
                ctrl = drive_to(LifecycleState.TAINTED)
                exc = assert_state_unchanged(
                    self, ctrl, lambda c=ctrl, e=event: invoke(c, e)
                )
                self.assertIsInstance(exc, IllegalTransitionError)

    def test_no_edge_targets_tainted_implicitly(self):
        """TAINTED is only ever reached through taint(), never a table edge."""
        self.assertNotIn(LifecycleState.TAINTED, set(LEGAL_TRANSITIONS.values()))

    def test_no_edge_leaves_tainted(self):
        self.assertEqual(
            [k for k in LEGAL_TRANSITIONS if k[0] is LifecycleState.TAINTED], []
        )


class TestRolloutAdmissionAcrossStates(unittest.TestCase):
    """Admission is legal in exactly one state, and rejection never taints."""

    def test_admit_only_in_ready(self):
        for state in LifecycleState:
            with self.subTest(state=state.value):
                ctrl = drive_to(state)
                if state in ADMITTING_STATES:
                    binding = ctrl.admit_rollout("r1")
                    self.assertEqual(binding.request_id, "r1")
                else:
                    exc = assert_state_unchanged(
                        self, ctrl, lambda c=ctrl: c.admit_rollout("r1")
                    )
                    self.assertIsInstance(exc, NotServingError)
                    # Rejecting a request is normal operation, not an engine
                    # fault: it must not *cause* a taint (it may already be in
                    # one, which is why the assertion is conditional).
                    if state is not LifecycleState.TAINTED:
                        self.assertNotEqual(ctrl.state, LifecycleState.TAINTED)

    def test_unknown_rollout_completion_is_rejected(self):
        for state in LifecycleState:
            with self.subTest(state=state.value):
                ctrl = drive_to(state)
                for method in (ctrl.finish_rollout, ctrl.abort_rollout):
                    if state in ROLLOUT_COMPLETION_STATES:
                        exc = assert_state_unchanged(
                            self, ctrl, lambda m=method: m("never-admitted")
                        )
                        self.assertIsInstance(exc, UnknownRolloutError)
                    else:
                        exc = assert_state_unchanged(
                            self, ctrl, lambda m=method: m("never-admitted")
                        )
                        self.assertIsInstance(exc, IllegalTransitionError)

    def test_double_completion_is_rejected(self):
        ctrl = drive_to(LifecycleState.READY)
        ctrl.admit_rollout("r1")
        ctrl.finish_rollout("r1")
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.finish_rollout("r1"))
        self.assertIsInstance(exc, UnknownRolloutError)

    def test_duplicate_admission_is_rejected(self):
        ctrl = drive_to(LifecycleState.READY)
        ctrl.admit_rollout("r1")
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.admit_rollout("r1"))
        self.assertEqual(exc.invariant, "I2-BINDING")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
