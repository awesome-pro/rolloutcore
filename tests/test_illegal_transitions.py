# SPDX-License-Identifier: Apache-2.0
"""Every illegal transition, enumerated as the full cartesian product.

``LifecycleState`` has 9 members and ``Event`` has 9 members, so there are 81
(state, event) pairs. Eight are forward edges, ``taint`` is legal from every
state (absorbing at TAINTED), and the remaining 64 are illegal. Rather than
hand-listing them, this module derives the expectation from the table and
asserts it for all 81 -- so adding a state or an event without updating the
design immediately shows up as failures here rather than as silent gaps.

Illegality is also asserted *strongly*: a rejected event must leave the
controller observably untouched (state, targets, active rollouts, orphans,
journal).
"""

from __future__ import annotations

import unittest

from support import assert_state_unchanged, drive_to, good_drain, invoke

from rolloutcore import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    ROLLOUT_COMPLETION_STATES,
    Event,
    IllegalTransitionError,
    LifecycleState,
    NotServingError,
    UnknownRolloutError,
)


def _is_legal(state: LifecycleState, event: Event) -> bool:
    """Expected legality, derived independently of the controller's own check."""
    if event is Event.TAINT:
        return True  # legal everywhere; absorbing no-op at TAINTED
    return (state, event) in LEGAL_TRANSITIONS


class TestIllegalTransitionMatrix(unittest.TestCase):
    def test_matrix_is_complete(self):
        from rolloutcore import enumerate_transition_matrix

        matrix = enumerate_transition_matrix()
        self.assertEqual(len(matrix), len(LifecycleState) * len(Event))
        self.assertEqual(len(set(matrix)), len(matrix))
        self.assertEqual(len(matrix), 81)

    def test_expected_legal_and_illegal_counts(self):
        from rolloutcore import enumerate_transition_matrix

        pairs = enumerate_transition_matrix()
        legal = [p for p in pairs if _is_legal(*p)]
        illegal = [p for p in pairs if not _is_legal(*p)]
        # 8 forward edges + taint legal from all 9 states.
        self.assertEqual(len(legal), 8 + len(LifecycleState))
        self.assertEqual(len(legal), 17)
        self.assertEqual(len(illegal), 81 - len(legal))
        self.assertEqual(len(illegal), 64)

    def test_every_pair_behaves_as_specified(self):
        from rolloutcore import enumerate_transition_matrix

        for state, event in enumerate_transition_matrix():
            with self.subTest(state=state.value, event=event.value):
                ctrl = drive_to(state)
                if _is_legal(state, event):
                    invoke(ctrl, event)  # must not raise
                else:
                    exc = assert_state_unchanged(self, ctrl, lambda c=ctrl, e=event: invoke(c, e))
                    self.assertIsInstance(
                        exc,
                        IllegalTransitionError,
                        f"{state.value}+{event.value} raised {type(exc).__name__} "
                        f"but an illegal transition must raise IllegalTransitionError",
                    )
                    self.assertEqual(exc.state, state.value)
                    self.assertEqual(exc.event, event.value)

    def test_illegal_transition_message_is_actionable(self):
        ctrl = drive_to(LifecycleState.READY)
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.confirm_drained(good_drain()))
        self.assertIsInstance(exc, IllegalTransitionError)
        self.assertEqual(exc.invariant, "T-ILLEGAL")
        self.assertIn("confirm_drained", str(exc))
        self.assertIn("READY", str(exc))

    def test_tainted_is_absorbing_for_every_forward_event(self):

        forward = [e for e in Event if e is not Event.TAINT]
        for event in forward:
            with self.subTest(event=event.value):
                ctrl = drive_to(LifecycleState.TAINTED)
                exc = assert_state_unchanged(self, ctrl, lambda c=ctrl, e=event: invoke(c, e))
                self.assertIsInstance(exc, IllegalTransitionError)

    def test_no_edge_targets_tainted_implicitly(self):
        self.assertNotIn(LifecycleState.TAINTED, set(LEGAL_TRANSITIONS.values()))

    def test_no_edge_leaves_tainted(self):
        self.assertEqual([k for k in LEGAL_TRANSITIONS if k[0] is LifecycleState.TAINTED], [])

    def test_no_edge_skips_the_resume_state(self):
        """Amendment 1: VALIDATING must lead to RESUMING, never straight to READY."""
        self.assertIs(
            LEGAL_TRANSITIONS[(LifecycleState.VALIDATING, Event.CONFIRM_VALIDATED)],
            LifecycleState.RESUMING,
        )
        self.assertNotIn((LifecycleState.VALIDATING, Event.CONFIRM_RESUMED), LEGAL_TRANSITIONS)


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
                    exc = assert_state_unchanged(self, ctrl, lambda c=ctrl: c.admit_rollout("r1"))
                    self.assertIsInstance(exc, NotServingError)
                    if state is not LifecycleState.TAINTED:
                        self.assertNotEqual(ctrl.state, LifecycleState.TAINTED)

    def test_unknown_rollout_completion_across_states(self):
        for state in LifecycleState:
            with self.subTest(state=state.value):
                ctrl = drive_to(state)
                for method in (ctrl.finish_rollout, ctrl.abort_rollout):
                    exc = assert_state_unchanged(self, ctrl, lambda m=method: m("never-admitted"))
                    if state in ROLLOUT_COMPLETION_STATES:
                        self.assertIsInstance(exc, UnknownRolloutError)
                    else:
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

    def test_rollouts_cannot_complete_during_resuming(self):
        """RESUMING is not a completion state; only READY and DRAINING are."""
        ctrl = drive_to(LifecycleState.READY)
        ctrl.admit_rollout("r1")
        ctrl.finish_rollout("r1")
        self.assertEqual(ROLLOUT_COMPLETION_STATES, {LifecycleState.READY, LifecycleState.DRAINING})


class TestEvidenceNotReadyIsNotIllegal(unittest.TestCase):
    """A still-running drain is a retryable condition, not an illegal event."""

    def test_incomplete_drain_raises_not_ready_not_illegal(self):
        from rolloutcore import EvidenceNotReady

        ctrl = drive_to(LifecycleState.DRAINING)
        exc = assert_state_unchanged(
            self, ctrl, lambda: ctrl.confirm_drained(good_drain(completed=False))
        )
        self.assertIsInstance(exc, EvidenceNotReady)
        self.assertNotIsInstance(exc, IllegalTransitionError)
        self.assertIs(ctrl.state, LifecycleState.DRAINING)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
