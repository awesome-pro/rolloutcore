# SPDX-License-Identifier: Apache-2.0
"""Every legal transition, asserted one at a time.

This file is deliberately exhaustive rather than representative: the lifecycle
has exactly seven forward transitions plus taint, and each one is exercised with
valid evidence and its postconditions asserted.
"""

from __future__ import annotations

import unittest

from rolloutcore import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    Event,
    InvalidateEvidence,
    LifecycleState,
    WeightVersion,
)
from support import (
    GOOD_BOOTSTRAP,
    GOOD_DRAIN,
    GOOD_INVALIDATE,
    assert_state_unchanged,
    bootstrapped,
    drive_to,
    good_update,
    good_validate,
)


class TestLegalTransitions(unittest.TestCase):
    def test_uninitialized_to_ready(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        self.assertIsNone(ctrl.current_version)
        ctrl.initialize(GOOD_BOOTSTRAP)
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_ready_to_draining(self):
        ctrl = bootstrapped()
        plan = ctrl.begin_drain()
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        # begin_drain must name the real drain primitive, not the unsafe one.
        self.assertIn("POST /pause?mode=wait&clear_cache=true", plan.steps)
        self.assertNotIn("mode=keep", " ".join(plan.steps))

    def test_draining_to_quiesced(self):
        ctrl = drive_to(LifecycleState.DRAINING)
        ctrl.confirm_drained(GOOD_DRAIN)
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)
        self.assertEqual(ctrl.active_rollout_count, 0)
        # Nothing is pending yet: QUIESCED is not a version change.
        self.assertIsNone(ctrl.pending_version)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_quiesced_to_updating(self):
        ctrl = drive_to(LifecycleState.QUIESCED)
        plan = ctrl.begin_update(WeightVersion(1))
        self.assertIs(ctrl.state, LifecycleState.UPDATING)
        self.assertEqual(ctrl.pending_version, WeightVersion(1))
        # Atomicity: the committed version must NOT move yet.
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertIn("POST /start_weight_update", plan.steps)

    def test_updating_to_invalidating(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        plan = ctrl.confirm_updated(good_update(WeightVersion(1)))
        self.assertIs(ctrl.state, LifecycleState.INVALIDATING)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.pending_version, WeightVersion(1))
        # The three resets are the whole point of this state.
        self.assertEqual(
            plan.steps,
            (
                "POST /reset_prefix_cache?reset_running_requests=true&reset_external=true",
                "POST /reset_encoder_cache",
                "POST /reset_mm_cache",
            ),
        )

    def test_invalidating_to_validating(self):
        ctrl = drive_to(LifecycleState.INVALIDATING)
        plan = ctrl.confirm_invalidated(GOOD_INVALIDATE)
        self.assertIs(ctrl.state, LifecycleState.VALIDATING)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(plan.steps, ("POST /resume", "GET /weight_info", "GET /is_paused"))

    def test_validating_to_ready_publishes_version(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ctrl.confirm_validated(good_validate(WeightVersion(1)))
        self.assertIs(ctrl.state, LifecycleState.READY)
        # I1: the commit point.
        self.assertEqual(ctrl.current_version, WeightVersion(1))
        self.assertIsNone(ctrl.pending_version)

    def test_full_cycle_v0_to_v1(self):
        """The complete happy path, asserting the state sequence exactly."""
        ctrl = bootstrapped()
        seen = [ctrl.state]
        ctrl.begin_drain()
        seen.append(ctrl.state)
        ctrl.confirm_drained(GOOD_DRAIN)
        seen.append(ctrl.state)
        ctrl.begin_update(WeightVersion(1))
        seen.append(ctrl.state)
        ctrl.confirm_updated(good_update(WeightVersion(1)))
        seen.append(ctrl.state)
        ctrl.confirm_invalidated(GOOD_INVALIDATE)
        seen.append(ctrl.state)
        ctrl.confirm_validated(good_validate(WeightVersion(1)))
        seen.append(ctrl.state)

        self.assertEqual(
            seen,
            [
                LifecycleState.READY,
                LifecycleState.DRAINING,
                LifecycleState.QUIESCED,
                LifecycleState.UPDATING,
                LifecycleState.INVALIDATING,
                LifecycleState.VALIDATING,
                LifecycleState.READY,
            ],
        )
        self.assertEqual(ctrl.current_version, WeightVersion(1))

    def test_two_consecutive_cycles(self):
        """Versions advance monotonically across cycles."""
        ctrl = bootstrapped()
        for target in (WeightVersion(1), WeightVersion(2), WeightVersion(3)):
            ctrl.begin_drain()
            ctrl.confirm_drained(GOOD_DRAIN)
            ctrl.begin_update(target)
            ctrl.confirm_updated(good_update(target))
            ctrl.confirm_invalidated(GOOD_INVALIDATE)
            ctrl.confirm_validated(good_validate(target))
            self.assertIs(ctrl.state, LifecycleState.READY)
            self.assertEqual(ctrl.current_version, target)

    def test_journal_records_every_hop(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        events = [r.event for r in ctrl.journal]
        self.assertEqual(
            events,
            [
                Event.INITIALIZE.value,
                Event.BEGIN_DRAIN.value,
                Event.CONFIRM_DRAINED.value,
                Event.BEGIN_UPDATE.value,
                Event.CONFIRM_UPDATED.value,
                Event.CONFIRM_INVALIDATED.value,
            ],
        )
        # Sequence numbers are dense and 1-based.
        self.assertEqual([r.seq for r in ctrl.journal], list(range(1, 7)))

    def test_taint_is_legal_from_every_non_tainted_state(self):
        for state in LifecycleState:
            with self.subTest(state=state.value):
                ctrl = drive_to(state)
                ctrl.taint("probe")
                self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_taint_from_tainted_is_idempotent_and_keeps_first_reason(self):
        ctrl = drive_to(LifecycleState.TAINTED)
        journal_len = len(ctrl.journal)
        ctrl.taint("second reason")
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.taint_reason, "operator abort")
        self.assertEqual(len(ctrl.journal), journal_len, "no-op taint must not log a hop")

    def test_transition_table_is_exactly_the_seven_forward_edges(self):
        """Guards against a transition being added without a test."""
        expected = {
            (LifecycleState.UNINITIALIZED, Event.INITIALIZE),
            (LifecycleState.READY, Event.BEGIN_DRAIN),
            (LifecycleState.DRAINING, Event.CONFIRM_DRAINED),
            (LifecycleState.QUIESCED, Event.BEGIN_UPDATE),
            (LifecycleState.UPDATING, Event.CONFIRM_UPDATED),
            (LifecycleState.INVALIDATING, Event.CONFIRM_INVALIDATED),
            (LifecycleState.VALIDATING, Event.CONFIRM_VALIDATED),
        }
        self.assertEqual(set(LEGAL_TRANSITIONS), expected)
        # Every target must be a real state.
        for target in LEGAL_TRANSITIONS.values():
            self.assertIsInstance(target, LifecycleState)

    def test_only_ready_admits_rollouts(self):
        self.assertEqual(ADMITTING_STATES, frozenset({LifecycleState.READY}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
