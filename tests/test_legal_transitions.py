# SPDX-License-Identifier: Apache-2.0
"""Every legal transition, asserted one at a time.

Deliberately exhaustive rather than representative: the lifecycle has exactly
eight forward transitions plus taint, and each is exercised with valid evidence
and its postconditions asserted.

The ordering assertions here are the amendment-1 contract: VALIDATING runs while
the engine is still paused, and the resume is issued only on entry to RESUMING.
"""

from __future__ import annotations

import unittest

from support import (
    GOOD_BOOTSTRAP,
    TARGET_V1,
    advance_to_version,
    bootstrapped,
    drive_to,
    good_drain,
    good_invalidate,
    good_resume,
    good_update,
    good_validate,
    run_full_cycle,
    target,
)

from rolloutcore import (
    ADMITTING_STATES,
    LEGAL_TRANSITIONS,
    PAUSED_STATES,
    Event,
    LifecycleState,
    WeightVersion,
)


class TestLegalTransitions(unittest.TestCase):
    def test_uninitialized_to_ready(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        self.assertIsNone(ctrl.current_version)
        self.assertIsNone(ctrl.current_identity)
        ctrl.initialize(GOOD_BOOTSTRAP)
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.current_identity, GOOD_BOOTSTRAP.weight_identity)

    def test_ready_to_draining(self):
        ctrl = bootstrapped()
        plan = ctrl.begin_drain()
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        self.assertIn("pause", plan.tags)
        self.assertIn("mode=wait", " ".join(plan.steps))
        self.assertNotIn("mode=keep", " ".join(plan.steps))

    def test_draining_to_quiesced(self):
        ctrl = drive_to(LifecycleState.DRAINING)
        ctrl.confirm_drained(good_drain())
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)
        self.assertEqual(ctrl.active_rollout_count, 0)
        self.assertIsNone(ctrl.pending_target)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_quiesced_to_updating(self):
        ctrl = drive_to(LifecycleState.QUIESCED)
        plan = ctrl.begin_update(TARGET_V1)
        self.assertIs(ctrl.state, LifecycleState.UPDATING)
        self.assertEqual(ctrl.pending_target, TARGET_V1)
        # Atomicity: the committed target must NOT move yet.
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.current_identity, GOOD_BOOTSTRAP.weight_identity)
        self.assertIn("start_weight_update", plan.tags)

    def test_updating_to_invalidating(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        plan = ctrl.confirm_updated(good_update(TARGET_V1))
        self.assertIs(ctrl.state, LifecycleState.INVALIDATING)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.pending_target, TARGET_V1)
        # All three resets are the point of this state.
        self.assertEqual(plan.tags, ("reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"))

    def test_invalidating_to_validating(self):
        ctrl = drive_to(LifecycleState.INVALIDATING)
        plan = ctrl.confirm_invalidated(good_invalidate())
        self.assertIs(ctrl.state, LifecycleState.VALIDATING)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        # Amendment 1: no resume yet.
        self.assertNotIn("resume", plan.tags)
        self.assertIn("get_weight_info", plan.tags)

    def test_validating_to_resuming_does_not_commit(self):
        """Amendment 1: passing validation must not publish the version."""
        ctrl = drive_to(LifecycleState.VALIDATING)
        plan = ctrl.confirm_validated(good_validate(TARGET_V1))
        self.assertIs(ctrl.state, LifecycleState.RESUMING)
        # Still not committed: the engine has been validated but not yet resumed.
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.current_identity, GOOD_BOOTSTRAP.weight_identity)
        self.assertEqual(ctrl.pending_target, TARGET_V1)
        self.assertIn("resume", plan.tags)

    def test_resuming_to_ready_commits(self):
        ctrl = drive_to(LifecycleState.RESUMING)
        ctrl.confirm_resumed(good_resume(TARGET_V1))
        self.assertIs(ctrl.state, LifecycleState.READY)
        self.assertEqual(ctrl.current_version, WeightVersion(1))
        self.assertEqual(ctrl.current_identity, TARGET_V1.identity)
        self.assertIsNone(ctrl.pending_target)

    def test_full_cycle_state_sequence(self):
        """The exact nine-state path mandated by amendment 1."""
        ctrl = bootstrapped()
        seen = [ctrl.state]
        ctrl.begin_drain()
        seen.append(ctrl.state)
        ctrl.confirm_drained(good_drain())
        seen.append(ctrl.state)
        ctrl.begin_update(TARGET_V1)
        seen.append(ctrl.state)
        ctrl.confirm_updated(good_update(TARGET_V1))
        seen.append(ctrl.state)
        ctrl.confirm_invalidated(good_invalidate())
        seen.append(ctrl.state)
        ctrl.confirm_validated(good_validate(TARGET_V1))
        seen.append(ctrl.state)
        ctrl.confirm_resumed(good_resume(TARGET_V1))
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
                LifecycleState.RESUMING,
                LifecycleState.READY,
            ],
        )

    def test_two_consecutive_cycles(self):
        ctrl = bootstrapped()
        for n in (1, 2, 3):
            t = target(n)
            run_full_cycle(ctrl, t)
            self.assertIs(ctrl.state, LifecycleState.READY)
            self.assertEqual(ctrl.current_version, WeightVersion(n))
            self.assertEqual(ctrl.current_identity, t.identity)

    def test_journal_records_every_hop(self):
        ctrl = drive_to(LifecycleState.RESUMING)
        self.assertEqual(
            [r.event for r in ctrl.journal],
            [
                Event.INITIALIZE.value,
                Event.BEGIN_DRAIN.value,
                Event.CONFIRM_DRAINED.value,
                Event.BEGIN_UPDATE.value,
                Event.CONFIRM_UPDATED.value,
                Event.CONFIRM_INVALIDATED.value,
                Event.CONFIRM_VALIDATED.value,
            ],
        )
        self.assertEqual([r.seq for r in ctrl.journal], list(range(1, 8)))

    def test_journal_carries_the_weight_identity(self):
        """I4 needs an identity, not just a generation number, in the record."""
        ctrl = drive_to(LifecycleState.READY)
        run_full_cycle(ctrl, TARGET_V1)
        publish = [r for r in ctrl.journal if r.event == Event.CONFIRM_RESUMED.value]
        self.assertEqual(len(publish), 1)
        self.assertEqual(publish[0].version, WeightVersion(1))
        self.assertEqual(publish[0].identity, TARGET_V1.identity)

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

    def test_transition_table_is_exactly_the_eight_forward_edges(self):
        """Guards against a transition being added without a test."""
        expected = {
            (LifecycleState.UNINITIALIZED, Event.INITIALIZE),
            (LifecycleState.READY, Event.BEGIN_DRAIN),
            (LifecycleState.DRAINING, Event.CONFIRM_DRAINED),
            (LifecycleState.QUIESCED, Event.BEGIN_UPDATE),
            (LifecycleState.UPDATING, Event.CONFIRM_UPDATED),
            (LifecycleState.INVALIDATING, Event.CONFIRM_INVALIDATED),
            (LifecycleState.VALIDATING, Event.CONFIRM_VALIDATED),
            (LifecycleState.RESUMING, Event.CONFIRM_RESUMED),
        }
        self.assertEqual(set(LEGAL_TRANSITIONS), expected)

    def test_only_ready_admits_rollouts(self):
        self.assertEqual(ADMITTING_STATES, frozenset({LifecycleState.READY}))

    def test_validating_is_a_paused_state(self):
        """Amendment 1: the engine must still be paused throughout VALIDATING."""
        self.assertIn(LifecycleState.VALIDATING, PAUSED_STATES)
        self.assertNotIn(LifecycleState.RESUMING, PAUSED_STATES)
        self.assertNotIn(LifecycleState.READY, PAUSED_STATES)

    def test_advance_to_version_helper(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(4))
        self.assertEqual(ctrl.current_version, WeightVersion(4))
        self.assertIs(ctrl.state, LifecycleState.READY)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
