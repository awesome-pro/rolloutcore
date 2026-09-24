# SPDX-License-Identifier: Apache-2.0
"""Invariant tests.

Each test names the invariant it defends. Invariants I1-I6 come from the
original project plan (``old-plan.md`` section 21); I7-I9 come from the
implementation amendments. The mapping from invariant to enforcing code is in
``docs/state-machine.md``.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rolloutcore import (
    DrainEvidence,
    EngineTaintedError,
    Event,
    InvalidateEvidence,
    LifecycleState,
    NotServingError,
    VersionMismatchError,
    WeightVersion,
)
from support import (
    GOOD_BOOTSTRAP,
    GOOD_DRAIN,
    GOOD_INVALIDATE,
    advance_to_version,
    assert_state_unchanged,
    bootstrapped,
    drive_to,
    good_update,
    good_validate,
)


class TestI1VersionAtomicity(unittest.TestCase):
    """An update is either completely committed or not visible."""

    def test_committed_version_does_not_move_until_validation(self):
        ctrl = bootstrapped()
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.begin_drain()
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_drained(GOOD_DRAIN)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.begin_update(WeightVersion(1))
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_updated(good_update(WeightVersion(1)))
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_invalidated(GOOD_INVALIDATE)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        # Only here.
        ctrl.confirm_validated(good_validate(WeightVersion(1)))
        self.assertEqual(ctrl.current_version, WeightVersion(1))

    def test_pending_version_is_cleared_on_commit(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        self.assertEqual(ctrl.pending_version, WeightVersion(1))
        ctrl.confirm_validated(good_validate(WeightVersion(1)))
        self.assertIsNone(ctrl.pending_version)

    def test_update_target_must_be_strictly_greater(self):
        for bad in (WeightVersion(0),):
            with self.subTest(target=bad.label):
                ctrl = drive_to(LifecycleState.QUIESCED)
                exc = assert_state_unchanged(
                    self, ctrl, lambda c=ctrl, t=bad: c.begin_update(t)
                )
                self.assertEqual(exc.invariant, "I1-MONOTONIC")
                self.assertEqual(ctrl.current_version, WeightVersion(0))
                self.assertIsNone(ctrl.pending_version)

    def test_lower_target_is_rejected(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(3))
        ctrl.begin_drain()
        ctrl.confirm_drained(GOOD_DRAIN)
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.begin_update(WeightVersion(2)))
        self.assertEqual(exc.invariant, "I1-MONOTONIC")
        self.assertEqual(ctrl.current_version, WeightVersion(3))
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)


class TestI2AdmissionBinding(unittest.TestCase):
    """A request's weight version is fixed at admission."""

    def test_binding_captures_the_committed_version(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(4))
        b = ctrl.admit_rollout("r1")
        self.assertEqual(b.version, WeightVersion(4))
        self.assertEqual(b.cache_salt, WeightVersion(4).label)

    def test_binding_is_frozen(self):
        ctrl = bootstrapped()
        b = ctrl.admit_rollout("r1")
        with self.assertRaises(Exception):
            b.version = WeightVersion(99)  # type: ignore[misc]

    def test_binding_survives_a_later_version_change(self):
        """A rollout admitted at vN still reports vN after the engine moves on."""
        ctrl = bootstrapped()
        b = ctrl.admit_rollout("r1")
        ctrl.finish_rollout("r1")
        advance_to_version(ctrl, WeightVersion(2))
        self.assertEqual(b.version, WeightVersion(0), "binding must not follow the engine")
        self.assertEqual(ctrl.current_version, WeightVersion(2))

    def test_no_admission_outside_ready(self):
        for state in LifecycleState:
            with self.subTest(state=state.value):
                ctrl = drive_to(state)
                if state is LifecycleState.READY:
                    continue
                exc = assert_state_unchanged(self, ctrl, lambda c=ctrl: c.admit_rollout("x"))
                self.assertIsInstance(exc, NotServingError)


class TestI3NoCrossVersionCacheReuse(unittest.TestCase):
    """KV created under version N may not be consumed under N+1."""

    def test_invalidate_evidence_requires_all_three_caches(self):
        """cache_salt isolates the prefix cache only; the others are mandatory."""
        for name in ("prefix_cache_reset", "encoder_cache_reset", "mm_cache_reset"):
            with self.subTest(missing=name):
                ctrl = drive_to(LifecycleState.INVALIDATING)
                ev = InvalidateEvidence(
                    prefix_cache_reset=name != "prefix_cache_reset",
                    encoder_cache_reset=name != "encoder_cache_reset",
                    mm_cache_reset=name != "mm_cache_reset",
                )
                self.assertIsNotNone(ev.failure_reason())
                before = ctrl.current_version
                with self.assertRaises(EngineTaintedError):
                    ctrl.confirm_invalidated(ev)
                self.assertIs(ctrl.state, LifecycleState.TAINTED)
                # A failed invalidation must never publish the new version.
                self.assertEqual(ctrl.current_version, before)
                self.assertIsNone(ctrl.pending_version)

    def test_all_three_present_passes(self):
        self.assertIsNone(GOOD_INVALIDATE.failure_reason())

    def test_cache_salt_is_version_scoped(self):
        """Distinct versions must not share a prefix-cache salt."""
        salts = {WeightVersion(n).cache_salt for n in range(8)}
        self.assertEqual(len(salts), 8)


class TestI4ReplayIdentity(unittest.TestCase):
    """A stored trajectory identifies the exact weights that produced it."""

    def test_repeated_cycles_produce_distinct_salts_and_versions(self):
        ctrl = bootstrapped()
        bounds = []
        for _ in range(3):
            b = ctrl.admit_rollout("r")
            bounds.append((b.version, b.cache_salt))
            ctrl.finish_rollout("r")
            advance_to_version(ctrl, ctrl.current_version.next())
        versions = [v for v, _ in bounds]
        salts = [s for _, s in bounds]
        self.assertEqual(versions, [WeightVersion(0), WeightVersion(1), WeightVersion(2)])
        self.assertEqual(len(set(salts)), 3)

    def test_admission_sequence_is_monotonic(self):
        ctrl = bootstrapped()
        seqs = [ctrl.admit_rollout(f"r{i}").admitted_seq for i in range(4)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 4)


class TestI5DrainCorrectness(unittest.TestCase):
    """No active old-version requests remain when update begins."""

    def test_confirm_drained_refuses_with_active_rollouts(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        exc = assert_state_unchanged(
            self, ctrl, lambda: ctrl.confirm_drained(GOOD_DRAIN)
        )
        self.assertEqual(exc.invariant, "I5-DRAIN")
        # Not a taint: the engine is healthy, the drain just is not finished.
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        self.assertIn("r1", str(exc))

    def test_confirm_drained_succeeds_once_all_released(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        ctrl.finish_rollout("r1")
        ctrl.abort_rollout("r2")
        ctrl.confirm_drained(GOOD_DRAIN)
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)
        self.assertEqual(ctrl.active_rollout_count, 0)

    def test_rollouts_finish_during_drain_but_cannot_be_admitted(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.begin_drain()
        # Finishing is legal in DRAINING...
        ctrl.finish_rollout("r1")
        # ...admitting is not.
        self.assertRaises(NotServingError, ctrl.admit_rollout, "r2")

    def test_drain_evidence_contradiction_taints(self):
        """Local count zero but evidence says non-zero -> bookkeeping disagreement."""
        ctrl = drive_to(LifecycleState.DRAINING)
        ev = DrainEvidence(active_rollouts=3, engine_pause_confirmed=True)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_drained(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_unconfirmed_pause_taints(self):
        ctrl = drive_to(LifecycleState.DRAINING)
        ev = DrainEvidence(active_rollouts=0, engine_pause_confirmed=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_drained(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)


class TestI6FailureSafety(unittest.TestCase):
    """A failed update never publishes its target version."""

    def test_update_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(WeightVersion(1), weights_loaded=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertIsNone(ctrl.pending_version)

    def test_finalize_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(WeightVersion(1), finish_acknowledged=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_data_plane_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(WeightVersion(1), data_plane_complete=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_mismatched_evidence_target_taints(self):
        """Evidence for the wrong version means the engine's version is ambiguous."""
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(WeightVersion(7))
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_bootstrap_without_weight_transfer_taints(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, weight_transfer_initialised=False))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)


class TestI7SingleWriter(unittest.TestCase):
    """RolloutCore is the single lifecycle/weight writer."""

    def test_no_public_setters(self):
        ctrl = drive_to(LifecycleState.READY)
        for attr in (
            "state",
            "current_version",
            "pending_version",
            "active_rollouts",
            "journal",
            "taint_reason",
        ):
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError):
                    setattr(ctrl, attr, "hijacked")

    def test_no_arbitrary_attributes(self):
        ctrl = drive_to(LifecycleState.READY)
        with self.assertRaises(AttributeError):
            ctrl.backdoor = True  # type: ignore[attr-defined]

    def test_seeding_over_a_managed_label_is_refused(self):
        """Bootstrap may only overwrite an unmanaged label."""
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        ev = replace(GOOD_BOOTSTRAP, seeded=True, pre_seed_label="rc-5")
        self.assertIsNotNone(ev.failure_reason())
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_seeding_over_unmanaged_label_is_allowed(self):
        self.assertIsNone(GOOD_BOOTSTRAP.failure_reason())


class TestI8NoAutomaticReconciliation(unittest.TestCase):
    """A version mismatch is a hard failure, never silently repaired."""

    def test_version_mismatch_taints_with_typed_error(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(WeightVersion(1), observed_engine_label="rc-99")
        with self.assertRaises(VersionMismatchError) as ctx:
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctx.exception.expected, WeightVersion(1))
        self.assertEqual(ctx.exception.observed, "rc-99")
        self.assertIn("reconciliation is disabled", str(ctx.exception))

    def test_foreign_label_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(WeightVersion(1), observed_engine_label="default")
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_still_paused_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(WeightVersion(1), is_paused=True)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_unacknowledged_resume_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(WeightVersion(1), resume_acknowledged=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(ev)

    def test_bootstrap_rejects_a_foreign_initial_label(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, observed_engine_label="default"))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_bootstrap_rejects_a_nonzero_initial_label(self):
        """An engine already at rc-7 is not fresh; we do not adopt it silently."""
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, observed_engine_label="rc-7"))

    def test_validation_evidence_for_the_wrong_target_taints(self):
        """Evidence about a different version means we cannot tell what is live."""
        ctrl = drive_to(LifecycleState.VALIDATING)  # v1 pending
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(good_validate(WeightVersion(9)))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_is_tainted_and_is_serving_reflect_state(self):
        ctrl = drive_to(LifecycleState.READY)
        self.assertTrue(ctrl.is_serving)
        self.assertFalse(ctrl.is_tainted)
        ctrl.taint("probe")
        self.assertFalse(ctrl.is_serving)
        self.assertTrue(ctrl.is_tainted)


class TestI9TaintedIsTerminal(unittest.TestCase):
    """V1 does not attempt rollback or resume from TAINTED."""

    def test_tainted_never_returns_to_serving(self):
        ctrl = drive_to(LifecycleState.READY)
        ctrl.admit_rollout("r1")
        ctrl.taint("partial update detected")
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertFalse(ctrl.is_serving)
        # Active rollouts are cleared: we cannot vouch for how they would finish.
        self.assertEqual(ctrl.active_rollout_count, 0)

    def test_no_path_from_tainted_to_ready(self):
        from rolloutcore import LEGAL_TRANSITIONS

        for (src, _event), dst in LEGAL_TRANSITIONS.items():
            if src is LifecycleState.TAINTED:
                self.fail(f"illegal edge out of TAINTED: {src} -> {dst}")

    def test_taint_reason_is_retained(self):
        ctrl = drive_to(LifecycleState.READY)
        ctrl.taint("disk filled during transfer")
        self.assertEqual(ctrl.taint_reason, "disk filled during transfer")


class TestEssentialCorrectnessExperiment(unittest.TestCase):
    """The plan's section-13 experiment, in pure-Python form.

    R1-R3 admitted at v4, update requested, R4 arrives mid-transition, update
    completes to v5, R4 runs at v5. No request spans two versions.
    """

    def test_no_mixed_version_rollout(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(4))
        self.assertEqual(ctrl.current_version, WeightVersion(4))

        r1 = ctrl.admit_rollout("R1")
        r2 = ctrl.admit_rollout("R2")
        r3 = ctrl.admit_rollout("R3")

        ctrl.begin_drain()

        # R4 arrives during drain: rejected, not queued. Queueing would let the
        # engine schedule it after resume under v5 while we had bound it to v4.
        self.assertRaises(NotServingError, ctrl.admit_rollout, "R4")

        ctrl.finish_rollout("R1")
        ctrl.finish_rollout("R2")
        ctrl.finish_rollout("R3")
        ctrl.confirm_drained(GOOD_DRAIN)

        target = WeightVersion(5)
        ctrl.begin_update(target)
        ctrl.confirm_updated(good_update(target))
        ctrl.confirm_invalidated(GOOD_INVALIDATE)
        ctrl.confirm_validated(good_validate(target))

        r4 = ctrl.admit_rollout("R4")

        self.assertEqual([r1.version, r2.version, r3.version], [WeightVersion(4)] * 3)
        self.assertEqual(r4.version, WeightVersion(5))
        # Salts differ across the version boundary, so no prefix-cache reuse.
        self.assertNotEqual(r1.cache_salt, r4.cache_salt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
