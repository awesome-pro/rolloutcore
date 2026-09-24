# SPDX-License-Identifier: Apache-2.0
"""Invariant tests.

Each test names the invariant it defends. I1-I6 come from the design in
``docs/state-machine.md``; I7-I10 come from the implementation amendments. That
document also carries the mapping from invariant to enforcing code.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from support import (
    GOOD_BOOTSTRAP,
    IDENTITY_V0,
    IDENTITY_V1,
    TARGET_V1,
    advance_to_version,
    assert_state_unchanged,
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
    DrainDisagreementError,
    EngineTaintedError,
    EvidenceNotReady,
    InvalidateEvidence,
    LifecycleState,
    NotServingError,
    VersionMismatchError,
    WeightVersion,
)


class TestI1VersionAtomicity(unittest.TestCase):
    """An update is either completely committed or not visible."""

    def test_target_does_not_move_until_resume_is_confirmed(self):
        ctrl = bootstrapped()
        ctrl.begin_drain()
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_drained(good_drain())
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.begin_update(TARGET_V1)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_updated(good_update(TARGET_V1))
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_invalidated(good_invalidate())
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        ctrl.confirm_validated(good_validate(TARGET_V1))
        # Amendment 1: validated is NOT committed. Still generation 0.
        self.assertIs(ctrl.state, LifecycleState.RESUMING)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        # Only here.
        ctrl.confirm_resumed(good_resume(TARGET_V1))
        self.assertEqual(ctrl.current_version, WeightVersion(1))

    def test_pending_target_is_cleared_on_commit(self):
        ctrl = drive_to(LifecycleState.RESUMING)
        self.assertEqual(ctrl.pending_target, TARGET_V1)
        ctrl.confirm_resumed(good_resume(TARGET_V1))
        self.assertIsNone(ctrl.pending_target)

    def test_update_target_must_be_strictly_greater(self):
        ctrl = drive_to(LifecycleState.QUIESCED)
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.begin_update(target(0)))
        self.assertEqual(exc.invariant, "I1-MONOTONIC")
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertIsNone(ctrl.pending_target)

    def test_lower_target_is_rejected(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(3))
        ctrl.begin_drain()
        ctrl.confirm_drained(good_drain())
        exc = assert_state_unchanged(self, ctrl, lambda: ctrl.begin_update(target(2)))
        self.assertEqual(exc.invariant, "I1-MONOTONIC")
        self.assertEqual(ctrl.current_version, WeightVersion(3))
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)

    def test_commit_publishes_identity_alongside_version(self):
        ctrl = bootstrapped()
        self.assertEqual(ctrl.current_identity, IDENTITY_V0)
        run_full_cycle(ctrl, TARGET_V1)
        self.assertEqual(ctrl.current_version, WeightVersion(1))
        self.assertEqual(ctrl.current_identity, IDENTITY_V1)


class TestI2AdmissionBinding(unittest.TestCase):
    """A request's weight version and identity are fixed at admission."""

    def test_binding_captures_version_and_identity(self):
        ctrl = bootstrapped()
        advance_to_version(ctrl, WeightVersion(4))
        b = ctrl.admit_rollout("r1")
        self.assertEqual(b.version, WeightVersion(4))
        self.assertEqual(b.weight_identity, ctrl.current_identity)
        self.assertEqual(b.cache_salt, WeightVersion(4).label)
        self.assertEqual(b.target, ctrl.current_target)

    def test_binding_is_frozen(self):
        ctrl = bootstrapped()
        b = ctrl.admit_rollout("r1")
        with self.assertRaises(FrozenInstanceError):
            b.version = WeightVersion(99)  # type: ignore[misc]

    def test_binding_survives_a_later_version_change(self):
        """A rollout admitted at vN still reports vN after the engine moves on."""
        ctrl = bootstrapped()
        b = ctrl.admit_rollout("r1")
        ctrl.finish_rollout("r1")
        advance_to_version(ctrl, WeightVersion(2))
        self.assertEqual(b.version, WeightVersion(0), "binding must not follow the engine")
        self.assertEqual(b.weight_identity, IDENTITY_V0)
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
    """KV created under generation N may not be consumed under N+1."""

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
                self.assertEqual(ctrl.current_version, before)
                self.assertIsNone(ctrl.pending_target)

    def test_all_three_present_passes(self):
        self.assertIsNone(good_invalidate().failure_reason())

    def test_cache_salt_is_generation_scoped(self):
        salts = {WeightVersion(n).cache_salt for n in range(8)}
        self.assertEqual(len(salts), 8)

    def test_caches_repopulated_before_validation_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(TARGET_V1, caches_still_clean=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)


class TestI4ReplayIdentity(unittest.TestCase):
    """A stored trajectory identifies the exact weights that produced it.

    This is why ``WeightIdentity`` exists. A generation number alone cannot
    support the claim: two controllers can publish different weights under the
    same ``rc-N``, so "generation 3" is not a weight identity.
    """

    def test_binding_carries_a_manifest_digest_not_just_a_number(self):
        ctrl = bootstrapped()
        b = ctrl.admit_rollout("r1")
        self.assertEqual(b.version, WeightVersion(0))
        self.assertEqual(b.weight_identity, IDENTITY_V0)
        self.assertNotEqual(b.weight_identity, IDENTITY_V1)

    def test_same_generation_different_weights_is_distinguishable(self):
        """The core reason generation != identity."""
        a = target(1, IDENTITY_V0)
        b = target(1, IDENTITY_V1)
        self.assertEqual(a.version, b.version)
        self.assertNotEqual(a.identity, b.identity)

    def test_distinct_generations_get_distinct_salts_and_identities(self):
        ctrl = bootstrapped()
        seen = []
        for n in (1, 2, 3):
            t = target(n)
            b = ctrl.admit_rollout("r")
            seen.append((b.version, b.cache_salt, b.weight_identity))
            ctrl.finish_rollout("r")
            run_full_cycle(ctrl, t)
        self.assertEqual([v for v, _, _ in seen][1:], [WeightVersion(1), WeightVersion(2)])
        self.assertEqual(len({s for _, s, _ in seen}), 3)
        self.assertEqual(len({i for _, _, i in seen}), 3)

    def test_admission_sequence_is_monotonic(self):
        ctrl = bootstrapped()
        seqs = [ctrl.admit_rollout(f"r{i}").admitted_seq for i in range(4)]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 4)

    def test_identity_is_invariant_across_the_cycle(self):
        """The identity validated pre-resume is the one published."""
        ctrl = drive_to(LifecycleState.VALIDATING)
        ctrl.confirm_validated(good_validate(TARGET_V1))
        self.assertEqual(ctrl.current_identity, IDENTITY_V0, "not committed yet")
        ctrl.confirm_resumed(good_resume(TARGET_V1))
        self.assertEqual(ctrl.current_identity, TARGET_V1.identity)


class TestI5DrainCorrectness(unittest.TestCase):
    """No active old-version requests remain when an update begins."""

    def test_incomplete_drain_is_retryable_not_fatal(self):
        """Amendment 2: before the engine reports completion, active work is normal."""
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        exc = assert_state_unchanged(
            self, ctrl, lambda: ctrl.confirm_drained(good_drain(completed=False))
        )
        self.assertIsInstance(exc, EvidenceNotReady)
        self.assertIs(ctrl.state, LifecycleState.DRAINING)
        self.assertEqual(ctrl.active_rollout_count, 2)

    def test_disagreement_after_engine_completion_taints(self):
        """Amendment 2: engine says drained, we still count work -> taint."""
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        with self.assertRaises(DrainDisagreementError) as ctx:
            ctrl.confirm_drained(good_drain(completed=True))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctx.exception.active_rollouts, 2)
        self.assertEqual(sorted(ctx.exception.request_ids), ["r1", "r2"])
        self.assertIn("disagree", str(ctx.exception))

    def test_agreeing_zero_completes_the_drain(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        ctrl.finish_rollout("r1")
        ctrl.abort_rollout("r2")
        ctrl.confirm_drained(good_drain(completed=True))
        self.assertIs(ctrl.state, LifecycleState.QUIESCED)
        self.assertEqual(ctrl.active_rollout_count, 0)

    def test_self_contradictory_engine_evidence_taints(self):
        """engine_drain_completed=True with engine_active_requests=3 is nonsense."""
        ctrl = drive_to(LifecycleState.DRAINING)
        ev = good_drain(completed=True, engine_active=3)
        self.assertIsNotNone(ev.failure_reason())
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_drained(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_rollouts_finish_during_drain_but_cannot_be_admitted(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.begin_drain()
        ctrl.finish_rollout("r1")
        self.assertRaises(NotServingError, ctrl.admit_rollout, "r2")


class TestI6FailureSafety(unittest.TestCase):
    """A failed update never publishes its target."""

    def test_update_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(TARGET_V1, weights_loaded=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))
        self.assertEqual(ctrl.current_identity, IDENTITY_V0)
        self.assertIsNone(ctrl.pending_target)

    def test_finalize_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(TARGET_V1, finish_acknowledged=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_data_plane_failure_taints_without_publishing(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(TARGET_V1, data_plane_complete=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_wrong_identity_landing_taints(self):
        """The engine reports a different identity than we tried to install."""
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(TARGET_V1, observed_identity=IDENTITY_V0)
        self.assertIsNotNone(ev.failure_reason())
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertEqual(ctrl.current_identity, IDENTITY_V0)

    def test_mismatched_evidence_target_taints(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        ev = good_update(target(7))
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_updated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_bootstrap_without_weight_transfer_taints(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, weight_transfer_initialised=False))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_resume_failure_taints_after_validation_passed(self):
        """Fail-closed: the engine was never serving, so nothing is published."""
        ctrl = drive_to(LifecycleState.RESUMING)
        ev = good_resume(TARGET_V1, resume_acknowledged=False)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_resumed(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_still_paused_after_resume_taints(self):
        ctrl = drive_to(LifecycleState.RESUMING)
        ev = good_resume(TARGET_V1, is_paused=True)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_resumed(ev)
        self.assertEqual(ctrl.current_version, WeightVersion(0))


class TestI7SingleWriter(unittest.TestCase):
    """RolloutCore is the single lifecycle/weight writer."""

    def test_no_public_setters(self):
        ctrl = drive_to(LifecycleState.READY)
        for attr in (
            "state",
            "current_target",
            "current_version",
            "current_identity",
            "pending_target",
            "pending_version",
            "active_rollouts",
            "orphaned_rollouts",
            "journal",
            "taint_reason",
            "tainted_in_state",
        ):
            with self.subTest(attr=attr), self.assertRaises(AttributeError):
                setattr(ctrl, attr, "hijacked")

    def test_no_arbitrary_attributes(self):
        ctrl = drive_to(LifecycleState.READY)
        with self.assertRaises(AttributeError):
            ctrl.backdoor = True  # type: ignore[attr-defined]


class TestI8NoAutomaticReconciliation(unittest.TestCase):
    """A version mismatch is a hard failure, never silently repaired."""

    def test_version_mismatch_during_validation_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(TARGET_V1, observed_engine_label="rc-99")
        with self.assertRaises(VersionMismatchError) as ctx:
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctx.exception.expected, WeightVersion(1))
        self.assertEqual(ctx.exception.observed, "rc-99")
        self.assertIn("reconciliation is disabled", str(ctx.exception))

    def test_foreign_label_during_validation_taints(self):
        ctrl = drive_to(LifecycleState.VALIDATING)
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(good_validate(TARGET_V1, observed_engine_label="default"))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_not_paused_during_validation_taints(self):
        """Amendment 1: VALIDATING requires the engine to still be paused."""
        ctrl = drive_to(LifecycleState.VALIDATING)
        ev = good_validate(TARGET_V1, is_paused=False)
        self.assertIsNotNone(ev.failure_reason())
        with self.assertRaises(EngineTaintedError):
            ctrl.confirm_validated(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_post_resume_version_drift_taints(self):
        ctrl = drive_to(LifecycleState.RESUMING)
        ev = good_resume(TARGET_V1, observed_engine_label="rc-42")
        with self.assertRaises(VersionMismatchError):
            ctrl.confirm_resumed(ev)
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertEqual(ctrl.current_version, WeightVersion(0))

    def test_bootstrap_rejects_a_foreign_initial_label(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, observed_engine_label="default"))
        self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_bootstrap_rejects_a_nonzero_initial_label(self):
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(replace(GOOD_BOOTSTRAP, observed_engine_label="rc-7"))


class TestI9BootstrapIsSingleOwner(unittest.TestCase):
    """Amendment 4: V1 may only seed a fresh, unmanaged engine."""

    def test_seeding_over_unmanaged_label_is_allowed(self):
        self.assertIsNone(GOOD_BOOTSTRAP.failure_reason())

    def test_adopting_a_managed_engine_is_refused(self):
        for managed in ("rc-0", "rc-9", "rc-1000"):
            with self.subTest(managed=managed):
                ctrl = drive_to(LifecycleState.UNINITIALIZED)
                ev = replace(GOOD_BOOTSTRAP, pre_seed_label=managed)
                self.assertIsNotNone(ev.failure_reason())
                with self.assertRaises(EngineTaintedError) as ctx:
                    ctrl.initialize(ev)
                self.assertIn("lease/recovery", str(ctx.exception))
                self.assertIs(ctrl.state, LifecycleState.TAINTED)

    def test_missing_pre_seed_label_is_refused(self):
        """An adapter that does not report the pre-seed label cannot prove freshness."""
        ctrl = drive_to(LifecycleState.UNINITIALIZED)
        ev = replace(GOOD_BOOTSTRAP, pre_seed_label=None)
        self.assertIsNotNone(ev.failure_reason())
        with self.assertRaises(EngineTaintedError):
            ctrl.initialize(ev)


class TestI10TaintedIsTerminal(unittest.TestCase):
    """V1 does not attempt rollback or resume from TAINTED."""

    def test_tainted_never_returns_to_serving(self):
        ctrl = drive_to(LifecycleState.READY)
        ctrl.taint("partial update detected")
        self.assertIs(ctrl.state, LifecycleState.TAINTED)
        self.assertFalse(ctrl.is_serving)

    def test_no_path_from_tainted_to_ready(self):
        from rolloutcore import LEGAL_TRANSITIONS

        for (src, _event), dst in LEGAL_TRANSITIONS.items():
            if src is LifecycleState.TAINTED:
                self.fail(f"illegal edge out of TAINTED: {src} -> {dst}")

    def test_taint_reason_and_state_are_retained(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        tainted_from = ctrl.state.value
        ctrl.taint("disk filled during transfer")
        self.assertEqual(ctrl.taint_reason, "disk filled during transfer")
        self.assertEqual(ctrl.tainted_in_state, tainted_from)

    def test_pending_target_is_cleared_on_taint(self):
        ctrl = drive_to(LifecycleState.UPDATING)
        self.assertIsNotNone(ctrl.pending_target)
        ctrl.taint("abort")
        self.assertIsNone(ctrl.pending_target)


class TestTaintForensics(unittest.TestCase):
    """Amendment 5: tainting must not discard in-flight bindings."""

    def test_active_bindings_are_preserved_not_discarded(self):
        ctrl = bootstrapped()
        b1 = ctrl.admit_rollout("r1")
        b2 = ctrl.admit_rollout("r2")
        ctrl.begin_drain()
        ctrl.taint("drain disagreement")

        # No longer active...
        self.assertEqual(ctrl.active_rollout_count, 0)
        # ...but retained for forensics.
        orphans = ctrl.orphaned_rollouts
        self.assertEqual(len(orphans), 2)
        self.assertEqual({o.request_id for o in orphans}, {"r1", "r2"})
        self.assertEqual({o.binding for o in orphans}, {b1, b2})

    def test_orphans_record_why_and_where(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.begin_drain()
        ctrl.taint("engine unreachable")

        (orphan,) = ctrl.orphaned_rollouts
        self.assertEqual(orphan.orphaned_in_state, LifecycleState.DRAINING.value)
        self.assertEqual(orphan.reason, "engine unreachable")
        self.assertEqual(orphan.version, WeightVersion(0))
        self.assertEqual(orphan.weight_identity, IDENTITY_V0)

    def test_orphans_survive_a_disagreement_taint(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.begin_drain()
        with self.assertRaises(DrainDisagreementError):
            ctrl.confirm_drained(good_drain(completed=True))
        self.assertEqual(len(ctrl.orphaned_rollouts), 1)
        self.assertEqual(ctrl.orphaned_rollouts[0].request_id, "r1")

    def test_dual_taint_keeps_both_the_first_reason_and_the_orphans(self):
        ctrl = bootstrapped()
        ctrl.admit_rollout("r1")
        ctrl.taint("first")
        ctrl.taint("second")
        self.assertEqual(ctrl.taint_reason, "first")
        self.assertEqual(len(ctrl.orphaned_rollouts), 1)

    def test_nothing_to_orphan_is_fine(self):
        ctrl = bootstrapped()
        ctrl.taint("idle abort")
        self.assertEqual(ctrl.orphaned_rollouts, ())


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
        ctrl.confirm_drained(good_drain())

        t = target(5)
        ctrl.begin_update(t)
        ctrl.confirm_updated(good_update(t))
        ctrl.confirm_invalidated(good_invalidate())
        ctrl.confirm_validated(good_validate(t))
        ctrl.confirm_resumed(good_resume(t))

        r4 = ctrl.admit_rollout("R4")

        self.assertEqual([r1.version, r2.version, r3.version], [WeightVersion(4)] * 3)
        self.assertEqual(r4.version, WeightVersion(5))
        self.assertNotEqual(r1.cache_salt, r4.cache_salt)
        self.assertEqual(r1.weight_identity, r2.weight_identity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
