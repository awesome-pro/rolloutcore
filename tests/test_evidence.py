# SPDX-License-Identifier: Apache-2.0
"""Evidence validation rules.

Evidence objects are the seam between the pure controller and the adapter. Each
encodes a postcondition that must hold before the controller will advance, so
their ``failure_reason`` logic is tested directly here as well as through the
controller.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from support import GOOD_BOOTSTRAP, IDENTITY_V0, TARGET_V1, target

from rolloutcore import (
    DrainEvidence,
    InvalidateEvidence,
    ResumeEvidence,
    UpdateEvidence,
    ValidateEvidence,
)


class TestBootstrapEvidence(unittest.TestCase):
    def test_good(self):
        self.assertIsNone(GOOD_BOOTSTRAP.failure_reason())

    def test_uninitialised_transfer_fails(self):
        ev = replace(GOOD_BOOTSTRAP, weight_transfer_initialised=False)
        self.assertIn("not initialised", ev.failure_reason())

    def test_missing_pre_seed_label_is_refused(self):
        """Amendment 4: freshness cannot be proven without the pre-seed label."""
        ev = replace(GOOD_BOOTSTRAP, pre_seed_label=None)
        reason = ev.failure_reason()
        self.assertIsNotNone(reason)
        self.assertIn("single-owner", reason)

    def test_adopting_a_managed_label_is_refused(self):
        for managed in ("rc-0", "rc-1", "rc-999"):
            with self.subTest(managed=managed):
                ev = replace(GOOD_BOOTSTRAP, pre_seed_label=managed)
                reason = ev.failure_reason()
                self.assertIn("refusing to claim", reason)
                self.assertIn("lease/recovery", reason)

    def test_every_unmanaged_label_is_seedable(self):
        for unmanaged in ("default", "v1", "", "step-42", "rc", "rc-", "RC-0"):
            with self.subTest(unmanaged=unmanaged):
                ev = replace(GOOD_BOOTSTRAP, pre_seed_label=unmanaged)
                self.assertIsNone(ev.failure_reason())


class TestDrainEvidence(unittest.TestCase):
    def test_completed_with_no_engine_count(self):
        ev = DrainEvidence(engine_drain_completed=True)
        self.assertIsNone(ev.failure_reason())

    def test_completed_with_zero_engine_count(self):
        ev = DrainEvidence(engine_drain_completed=True, engine_active_requests=0)
        self.assertIsNone(ev.failure_reason())

    def test_incomplete_reports_a_reason(self):
        """The controller maps this to ``EvidenceNotReady``, not a taint."""
        ev = DrainEvidence(engine_drain_completed=False)
        self.assertIn("not confirmed", ev.failure_reason())

    def test_self_contradictory_evidence_is_a_failure(self):
        ev = DrainEvidence(engine_drain_completed=True, engine_active_requests=3)
        reason = ev.failure_reason()
        self.assertIn("self-contradictory", reason)
        self.assertIn("3", reason)


class TestUpdateEvidence(unittest.TestCase):
    def _ev(self, **kw):
        base = dict(
            target=TARGET_V1,
            weights_loaded=True,
            finish_acknowledged=True,
            observed_identity=TARGET_V1.identity,
        )
        base.update(kw)
        return UpdateEvidence(**base)

    def test_good(self):
        self.assertIsNone(self._ev().failure_reason())

    def test_weights_not_loaded(self):
        self.assertIn("not all weight chunks", self._ev(weights_loaded=False).failure_reason())

    def test_data_plane_incomplete(self):
        self.assertIn("data plane", self._ev(data_plane_complete=False).failure_reason())

    def test_finish_not_acknowledged(self):
        self.assertIn("not acknowledged", self._ev(finish_acknowledged=False).failure_reason())

    def test_observed_identity_mismatch(self):
        reason = self._ev(observed_identity=IDENTITY_V0).failure_reason()
        self.assertIn("identity", reason)

    def test_unknown_observed_identity_is_tolerated(self):
        """Adapters that cannot read back an identity must not be blocked."""
        self.assertIsNone(self._ev(observed_identity=None).failure_reason())


class TestInvalidateEvidence(unittest.TestCase):
    def test_all_present(self):
        self.assertIsNone(InvalidateEvidence(True, True, True).failure_reason())

    def test_each_missing_is_reported(self):
        cases = {
            "prefix cache": InvalidateEvidence(False, True, True),
            "encoder cache": InvalidateEvidence(True, False, True),
            "multimodal cache": InvalidateEvidence(True, True, False),
        }
        for name, ev in cases.items():
            with self.subTest(name=name):
                reason = ev.failure_reason()
                self.assertIsNotNone(reason)
                self.assertIn(name, reason)

    def test_all_missing_lists_all_three(self):
        reason = InvalidateEvidence(False, False, False).failure_reason()
        for name in ("prefix cache", "encoder cache", "multimodal cache"):
            self.assertIn(name, reason)


class TestValidateEvidence(unittest.TestCase):
    """Pre-resume evidence. The engine is expected to still be paused."""

    def _ev(self, **kw):
        base = dict(
            target=TARGET_V1,
            observed_engine_label=TARGET_V1.label,
            is_paused=True,
            caches_still_clean=True,
        )
        base.update(kw)
        return ValidateEvidence(**base)

    def test_good(self):
        self.assertIsNone(self._ev().failure_reason())

    def test_not_paused_is_a_failure(self):
        """Amendment 1: an unpaused engine here means someone else resumed it."""
        reason = self._ev(is_paused=False).failure_reason()
        self.assertIn("not paused", reason)
        self.assertIn("outside RolloutCore's control", reason)

    def test_version_mismatch(self):
        reason = self._ev(observed_engine_label="rc-3").failure_reason()
        self.assertIn("rc-3", reason)
        self.assertIn("refusing to reconcile", reason)

    def test_foreign_label_is_a_mismatch(self):
        reason = self._ev(observed_engine_label="default").failure_reason()
        self.assertIn("refusing to reconcile", reason)

    def test_dirty_caches_fail(self):
        self.assertIn("repopulated", self._ev(caches_still_clean=False).failure_reason())


class TestResumeEvidence(unittest.TestCase):
    def _ev(self, **kw):
        base = dict(
            target=TARGET_V1,
            resume_acknowledged=True,
            is_paused=False,
            observed_engine_label=TARGET_V1.label,
        )
        base.update(kw)
        return ResumeEvidence(**base)

    def test_good(self):
        self.assertIsNone(self._ev().failure_reason())

    def test_resume_not_acknowledged(self):
        self.assertIn("resume", self._ev(resume_acknowledged=False).failure_reason())

    def test_still_paused(self):
        self.assertIn("still reports paused", self._ev(is_paused=True).failure_reason())

    def test_post_resume_drift(self):
        reason = self._ev(observed_engine_label="rc-9").failure_reason()
        self.assertIn("after resume", reason)

    def test_unknown_post_resume_label_is_tolerated(self):
        self.assertIsNone(self._ev(observed_engine_label=None).failure_reason())


class TestEvidenceImmutability(unittest.TestCase):
    def test_all_evidence_is_frozen(self):
        evs = [
            GOOD_BOOTSTRAP,
            DrainEvidence(engine_drain_completed=True),
            InvalidateEvidence(True, True, True),
        ]
        for ev in evs:
            with self.subTest(ev=type(ev).__name__), self.assertRaises(FrozenInstanceError):
                ev.failure_reason = lambda: None  # type: ignore[method-assign]

    def test_update_target_is_frozen(self):
        t = target(1)
        with self.assertRaises(FrozenInstanceError):
            t.version = target(2).version  # type: ignore[misc]


class TestIdentityIsCarriedEndToEnd(unittest.TestCase):
    """Amendment 3: the identity must ride every target-shaped evidence object."""

    def test_every_target_carrying_evidence_exposes_identity(self):
        objs = [
            UpdateEvidence(target=TARGET_V1, weights_loaded=True, finish_acknowledged=True),
            ValidateEvidence(
                target=TARGET_V1, observed_engine_label=TARGET_V1.label, is_paused=True
            ),
            ResumeEvidence(target=TARGET_V1, resume_acknowledged=True, is_paused=False),
        ]
        for obj in objs:
            with self.subTest(ev=type(obj).__name__):
                self.assertEqual(obj.target.identity, TARGET_V1.identity)
                self.assertNotEqual(obj.target.identity, IDENTITY_V0)


class TestBootstrapEvidenceIsFrozen(unittest.TestCase):
    def test_cannot_mutate(self):
        with self.assertRaises(FrozenInstanceError):
            GOOD_BOOTSTRAP.pre_seed_label = "rc-5"  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
