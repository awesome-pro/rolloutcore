# SPDX-License-Identifier: Apache-2.0
"""Evidence validation rules.

Evidence objects are the seam between the pure controller and the Phase 2
adapter. Each one encodes a postcondition that must hold before the controller
will advance, so their ``failure_reason`` logic is tested directly here as well
as through the controller.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from rolloutcore import (
    BootstrapEvidence,
    DrainEvidence,
    InvalidateEvidence,
    UpdateEvidence,
    ValidateEvidence,
    WeightVersion,
)
from support import GOOD_BOOTSTRAP


class TestBootstrapEvidence(unittest.TestCase):
    def test_good(self):
        self.assertIsNone(GOOD_BOOTSTRAP.failure_reason())

    def test_uninitialised_transfer_fails(self):
        ev = replace(GOOD_BOOTSTRAP, weight_transfer_initialised=False)
        self.assertIn("not initialised", ev.failure_reason())

    def test_pre_seed_label_without_seeded_flag_fails(self):
        ev = replace(GOOD_BOOTSTRAP, seeded=False, pre_seed_label="default")
        self.assertIn("seeded is False", ev.failure_reason())

    def test_seeded_without_pre_seed_label_fails(self):
        ev = replace(GOOD_BOOTSTRAP, seeded=True, pre_seed_label=None)
        self.assertIn("did not report the pre-seed label", ev.failure_reason())

    def test_not_seeded_and_no_pre_seed_label_is_fine(self):
        """An engine already carrying rc-0 needs no seeding."""
        ev = BootstrapEvidence(
            observed_engine_label="rc-0", weight_transfer_initialised=True
        )
        self.assertIsNone(ev.failure_reason())

    def test_seeding_over_managed_label_fails(self):
        for managed in ("rc-0", "rc-9"):
            with self.subTest(managed=managed):
                ev = replace(GOOD_BOOTSTRAP, pre_seed_label=managed)
                self.assertIn("refusing to seed", ev.failure_reason())

    def test_seeding_over_each_unmanaged_label_is_allowed(self):
        for unmanaged in ("default", "v1", "", "step-42"):
            with self.subTest(unmanaged=unmanaged):
                ev = replace(GOOD_BOOTSTRAP, pre_seed_label=unmanaged)
                self.assertIsNone(ev.failure_reason())


class TestDrainEvidence(unittest.TestCase):
    def test_good(self):
        self.assertIsNone(
            DrainEvidence(active_rollouts=0, engine_pause_confirmed=True).failure_reason()
        )

    def test_unconfirmed(self):
        ev = DrainEvidence(active_rollouts=0, engine_pause_confirmed=False)
        self.assertIn("did not confirm", ev.failure_reason())

    def test_contradictory_count(self):
        ev = DrainEvidence(active_rollouts=2, engine_pause_confirmed=True)
        self.assertIn("2 rollouts are active", ev.failure_reason())


class TestUpdateEvidence(unittest.TestCase):
    def _ev(self, **kw):
        base = dict(
            target_version=WeightVersion(1),
            weights_loaded=True,
            finish_acknowledged=True,
        )
        base.update(kw)
        return UpdateEvidence(**base)

    def test_good(self):
        self.assertIsNone(self._ev().failure_reason())

    def test_weights_not_loaded(self):
        self.assertIn("not all weight chunks", self._ev(weights_loaded=False).failure_reason())

    def test_data_plane_incomplete(self):
        self.assertIn(
            "data plane", self._ev(data_plane_complete=False).failure_reason()
        )

    def test_finish_not_acknowledged(self):
        self.assertIn(
            "not acknowledged", self._ev(finish_acknowledged=False).failure_reason()
        )


class TestInvalidateEvidence(unittest.TestCase):
    def test_all_present(self):
        ev = InvalidateEvidence(True, True, True)
        self.assertIsNone(ev.failure_reason())

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
    def _ev(self, **kw):
        base = dict(
            target_version=WeightVersion(2),
            observed_engine_label="rc-2",
            resume_acknowledged=True,
            is_paused=False,
        )
        base.update(kw)
        return ValidateEvidence(**base)

    def test_good(self):
        self.assertIsNone(self._ev().failure_reason())

    def test_resume_not_acknowledged(self):
        self.assertIn("resume", self._ev(resume_acknowledged=False).failure_reason())

    def test_still_paused(self):
        self.assertIn("still reports paused", self._ev(is_paused=True).failure_reason())

    def test_version_mismatch(self):
        reason = self._ev(observed_engine_label="rc-3").failure_reason()
        self.assertIn("rc-3", reason)
        self.assertIn("refusing to reconcile", reason)

    def test_foreign_label_is_a_mismatch(self):
        reason = self._ev(observed_engine_label="default").failure_reason()
        self.assertIn("refusing to reconcile", reason)


class TestEvidenceImmutability(unittest.TestCase):
    def test_all_evidence_is_frozen(self):
        evs = [
            GOOD_BOOTSTRAP,
            DrainEvidence(active_rollouts=0, engine_pause_confirmed=True),
            InvalidateEvidence(True, True, True),
        ]
        for ev in evs:
            with self.subTest(ev=type(ev).__name__):
                with self.assertRaises(Exception):
                    ev.failure_reason = lambda: None  # type: ignore[method-assign]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
