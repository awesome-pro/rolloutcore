# SPDX-License-Identifier: Apache-2.0
"""The Phase 3A harness, proven against the in-repo stub dev server.

This does not replace the real-GPU run -- it exists so that the harness's own
logic (ordering, assertions, JSON artifact, failure paths) is verified before
anyone pays for a GPU. `tests/fake_dev_server.py` models the vLLM behaviours
Phase 3A depends on, including the ones that are easy to get wrong:

* `pause(mode="wait")` sets paused first and then waits for in-flight work;
* the in-process engine rejects `mode="wait"` with HTTP 400;
* `/reset_prefix_cache` returns `{"success": false}` while requests are active;
* the weight-transfer endpoints are not implemented at all, so a Phase 3A run
  that touched them would fail loudly.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import live_control_plane_smoke as harness  # noqa: E402
from fake_dev_server import running  # noqa: E402

STUB_SHA = "00b7847c8036b667742b4efb21aab1de51fd4721"


def make_config(tmp: Path, base_url: str, **overrides: object) -> harness.Config:
    kwargs: dict = {
        "base_url": base_url,
        "model": "stub-model",
        "vllm_sha": STUB_SHA,
        "json_out": tmp / "phase3a.json",
        "baseline_max_tokens": 8,
        "long_max_tokens": 24,
        "drain_timeout": 30.0,
        "generation_timeout": 30.0,
        "drain_interval": 0.01,
    }
    kwargs.update(overrides)
    return harness.Config(**kwargs)


class TestHarnessAgainstStub(unittest.TestCase):
    def test_full_phase3a_sequence_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, token_delay=0.005) as server:
                code = harness.run(make_config(tmp, server.base_url))
            self.assertEqual(code, 0, "the harness failed against a faithful stub")

            report = json.loads((tmp / "phase3a.json").read_text())
            self.assertTrue(report["ok"])
            self.assertEqual(report["phase"], "3A")
            self.assertEqual(report["failed_checks"], [])
            self.assertEqual(report["environment"]["vllm_sha_matches_report"], True)
            self.assertEqual(report["controller"]["final_state"], "TAINTED")
            self.assertTrue(report["controller"]["expected_final_state"] == "TAINTED")

            statuses = {c["name"]: c["status"] for c in report["checks"]}
            # Every requirement the planner listed must be a passing check.
            for name in (
                "fresh_weight_info_is_default",
                "control_plane_bootstrap_seeds_rc0",
                "second_bootstrap_refused_without_write",
                "deterministic_generation_before_pause",
                "pause_is_nonblocking_on_the_rolloutcore_side",
                "pause_waits_for_inflight_request",
                "engine_reports_paused",
                "cache_resets_succeed",
                "pre_resume_validation_sees_rc0_and_paused",
                "resume_succeeds",
                "deterministic_generation_after_resume",
            ):
                self.assertEqual(statuses.get(name), "pass", f"{name} did not pass")

            # The artifact carries the fields the review asked for.
            for key in ("vllm_sha", "rolloutcore_sha", "gpu", "model", "timings"):
                self.assertIn(key, report)
            self.assertTrue(report["generation"]["identical_to_baseline"])
            self.assertEqual(report["checks"][0]["name"], "environment")

    def test_the_drain_really_waited(self):
        """`pause_waits_for_inflight_request` must see a full-length request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, token_delay=0.005) as server:
                harness.run(make_config(tmp, server.base_url, long_max_tokens=40))
            report = json.loads((tmp / "phase3a.json").read_text())
            drain = next(
                c for c in report["checks"] if c["name"] == "pause_waits_for_inflight_request"
            )
            self.assertGreaterEqual(drain["data"]["tokens"], 40)
            nonblocking = next(
                c
                for c in report["checks"]
                if c["name"] == "pause_is_nonblocking_on_the_rolloutcore_side"
            )
            self.assertLess(nonblocking["data"]["first_await_drain_s"], 0.5)

    def test_post_resume_drift_is_caught(self):
        """The determinism check must fail when the engine changes its answer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, token_delay=0.005, drift_after_resume=True) as server:
                code = harness.run(make_config(tmp, server.base_url))
            self.assertEqual(code, 1, "drifting output must fail the harness")
            report = json.loads((tmp / "phase3a.json").read_text())
            self.assertFalse(report["ok"])
            self.assertIn("deterministic_generation_after_resume", report["failed_checks"])

    def test_post_resume_drift_can_be_downgraded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, token_delay=0.005, drift_after_resume=True) as server:
                code = harness.run(make_config(tmp, server.base_url, strict_generation=False))
            self.assertEqual(code, 0, "non-strict mode should record drift as a warning")
            report = json.loads((tmp / "phase3a.json").read_text())
            self.assertIn("deterministic_generation_after_resume", report["warned_checks"])
            self.assertFalse(report["generation"]["identical_to_baseline"])

    def test_managed_engine_is_refused_and_the_label_survives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, initial_label="rc-3") as server:
                code = harness.run(make_config(tmp, server.base_url))
                # A warm server is not fresh: the run must refuse to proceed.
                self.assertEqual(code, 1)
                report = json.loads((tmp / "phase3a.json").read_text())
                self.assertIn("fresh_weight_info_is_default", report["failed_checks"])
                self.assertEqual(server.state.weight_version, "rc-3", "label must be untouched")

    def test_warm_server_can_be_reset_explicitly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, initial_label="rc-3", token_delay=0.005) as server:
                code = harness.run(make_config(tmp, server.base_url, reset_label_to_default=True))
            self.assertEqual(code, 0)
            report = json.loads((tmp / "phase3a.json").read_text())
            self.assertIn("label_reset_to_default", report["warned_checks"])

    def test_inproc_engine_rejection_is_reported(self):
        """`mode=wait` on an in-process engine: the harness must not claim success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, reject_wait_mode=True) as server:
                code = harness.run(make_config(tmp, server.base_url))
            self.assertEqual(code, 1)
            report = json.loads((tmp / "phase3a.json").read_text())
            self.assertFalse(report["ok"])
            self.assertTrue(
                any("inproc" in c["detail"] for c in report["checks"] if c["status"] == "fail")
            )
            # And the weight-transfer surface must never have been touched.
            self.assertEqual(server.state.calls_for("/start_weight_update"), 0)
            self.assertEqual(server.state.calls_for("/init_weight_transfer_engine"), 0)

    def test_no_weight_transfer_calls_at_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with running(vllm_sha=STUB_SHA, token_delay=0.005) as server:
                harness.run(make_config(tmp, server.base_url))
                for path in (
                    "/init_weight_transfer_engine",
                    "/start_weight_update",
                    "/update_weights",
                    "/finish_weight_update",
                ):
                    self.assertEqual(server.state.calls_for(path), 0, f"{path} was called")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
