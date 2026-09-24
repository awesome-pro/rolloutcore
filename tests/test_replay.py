# SPDX-License-Identifier: Apache-2.0
"""Item 8: the replay validator, which needs no engine and no GPU.

Every number the verdict reports is pinned here, because the verdict is the whole
point of the module: an operator reading "AGREE" or "DISAGREE" has to be able to
trust that the arithmetic behind it is the arithmetic they expect. The logprob
statistics come with hand-computed expectations in the comments.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from support import IDENTITY_V0
from test_trajectory import binding, identity_for, trajectory

from rolloutcore import (
    ReplayError,
    ReplayObservation,
    ReplayPlan,
    WeightProvenance,
    validate_replay,
)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "validate_replay.py"

STEP0 = WeightProvenance(checkpoint="facebook/opt-125m", run_id="run-7", step=0)


def planned(
    token_ids: tuple[int, ...] = (5, 812, 9),
    logprobs: tuple[float, ...] = (),
    prompt: str = "The capital of France is",
) -> ReplayPlan:
    """A plan for a declared-source record, ready to be replayed."""
    record = trajectory(
        binding=binding(version=0, identity=identity_for(STEP0)),
        prompt=prompt,
        token_ids=token_ids,
        logprobs=logprobs,
    )
    return ReplayPlan.from_trajectory(record)


def observed(
    token_ids: tuple[int, ...] = (5, 812, 9),
    logprobs: tuple[float, ...] = (),
    *,
    version: str = "rc-0",
    identity_digest: str | None = None,
    prompt: str | None = None,
    request_id: str = "R1",
    tokens_were_forced: bool = False,
) -> ReplayObservation:
    return ReplayObservation(
        request_id=request_id,
        version=version,
        token_ids=token_ids,
        logprobs=logprobs,
        identity_digest=identity_digest,
        prompt=prompt,
        tokens_were_forced=tokens_were_forced,
    )


class TestPlan:
    def test_a_manifest_only_record_cannot_be_planned(self):
        """Item 5's gate: an architecture is not a training step."""
        manifest_only = trajectory(binding=binding(identity=IDENTITY_V0))
        with pytest.raises(ReplayError) as ctx:
            ReplayPlan.from_trajectory(manifest_only)
        message = str(ctx.value)
        self_digest = IDENTITY_V0.short
        assert self_digest in message, message
        assert "manifest-only" in message
        assert "declared provenance" in message

    def test_plan_carries_the_record(self):
        plan = planned(token_ids=(1, 2, 3), logprobs=(-0.5, -1.5, -2.5))
        assert plan.request_id == "R1"
        assert plan.prompt == "The capital of France is"
        assert plan.expected_token_ids == (1, 2, 3)
        assert plan.version == "rc-0"
        assert plan.identity_digest == identity_for(STEP0).digest
        assert plan.cache_salt == "rc-0"
        assert plan.recorded_logprobs == (-0.5, -1.5, -2.5)
        assert plan.has_expected_tokens is True
        assert plan.max_tokens == 3

    def test_an_empty_record_pins_nothing(self):
        plan = planned(token_ids=())
        assert plan.has_expected_tokens is False
        assert plan.max_tokens == 0

    def test_describe_names_the_source_and_the_lane(self):
        with_logprobs = planned(token_ids=(1, 2), logprobs=(-0.1, -0.2)).describe()
        assert with_logprobs.startswith("R1 rc-0 ")
        assert identity_for(STEP0).short in with_logprobs
        assert "expected 2 tokens" in with_logprobs
        assert "2 recorded logprobs" in with_logprobs

        token_only = planned(token_ids=(1, 2)).describe()
        assert "no recorded logprobs (token lane only)" in token_only


class TestProtocol:
    """Each violation names what went wrong and forces disagreement."""

    def test_identical_replay_agrees(self):
        verdict = validate_replay(planned(), observed())
        assert verdict.agrees is True
        assert verdict.protocol == ()
        assert verdict.token_mismatches == 0
        assert verdict.first_divergence is None
        assert verdict.shared_prefix == 3

    def test_version_mismatch_is_reported_and_refuses_to_compare(self):
        verdict = validate_replay(planned(), observed(version="rc-1"))
        assert verdict.agrees is False
        assert len(verdict.protocol) == 1
        assert "rc-0" in verdict.protocol[0] and "rc-1" in verdict.protocol[0]
        assert "meaningless" in verdict.protocol[0]

    def test_digest_mismatch_is_reported(self):
        verdict = validate_replay(planned(), observed(identity_digest="sha256:" + "ab" * 32))
        assert verdict.agrees is False
        assert len(verdict.protocol) == 1
        assert "not the same weight source" in verdict.protocol[0]

    def test_an_absent_digest_is_unknown_not_wrong(self):
        """A bare completion API returns no digest; that is not a violation."""
        verdict = validate_replay(planned(), observed(identity_digest=None))
        assert verdict.agrees is True
        assert verdict.protocol == ()

    def test_prompt_mismatch_is_reported(self):
        verdict = validate_replay(planned(), observed(prompt="The capital of Italy is"))
        assert verdict.agrees is False
        assert "prompt" in verdict.protocol[0]
        assert "Italy" in verdict.protocol[0]

    def test_an_absent_prompt_is_unknown_not_wrong(self):
        verdict = validate_replay(planned(), observed(prompt=None))
        assert verdict.agrees is True

    def test_early_stop_is_a_violation(self):
        verdict = validate_replay(planned(), observed(token_ids=(5, 812)))
        assert verdict.agrees is False
        assert any("early stop" in violation for violation in verdict.protocol)
        assert verdict.token_mismatches == 0
        assert verdict.first_divergence == 2, "the first position never compared"
        assert verdict.shared_prefix == 2

    def test_a_longer_replay_is_not_a_protocol_violation_but_does_not_agree(self):
        """The engine may have kept sampling; the record still didn't replay as written."""
        verdict = validate_replay(planned(), observed(token_ids=(5, 812, 9, 1470)))
        assert verdict.protocol == ()
        assert verdict.token_mismatches == 0
        assert verdict.first_divergence == 3
        assert verdict.shared_prefix == 3
        assert verdict.agrees is False
        assert any("beyond the expected sequence" in note for note in verdict.notes)


class TestTokens:
    def test_divergence_at_index_k(self):
        verdict = validate_replay(planned(), observed(token_ids=(5, 812, 1470)))
        assert verdict.token_mismatches == 1
        assert verdict.first_divergence == 2
        assert verdict.shared_prefix == 2
        assert verdict.agrees is False
        assert verdict.protocol == ()

    def test_two_divergences_count_both_but_report_the_first(self):
        verdict = validate_replay(planned(), observed(token_ids=(5, 999, 999)))
        assert verdict.token_mismatches == 2
        assert verdict.first_divergence == 1
        assert verdict.shared_prefix == 1

    def test_divergence_at_index_zero(self):
        verdict = validate_replay(planned(), observed(token_ids=(999, 812, 9)))
        assert verdict.token_mismatches == 1
        assert verdict.first_divergence == 0
        assert verdict.shared_prefix == 0


class TestLogprobLane:
    def test_the_four_lanes(self):
        recorded = (-0.5, -1.5, -2.5)
        assert (
            validate_replay(planned(logprobs=recorded), observed(logprobs=recorded)).logprob_lane
            == "compared"
        )
        assert validate_replay(planned(), observed(logprobs=recorded)).logprob_lane == (
            "record-missing"
        )
        assert validate_replay(planned(logprobs=recorded), observed()).logprob_lane == (
            "replay-missing"
        )
        assert validate_replay(planned(), observed()).logprob_lane == "both-missing"

    def test_a_token_only_verdict_says_so_and_reports_no_statistics(self):
        verdict = validate_replay(planned(logprobs=(-0.5, -1.5, -2.5)), observed())
        assert verdict.logprob_lane == "replay-missing"
        assert verdict.compared_logprobs == 0
        assert verdict.mean_abs_delta is None
        assert verdict.max_abs_delta is None
        assert verdict.p99_abs_delta is None
        assert verdict.worst_index is None
        assert any("token-only" in note for note in verdict.notes)
        assert verdict.agrees is True, "no logprobs on one side is not a disagreement"

    def test_statistics_are_hand_computed(self):
        # deltas: |(-0.10) - (-0.13)| = 0.03, |(-1.00) - (-1.30)| = 0.30,
        #         |(-2.50) - (-2.55)| = 0.05, |(-3.00) - (-3.00)| = 0.00
        # mean = 0.38 / 4 = 0.095 ; max = 0.30 at index 1
        # sorted = [0.00, 0.03, 0.05, 0.30] ; p99 nearest-rank = ceil(0.99*4)-1 = 3 -> 0.30
        verdict = validate_replay(
            planned((1, 2, 3, 4), logprobs=(-0.10, -1.00, -2.50, -3.00)),
            observed((1, 2, 3, 4), logprobs=(-0.13, -1.30, -2.55, -3.00)),
        )
        assert verdict.logprob_lane == "compared"
        assert verdict.compared_logprobs == 4
        assert verdict.mean_abs_delta == pytest.approx(0.095)
        assert verdict.max_abs_delta == pytest.approx(0.30)
        assert verdict.worst_index == 1
        assert verdict.p99_abs_delta == pytest.approx(0.30)

    def test_p99_is_nearest_rank_not_interpolated(self):
        # 10 deltas 0.01 .. 0.10 ; sorted ; ceil(0.99*10)-1 = 9 -> the largest, 0.10
        recorded = tuple(-1.0 for _ in range(10))
        observed_values = tuple(-1.0 + 0.01 * (i + 1) for i in range(10))
        verdict = validate_replay(
            planned(tuple(range(10)), logprobs=recorded),
            observed(tuple(range(10)), logprobs=observed_values),
        )
        assert verdict.p99_abs_delta == pytest.approx(0.10)
        assert verdict.mean_abs_delta == pytest.approx(0.055)

    def test_one_delta_p99_is_that_delta(self):
        verdict = validate_replay(planned((1,), logprobs=(-1.0,)), observed((1,), logprobs=(-1.4,)))
        assert verdict.p99_abs_delta == pytest.approx(0.4)
        assert verdict.max_abs_delta == pytest.approx(0.4)
        assert verdict.worst_index == 0


class TestTolerance:
    def test_just_inside_the_tolerance_agrees(self):
        verdict = validate_replay(
            planned((1,), logprobs=(-1.0,)),
            observed((1,), logprobs=(-1.0009,)),
            logprob_tolerance=1e-3,
        )
        assert verdict.max_abs_delta == pytest.approx(0.0009)
        assert verdict.agrees is True
        assert not any("exceeds the tolerance" in note for note in verdict.notes)

    def test_just_outside_the_tolerance_disagrees_and_says_why(self):
        verdict = validate_replay(
            planned((1,), logprobs=(-1.0,)),
            observed((1,), logprobs=(-1.0011,)),
            logprob_tolerance=1e-3,
        )
        assert verdict.max_abs_delta == pytest.approx(0.0011)
        assert verdict.agrees is False
        assert verdict.tolerance == pytest.approx(1e-3)
        assert any("exceeds the tolerance" in note for note in verdict.notes)

    def test_the_tolerance_is_recorded_on_the_verdict(self):
        verdict = validate_replay(planned(), observed(), logprob_tolerance=0.25)
        assert verdict.tolerance == pytest.approx(0.25)

    def test_a_logprob_delta_cannot_be_rescued_by_matching_tokens(self):
        verdict = validate_replay(
            planned((1, 2), logprobs=(-1.0, -2.0)),
            observed((1, 2), logprobs=(-1.0, -9.0)),
        )
        assert verdict.token_mismatches == 0
        assert verdict.agrees is False


class TestDescribe:
    def test_agreement_line(self):
        line = validate_replay(planned(), observed()).describe()
        assert line == "R1 AGREE: shared prefix 3, 0 token mismatch(es), token-only (both-missing)"

    def test_disagreement_line_reports_the_first_divergence_and_the_delta(self):
        line = validate_replay(
            planned((1, 2, 3), logprobs=(-1.0, -1.0, -1.0)),
            observed((1, 9, 3), logprobs=(-1.0, -4.0, -1.0)),
        ).describe()
        assert "R1 DISAGREE" in line
        assert "shared prefix 1" in line
        assert "1 token mismatch(es)" in line
        assert "first divergence at 1" in line
        assert "max|Δ|=3.000e+00" in line


class TestReplayModes:
    """A token mismatch means different things in the two modes.

    A trace-forced replay dictates the ids, so failing to echo them is a fault in
    the replay; a greedy replay only expects them, so a mismatch is the finding.
    Reading one as the other is how a harness bug becomes a false weight verdict.
    """

    def test_a_trace_forced_replay_that_echoes_agrees(self):
        verdict = validate_replay(
            planned((5, 812, 9)), observed((5, 812, 9), tokens_were_forced=True)
        )
        assert verdict.agrees is True
        assert verdict.protocol == ()
        assert verdict.token_mismatches == 0
        assert not any("echo" in note for note in verdict.notes)

    def test_a_trace_forced_replay_that_does_not_echo_is_a_protocol_violation(self):
        """Identical lengths, different ids: cannot be a weight finding."""
        verdict = validate_replay(
            planned((5, 812, 9)), observed((5, 812, 1470), tokens_were_forced=True)
        )
        assert verdict.agrees is False
        assert any(violation.startswith("echo:") for violation in verdict.protocol)
        echo = next(v for v in verdict.protocol if v.startswith("echo:"))
        assert "says nothing about the weights" in echo
        assert verdict.token_mismatches == 1, "the counts are still reported for diagnosis"

    def test_a_trace_forced_replay_with_the_wrong_length_is_a_violation(self):
        verdict = validate_replay(planned((5, 812, 9)), observed((5, 812), tokens_were_forced=True))
        assert verdict.agrees is False
        assert any(violation.startswith("echo:") for violation in verdict.protocol)

    def test_a_greedy_replay_with_a_mismatch_reports_the_mismatch_as_the_finding(self):
        verdict = validate_replay(planned((5, 812, 9)), observed((5, 812, 1470)))
        assert verdict.agrees is False
        assert verdict.protocol == (), "nothing was dictated, so nothing failed to echo"
        assert verdict.token_mismatches == 1
        assert verdict.first_divergence == 2
        assert any(
            "the weights did not reproduce their own argmax" in note for note in verdict.notes
        )

    def test_a_greedy_replay_that_matches_agrees(self):
        verdict = validate_replay(planned((5, 812, 9)), observed((5, 812, 9)))
        assert verdict.agrees is True
        assert not any("argmax" in note for note in verdict.notes)


class TestCli:
    """End to end, through a real process, with no PYTHONPATH to lean on."""

    @staticmethod
    def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            check=False,
        )

    def test_agreement_exits_zero(self, tmp_path: Path):
        trajectories = tmp_path / "trajectories.jsonl"
        replays = tmp_path / "replays.jsonl"
        trajectories.write_text(
            json.dumps(
                trajectory(
                    binding=binding(identity=identity_for(STEP0)), token_ids=(1, 2)
                ).to_json()
            )
            + "\n",
            encoding="utf-8",
        )
        replays.write_text(
            json.dumps({"request_id": "R1", "version": "rc-0", "token_ids": [1, 2]}) + "\n",
            encoding="utf-8",
        )

        result = self._run(
            "--trajectories",
            str(trajectories),
            "--replays",
            str(replays),
            "--json-out",
            str(tmp_path / "out.json"),
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "AGREE" in result.stdout
        assert "PASS: every record replayed and agreed" in result.stdout
        artifact = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert artifact["summary"]["agreed"] == 1
        assert artifact["summary"]["max_abs_delta"] is None

    def test_disagreement_exits_one(self, tmp_path: Path):
        trajectories = tmp_path / "trajectories.jsonl"
        replays = tmp_path / "replays.jsonl"
        trajectories.write_text(
            json.dumps(
                trajectory(
                    binding=binding(identity=identity_for(STEP0)), token_ids=(1, 2)
                ).to_json()
            )
            + "\n",
            encoding="utf-8",
        )
        replays.write_text(
            json.dumps({"request_id": "R1", "version": "rc-0", "token_ids": [1, 999]}) + "\n",
            encoding="utf-8",
        )

        result = self._run(
            "--trajectories",
            str(trajectories),
            "--replays",
            str(replays),
            "--no-json",
            cwd=tmp_path,
        )
        assert result.returncode == 1, result.stdout
        assert "DISAGREE" in result.stdout
        assert "FAIL: 1 problem(s)" in result.stdout

    def test_tokens_were_forced_round_trips_through_the_cli(self, tmp_path: Path):
        """The mode decides how a mismatch is read, so it has to survive the file."""
        trajectories = tmp_path / "trajectories.jsonl"
        trajectories.write_text(
            json.dumps(
                trajectory(
                    binding=binding(identity=identity_for(STEP0)), token_ids=(1, 2, 3)
                ).to_json()
            )
            + "\n",
            encoding="utf-8",
        )

        echoing = tmp_path / "echoing.jsonl"
        echoing.write_text(
            json.dumps(
                {
                    "request_id": "R1",
                    "version": "rc-0",
                    "token_ids": [1, 2, 3],
                    "tokens_were_forced": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ok = self._run(
            "--trajectories",
            str(trajectories),
            "--replays",
            str(echoing),
            "--json-out",
            str(tmp_path / "ok.json"),
            cwd=tmp_path,
        )
        assert ok.returncode == 0, ok.stdout
        artifact = json.loads((tmp_path / "ok.json").read_text(encoding="utf-8"))
        assert artifact["summary"]["trace_forced"] == 1
        assert artifact["summary"]["greedy"] == 0

        not_echoing = tmp_path / "not-echoing.jsonl"
        not_echoing.write_text(
            json.dumps(
                {
                    "request_id": "R1",
                    "version": "rc-0",
                    "token_ids": [1, 2, 999],
                    "tokens_were_forced": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bad = self._run(
            "--trajectories",
            str(trajectories),
            "--replays",
            str(not_echoing),
            "--no-json",
            cwd=tmp_path,
        )
        assert bad.returncode == 1, bad.stdout
        assert "DISAGREE" in bad.stdout
        assert "echo:" in bad.stdout
        assert "says nothing about the weights" in bad.stdout

    def test_the_committed_artifact_demos_the_token_lane(self, tmp_path: Path):
        """The documented demo command, run for real, against real repository files."""
        result = self._run(
            "--trajectories",
            str(REPO / "results" / "phase5-trajectories.jsonl"),
            "--replays",
            str(REPO / "results" / "phase5-replay-synthetic.jsonl"),
            "--json-out",
            str(tmp_path / "verdicts.json"),
            "--quiet",
            cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        assert "2/2 replay(ies) agreed" in result.stdout

        artifact = json.loads((tmp_path / "verdicts.json").read_text(encoding="utf-8"))
        assert artifact["summary"]["logprob_lanes"] == {"both-missing": 2}
        assert artifact["summary"]["token_mismatches"] == 0
        assert {v["request_id"] for v in artifact["verdicts"]} == {"R-long", "R2"}
