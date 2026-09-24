# SPDX-License-Identifier: Apache-2.0
"""Item 5: trajectory records that say which weights produced the tokens.

The claim a trajectory has to support is narrow, and Phase 3C and 4C both showed
why: an identity derived only from the parameter manifest cannot separate two
checkpoints of the same architecture. Phase 3C measured the *same* digest
(``925369d663bc``) for dummy-initialised weights and the real ones; Phase 4C
measured it surviving a deliberately corrupted checkpoint. So the tests here care
most about the case where the manifest is identical and only the *declared
provenance* differs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from support import IDENTITY_V0

from rolloutcore import (
    ParamSpec,
    RolloutBinding,
    Trajectory,
    TrajectoryError,
    TrajectoryRecorder,
    WeightIdentity,
    WeightProvenance,
    WeightVersion,
)

REPO = Path(__file__).resolve().parent.parent

STEP0 = WeightProvenance(checkpoint="facebook/opt-125m", run_id="run-7", step=0)
STEP1 = WeightProvenance(checkpoint="facebook/opt-125m", run_id="run-7", step=1)

#: One architecture. `manifest_identity(seed)` varies a parameter *name*, which
#: changes the manifest -- the opposite of the case that matters here, where the
#: manifest is identical and only the declared step differs.
SPECS = (
    ParamSpec("model.embed_tokens.weight", "bfloat16", (151936, 2048)),
    ParamSpec("model.layers.0.self_attn.q_proj.weight", "bfloat16", (2048, 2048)),
)


def identity_for(provenance: WeightProvenance | None) -> WeightIdentity:
    return WeightIdentity.from_param_specs(SPECS, source=provenance)


def binding(
    request_id: str = "R1",
    version: int = 0,
    identity: WeightIdentity | None = None,
) -> RolloutBinding:
    v = WeightVersion(version)
    return RolloutBinding(
        request_id=request_id,
        version=v,
        weight_identity=identity if identity is not None else IDENTITY_V0,
        cache_salt=v.cache_salt,
        admitted_seq=1,
    )


def trajectory(**overrides: object) -> Trajectory:
    kwargs: dict = dict(
        binding=binding(),
        prompt="The capital of France is",
        text=" Paris",
        token_ids=(5, 812),
        finish_reason="length",
        seconds=0.1,
    )
    kwargs.update(overrides)
    return Trajectory(**kwargs)  # type: ignore[arg-type]


class TestWhatATrajectoryClaims:
    def test_a_manifest_only_identity_is_not_replay_ready(self):
        """The honest gate: an architecture is not a training step."""
        t = trajectory()
        assert t.exactness == "manifest-only"
        assert t.replay_ready is False
        assert "[NOT replay-ready]" in t.describe()

    def test_a_declared_source_is_replay_ready(self):
        identity = identity_for(STEP0)
        t = trajectory(binding=binding(identity=identity), engine_version_at_admission="rc-0")
        assert t.exactness == "declared-source"
        assert t.replay_ready is True
        assert t.provenance == STEP0

    def test_two_steps_of_one_architecture_are_distinguishable(self):
        """Item 5 in one test: same manifest, same size, different identity."""
        v0 = identity_for(STEP0)
        v1 = identity_for(STEP1)

        assert v0.manifest_digest == v1.manifest_digest, "same architecture"
        assert v0.num_tensors == v1.num_tensors
        assert v0.digest != v1.digest, "different declared steps"

        first = trajectory(binding=binding(version=0, identity=v0))
        second = trajectory(binding=binding("R2", version=1, identity=v1))
        assert first.identity != second.identity
        assert first.replay_ready and second.replay_ready

    def test_the_admission_check_is_unknown_without_an_observation(self):
        assert trajectory().admission_agrees is None

    def test_the_admission_check_compares_against_the_engine(self):
        """The binding is our bookkeeping; only the engine can corroborate it."""
        agrees = trajectory(engine_version_at_admission="rc-0")
        disagrees = trajectory(engine_version_at_admission="rc-1")
        assert agrees.admission_agrees is True
        assert disagrees.admission_agrees is False

    def test_a_rollout_that_spans_an_update_is_still_bound_to_its_own_version(self):
        """Phase 4A's rollout: admitted at rc-0, finished after the update."""
        t = trajectory(
            engine_version_at_admission="rc-0",
            engine_version_at_completion="rc-1",
        )
        assert t.version == "rc-0", "the binding, not the engine's last word"
        assert t.spans_an_update is True
        assert t.admission_agrees is True


class TestValidation:
    def test_a_prompt_is_required(self):
        with pytest.raises(TrajectoryError, match="non-empty prompt"):
            trajectory(prompt="")

    def test_token_ids_must_be_ints(self):
        with pytest.raises(TrajectoryError, match="tuple of ints"):
            trajectory(token_ids=(1, "two"))
        with pytest.raises(TrajectoryError, match="tuple of ints"):
            trajectory(token_ids=(1, True))

    def test_negative_seconds_are_rejected(self):
        with pytest.raises(TrajectoryError, match="seconds"):
            trajectory(seconds=-1.0)


class TestLogprobs:
    """The field the replay validator reads (item 8).

    Its shape is an invariant, not a convention: ``ReplayPlan`` compares position
    *i* of the record against position *i* of the replay, so a short logprob tuple
    would silently shift every later comparison.
    """

    def test_logprobs_must_be_one_per_generated_token(self):
        trajectory(token_ids=(1, 2), logprobs=(-0.1, -0.2))
        with pytest.raises(TrajectoryError, match="one per generated token"):
            trajectory(token_ids=(1, 2), logprobs=(-0.1,))
        with pytest.raises(TrajectoryError, match="one per generated token"):
            trajectory(token_ids=(1, 2), logprobs=(-0.1, -0.2, -0.3))

    def test_logprobs_may_be_absent(self):
        assert trajectory(token_ids=(1, 2)).logprobs == ()

    def test_logprobs_must_be_real_numbers(self):
        with pytest.raises(TrajectoryError, match="real numbers"):
            trajectory(token_ids=(1,), logprobs=("low",))
        with pytest.raises(TrajectoryError, match="real numbers"):
            trajectory(token_ids=(1, 2), logprobs=(-0.1, None))

    def test_a_bool_is_not_a_logprob(self):
        """`bool` is an `int`; `True` would pass every arithmetic comparison."""
        with pytest.raises(TrajectoryError, match="real numbers"):
            trajectory(token_ids=(1,), logprobs=(True,))

    def test_logprobs_survive_a_round_trip(self):
        original = trajectory(token_ids=(1, 2), logprobs=(-0.25, -1.75))
        restored = Trajectory.from_json(json.loads(json.dumps(original.to_json())))
        assert restored == original
        assert restored.logprobs == (-0.25, -1.75)

    def test_the_phase5_artifact_still_loads_and_round_trips(self):
        """It predates the field, so `from_json` must default it to empty -- and the
        round trip must be unchanged, or replaying a committed record would shift."""
        path = REPO / "results" / "phase5-trajectories.jsonl"
        records = TrajectoryRecorder.read_jsonl(path)
        assert len(records) == 2
        assert all(record.logprobs == () for record in records)
        assert all(record.replay_ready for record in records)
        for record in records:
            assert Trajectory.from_json(record.to_json()) == record


class TestCodec:
    def test_a_declared_source_survives_a_round_trip(self):
        """`WeightIdentity.parse` would drop the provenance; this must not."""
        original = trajectory(
            binding=binding(identity=identity_for(STEP1)),
            engine_version_at_admission="rc-0",
            engine_version_at_completion="rc-1",
        )
        restored = Trajectory.from_json(json.loads(json.dumps(original.to_json())))

        assert restored == original
        assert restored.identity == original.identity
        assert restored.provenance == STEP1
        assert restored.replay_ready is True

    def test_a_manifest_only_round_trip_stays_manifest_only(self):
        restored = Trajectory.from_json(json.loads(json.dumps(trajectory().to_json())))
        assert restored.exactness == "manifest-only"
        assert restored.provenance is None

    def test_jsonl_round_trip_preserves_order_and_identity(self, tmp_path: Path):
        recorder = TrajectoryRecorder()
        v0, v1 = identity_for(STEP0), identity_for(STEP1)
        recorder.record(trajectory(binding=binding(version=0, identity=v0)))
        recorder.record(trajectory(binding=binding("R2", version=1, identity=v1)))

        path = recorder.write_jsonl(tmp_path / "trajectories.jsonl")
        restored = TrajectoryRecorder.read_jsonl(path)

        assert [t.request_id for t in restored] == ["R1", "R2"]
        assert [t.identity for t in restored] == [v0, v1]
        assert all(t.replay_ready for t in restored)


class TestRecorder:
    def test_manifest_only_records_are_surfaced_not_hidden(self):
        recorder = TrajectoryRecorder()
        recorder.record(trajectory())
        recorder.record(trajectory(binding=binding("R2", identity=identity_for(STEP0))))

        assert len(recorder) == 2
        assert len(recorder.manifest_only()) == 1
        assert "1 replay-ready, 1 manifest-only" in recorder.summary()

    def test_concurrent_recording_loses_nothing(self):
        """Appended from a rollout thread while the runner is mid-cycle."""
        recorder = TrajectoryRecorder()
        per_thread = 200
        failures: list[BaseException] = []

        def churn(worker: int) -> None:
            try:
                for i in range(per_thread):
                    recorder.record(trajectory(binding=binding(f"R{worker}-{i}")))
            except BaseException as exc:  # pragma: no cover - reported below
                failures.append(exc)

        threads = [threading.Thread(target=churn, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not failures
        assert len(recorder) == 4 * per_thread
        assert len({t.request_id for t in recorder}) == 4 * per_thread
