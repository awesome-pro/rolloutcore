#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Item 8: validate replays against the trajectories they claim to reproduce.

A trajectory says "these tokens came from this declared weight source". This tool
tests that statement: for every record it builds a
:class:`~rolloutcore.replay.ReplayPlan`, pairs it with the
:class:`~rolloutcore.replay.ReplayObservation` a replayed engine returned, and
reports the verdict. It needs no engine, no GPU and no network -- it reads two
JSONL files and does arithmetic -- so it can be run on a laptop against records
produced on a GPU box.

Two lanes, and it matters which one a run is in:

* **token lane** -- the recorded token ids are compared position by position.
  Available whenever a record has ``token_ids``, which the Phase 5 harness always
  captured.
* **logprob lane** -- ``|Δ logprob|`` over the same positions, reported as mean,
  max and p99 (nearest-rank), plus the worst position. This is the lane that can
  see a distribution drift the argmax hides, and it needs both sides to have
  requested logprobs.

**The committed Phase 5 artifact has no logprobs.** ``results/phase5-trajectories.jsonl``
was produced by a harness that asked only for ``return_token_ids``, so the first
replay run over it exercises the token lane and reports
``logprob_lane == "both-missing"``. The logprob lane has therefore never been
exercised against a real engine -- say so when quoting a result from it.

Producing replays on the GPU side
--------------------------------

A replay reproduces a record in one of two modes, and the mode decides how a
token mismatch should be read -- so each observation line declares which one it
came from via ``tokens_were_forced``.

**Trace-forced** (``tokens_were_forced: true``). vLLM can be told exactly which
ids to emit: ``SamplingParams.trace_decode_token_ids`` (``sampling_params.py:374``)
"forces the engine to emit this predetermined sequence of token IDs during
decoding instead of sampling randomly. Real logprobs are still computed"
(``sampling_params.py:375-377``), and it requires the engine to be started with
``--enable-trace-replay`` (``config/model.py:276``), otherwise the request is
rejected (``v1/engine/input_processor.py:169-176``). The sampler overwrites the
sampled token while computing logprobs from the unmodified logit distribution
(``v1/worker/gpu/sample/trace_replay.py:14``). The engine is therefore *expected
to echo*, and failing to is a fault in the replay -- the validator reports it as
an ``echo`` protocol violation and refuses to read it as a weight finding.

That field is **not** reachable through the OpenAI-compatible server: ``vllm/entrypoints/``
has no reference to it, so ``POST /v1/completions`` cannot force a sequence today.
Two honest options, and they are not equivalent:

1. **In-process, trace-forced** -- the strong mode. Start an ``LLM`` with
   ``enable_trace_replay=True`` and pass a ``SamplingParams`` carrying
   ``max_tokens=len(record.token_ids)``, ``logprobs=1`` and
   ``trace_decode_token_ids=list(record.token_ids)``.
   Record ``tokens_were_forced: true``.
2. **Over HTTP, greedy** -- the weaker mode. Re-request with ``temperature=0``,
   ``logprobs=1`` and the recorded text appended to the prompt. The recorded tokens
   are an *expectation* the weights have to reproduce, not a script, so a mismatch
   **is** the finding. Leave ``tokens_were_forced`` false (its default).

Either way, write one JSON object per line with the observation fields below. A
completion response maps as: ``request_id`` by your own bookkeeping, ``version``
from ``GET /weight_info`` *at replay time*, ``token_ids`` from the response, and
``logprobs`` from ``choices[0].logprobs.token_logprobs``
(``entrypoints/openai/completion/protocol.py:625``). That list can contain
``null``; a position without a logprob means the lane does not cover it, so either
drop the record to the token lane or fix the request.

Observation line format (unknown keys are ignored, so a hand-written file may
annotate itself)::

    {"request_id": "R2", "version": "rc-1", "token_ids": [5, 812],
     "logprobs": [-0.12, -1.03], "identity_digest": "sha256:...", "prompt": "...",
     "tokens_were_forced": true}

Usage
-----

Against the committed Phase 5 record, with a synthetic replay file kept in the
repository so the command is runnable as written (token lane only, see above)::

    python3 scripts/validate_replay.py \\
        --trajectories results/phase5-trajectories.jsonl \\
        --replays results/phase5-replay-synthetic.jsonl

Against replays you produced on a GPU box, writing a JSON artifact::

    python3 scripts/validate_replay.py --replays /tmp/replays.jsonl \\
        --json-out results/replay-verdicts.json --tolerance 1e-3

Exit codes: ``0`` every record replayed and agreed, ``1`` any disagreement, any
trajectory without a replay, any replay without a trajectory, or any record that
cannot be replayed at all (a manifest-only identity), ``2`` a usage or I/O error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:  # allow running straight from a checkout
    sys.path.insert(0, str(_REPO / "src"))

from rolloutcore import (  # noqa: E402 - path bootstrap must precede the import
    Trajectory,
    TrajectoryRecorder,
    validate_replay,
)
from rolloutcore.replay import (  # noqa: E402 - path bootstrap must precede the import
    DEFAULT_LOGPROB_TOLERANCE,
    ReplayError,
    ReplayObservation,
    ReplayPlan,
    ReplayVerdict,
)
from rolloutcore.trajectory import TrajectoryError  # noqa: E402 - see bootstrap above

DEFAULT_TRAJECTORIES = Path("results/phase5-trajectories.jsonl")
DEFAULT_JSON_OUT = Path("results/replay-verdicts.json")


@dataclass(frozen=True, slots=True)
class Config:
    trajectories: Path
    replays: Path
    tolerance: float
    json_out: Path | None
    quiet: bool


def parse_args(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--trajectories",
        type=Path,
        default=DEFAULT_TRAJECTORIES,
        help=f"a TrajectoryRecorder.write_jsonl file (default: {DEFAULT_TRAJECTORIES})",
    )
    ap.add_argument(
        "--replays",
        type=Path,
        required=True,
        help="JSONL of ReplayObservation objects, one per line, matched by request_id",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_LOGPROB_TOLERANCE,
        help="|delta logprob| at or below this counts as agreement (default: %(default)s)",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help=f"write every verdict and the aggregate here (default: {DEFAULT_JSON_OUT})",
    )
    ap.add_argument(
        "--no-json",
        dest="json_out",
        action="store_const",
        const=None,
        help="skip the JSON artifact",
    )
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args(argv)
    return Config(
        trajectories=args.trajectories,
        replays=args.replays,
        tolerance=args.tolerance,
        json_out=args.json_out,
        quiet=args.quiet,
    )


def _numbers(raw: Any, *, where: str) -> tuple[float, ...]:
    """Coerce a JSON list to floats, refusing anything that is not a number.

    A ``null`` from ``token_logprobs`` is refused rather than coerced: silently
    dropping it would shorten the tuple and quietly change which positions the
    logprob lane covers.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where}: expected a list, got {type(raw).__name__}")
    out = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: logprobs must be numbers, got {value!r}")
        out.append(float(value))
    return tuple(out)


def load_observations(path: Path) -> tuple[ReplayObservation, ...]:
    """Read one ``ReplayObservation`` per non-blank line, with line numbers on error."""
    observations = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        where = f"{path}:{number}"
        try:
            observations.append(
                ReplayObservation(
                    request_id=str(data["request_id"]),
                    version=str(data["version"]),
                    token_ids=tuple(int(t) for t in data.get("token_ids", ())),
                    logprobs=_numbers(data.get("logprobs"), where=where),
                    identity_digest=data.get("identity_digest"),
                    prompt=data.get("prompt"),
                    tokens_were_forced=bool(data.get("tokens_were_forced", False)),
                )
            )
        except KeyError as exc:
            raise ValueError(f"{where}: missing required field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
    return tuple(observations)


def _plan_for(trajectory: Trajectory) -> ReplayPlan | None:
    """The plan, or ``None`` after reporting why the record cannot be replayed."""
    try:
        return ReplayPlan.from_trajectory(trajectory)
    except (ReplayError, TrajectoryError) as exc:
        print(f"[unreplayable] {trajectory.request_id}: {exc}")
        return None


def run(cfg: Config) -> int:
    trajectories = TrajectoryRecorder.read_jsonl(cfg.trajectories)
    observations = load_observations(cfg.replays)
    by_request = {observation.request_id: observation for observation in observations}

    verdicts: list[ReplayVerdict] = []
    failures: list[str] = []
    matched: set[str] = set()

    for trajectory in trajectories:
        plan = _plan_for(trajectory)
        if plan is None:
            failures.append(f"{trajectory.request_id}: not replayable")
            continue
        observation = by_request.get(plan.request_id)
        if observation is None:
            print(f"[no-replay] {plan.request_id}: no observation to compare against")
            failures.append(f"{plan.request_id}: no replay")
            continue
        matched.add(plan.request_id)
        verdict = validate_replay(plan, observation, logprob_tolerance=cfg.tolerance)
        verdicts.append(verdict)
        if not cfg.quiet:
            print(verdict.describe())
            for note in verdict.notes:
                print(f"    note: {note}")
            for violation in verdict.protocol:
                print(f"    protocol: {violation}")
        if not verdict.agrees:
            failures.append(f"{verdict.request_id}: disagrees")

    for request_id in sorted(set(by_request) - matched):
        print(f"[no-trajectory] {request_id}: no trajectory to compare against")
        failures.append(f"{request_id}: no trajectory")

    agreed = sum(1 for verdict in verdicts if verdict.agrees)
    forced = sum(1 for observation in observations if observation.tokens_were_forced)
    lanes: dict[str, int] = {}
    for verdict in verdicts:
        lanes[verdict.logprob_lane] = lanes.get(verdict.logprob_lane, 0) + 1
    max_deltas = [
        verdict.max_abs_delta for verdict in verdicts if verdict.max_abs_delta is not None
    ]

    summary = {
        "trajectories": len(trajectories),
        "replays": len(observations),
        "compared": len(verdicts),
        "trace_forced": forced,
        "greedy": len(observations) - forced,
        "agreed": agreed,
        "disagreed": len(verdicts) - agreed,
        "failures": failures,
        "token_mismatches": sum(verdict.token_mismatches for verdict in verdicts),
        "logprob_lanes": lanes,
        "max_abs_delta": max(max_deltas) if max_deltas else None,
        "tolerance": cfg.tolerance,
    }

    print(
        f"\n{agreed}/{len(verdicts)} replay(ies) agreed over {len(trajectories)} "
        f"trajectory record(s); {forced} trace-forced / {len(observations) - forced} greedy, "
        f"lanes {lanes or '{}'}, "
        f"{summary['token_mismatches']} token mismatch(es) in total"
    )
    if failures:
        print(f"FAIL: {len(failures)} problem(s): {'; '.join(failures)}")
    else:
        print("PASS: every record replayed and agreed")

    if cfg.json_out is not None:
        artifact = {
            "trajectories": str(cfg.trajectories),
            "replays": str(cfg.replays),
            "verdicts": [asdict(verdict) for verdict in verdicts],
            "summary": summary,
        }
        cfg.json_out.parent.mkdir(parents=True, exist_ok=True)
        cfg.json_out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {cfg.json_out}")

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except (OSError, ValueError, TrajectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
