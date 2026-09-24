# RolloutCore

A versioned RL rollout runtime for vLLM.

RolloutCore sits around a vLLM inference engine and cycles it through

```
rollout(vN) → drain → update(vN+1) → invalidate stale cache → resume → rollout(vN+1)
```

binding every trajectory to the generation and the declared weight source that
produced it.

## Status

Phases 1 to 6 are complete. The lifecycle state machine, the typed adapter port,
an HTTP adapter for the control plane, a real NCCL weight-transfer driver,
trajectory records and a replay validator all exist and are tested. Everything
that names hardware was measured on a real pod at the commit the source map
audits.

| Phase | What it establishes | Result | Evidence |
|---|---|---|---|
| 3A | Control plane against a real `vllm serve` | 16/16 | `docs/phase3a-results.md` |
| 3C | First full cycle, dummy weights replaced by real ones over NCCL | 5/5 | `docs/phase3c-results.md` |
| 4A | An update requested while a generation is in flight is held off until it finishes | 12/12 | `docs/phase4a-results.md` |
| 4B | Failure paths: a timed-out drain, an identity mismatch, a killed engine, recovery | 21/21 | `docs/phase4b-results.md` |
| 4C | No cross-version KV reuse, plus a control that makes the reuse happen | 8/8 | `docs/phase4c-results.md` |
| 4D | Engine killed mid-update: detected in 0.177 s, nothing published | 8/8 | `docs/phase4d-results.md` |
| 5 | Trajectories that name the training step, not just the architecture | 16/16 | `docs/phase5-results.md` |
| 6 | Hot update against a restart: 130.63x at 125M, 8.71x at 8B | 11/11, 6/6 | `docs/phase6-results.md` |

Item 8, the token/logprob replay validator, runs on a laptop
(`scripts/validate_replay.py`, `docs/replay-validator.md`). The committed Phase 5
records carry tokens but no logprobs, so only the token lane has been run against
a real engine. Say so before quoting a logprob number.

## Quick start

```bash
./scripts/test.sh          # tests + ruff + mypy
./scripts/test.sh --fast   # tests only
python3 -m rolloutcore.demo # full cycle, fake engine, no GPU

# Re-derive every file:LINE claim in the docs against a vLLM checkout:
VLLM_CHECKOUT=/path/to/vllm ./scripts/test.sh
python3 scripts/verify_anchors.py --vllm /path/to/vllm

# Phase 3A harness, rehearsed with no GPU against the in-repo stub server:
PYTHONPATH=src:tests python3 tests/fake_dev_server.py --port 8123 &
python3 scripts/live_control_plane_smoke.py --base-url http://127.0.0.1:8123 \
    --model facebook/opt-125m --json-out /tmp/phase3a-stub.json

# Phase 3A for real (on a GPU pod, vLLM >= the audited commit):
python3 scripts/live_control_plane_smoke.py --launch --model facebook/opt-125m

# Item 8 on a laptop, against the committed Phase 5 records. They carry no
# logprobs, so this exercises the token lane; the synthetic replay file is kept
# in the repo so the command runs as written:
python3 scripts/validate_replay.py \
    --trajectories results/phase5-trajectories.jsonl \
    --replays results/phase5-replay-synthetic.jsonl --no-json
```

`src/rolloutcore` is pure stdlib on Python 3.11 and later, with nothing to
install. `scripts/test.sh` uses `.venv` if one is present, and works without
pytest, ruff or mypy (the test suite runs under `unittest` on its own).

```python3
from rolloutcore import LifecycleController, LifecycleRunner
from rolloutcore.adapters import FakeVLLMAdapter, FakeVLLMEngine, manifest_identity

engine = FakeVLLMEngine()
engine.seed_fresh_with(manifest_identity("A"))     # fresh, unmanaged label
ctrl = LifecycleController()
runner = LifecycleRunner(ctrl, FakeVLLMAdapter(engine), sleep=lambda _s: None)

runner.bootstrap()                                 # -> READY at rc-0
binding = ctrl.admit_rollout("R1")                 # bound to rc-0 + identity A
ctrl.finish_rollout("R1")
runner.install_next(manifest_identity("B"))        # -> READY at rc-1
```

## Documents

| File | What it is |
|---|---|
| `PROJECT.md` | Project brief: why it exists, invariants, roadmap, upstream-contribution policy |
| `docs/state-machine.md` | Design: states, transitions, failure policy, adapter port, fake-engine fidelity |
| `docs/phase3a-results.md` | Results from the real-GPU Phase 3A run: environment, measurements, what it does and does not prove |
| `docs/phase3a-runbook.md` | Runbook for a one-GPU pod: setup, exact commands, checkpoints, pass/fail criteria, troubleshooting |
| `docs/phase3b-notes.md` | Phase 3B design: the target-aware weight-sync client and the `finish_weight_update` version gap |
| `docs/phase3b-runbook.md` | Runbook for the two-GPU pod, the NCCL hang checklist, and (sections 10 and 11) the four failure modes found later plus the thread-local device trap |
| `docs/phase3c-results.md` | Results: the first full cycle on two GPUs, and the identity limit it exposed |
| `docs/phase4a-results.md` | Results: the update is held off until the in-flight generation finishes, so no rollout spans a mutation |
| `docs/phase4b-results.md` | Results: the failure paths, and the two places where a correctly detected failure has nowhere to go |
| `docs/phase4c-results.md` | Results: no cross-version KV reuse, measured, and then made to happen on purpose |
| `docs/phase4d-results.md` | Results: the engine dies mid-update and is detected in 0.177 s, with nothing published |
| `docs/phase5-results.md` | Results: trajectories that name the training step, and why the engine's last word is not the binding |
| `docs/phase6-results.md` | Results: hot update against a restart at two model sizes, and the kill inside a long collective that never returns |
| `docs/replay-validator.md` | Design: the token/logprob replay validator, what it proves, what it cannot, and how to produce replays on a GPU box |
| `docs/upstream-contribution.md` | Proposal: the upstream contribution extracted from the findings, with a verdict for each defect and a plan for filing it |
| `docs/plan-delta.md` | Original plan compared with the source-map findings and the implementation amendments |
| `source-map-vllm-main.md` | Source-level map of vLLM `main`: 9 subsystems, every claim anchored to `file:LINE`, plus an RFC cross-check |
| `mvp-plan.md` | The minimal real-vLLM cycle: HTTP call sequence, integration surface, guardrails |
| `old-plan.md` | The original project plan, kept for reference |
| `research/` | RFC bodies and comment threads fetched from GitHub, verbatim |

Suggested read order: `PROJECT.md`, then `docs/plan-delta.md` and
`docs/state-machine.md`, with `source-map-vllm-main.md` for the evidence behind
any claim.

## Why this exists

Two things RL correctness depends on have no engine-side support in vLLM `main`
at the commit audited here. The details are in `PROJECT.md`.

1. **One weight version per rollout.** PR #49040 added the `weight_version` query
   API and removed binding a version to `Request`/`RequestOutput`, because one
   request may span two versions. The contract is still open (RFC #48306 §2.2).
   A generation number is a lifecycle position, not a weight identity, so
   RolloutCore carries both: `WeightVersion` for ordering, and a `WeightIdentity`
   over the manifest and the declared trainer provenance.
2. **No cross-version cache reuse.** Prefix-cache keys carry no weight generation
   (`vllm/v1/core/kv_cache_utils.py:610-646`), `finish_weight_update` invalidates
   no cache (`vllm/v1/worker/gpu_worker.py:1488-1505`, and its only cleanup is
   `reset_lora_state()`), and the encoder-cache fix (PR #48762) was closed
   unmerged.

Both guarantees have to come from the code that drives the engine, so RolloutCore
provides them there.

## Provenance of the vLLM analysis

Every `file:LINE` anchor in `source-map-vllm-main.md` was verified against the
real GitHub tip, not a local fork:

| | |
|---|---|
| Repo | `https://github.com/vllm-project/vllm.git` |
| Commit | `00b7847c8036b667742b4efb21aab1de51fd4721` |
| Subject | *"[Perf] Use Conv3dLayer for MiniMax M3 patch embedding (#58512)"* |
| Committer date | `2026-09-24T07:41:36Z` |
| Worktree | `/Users/abhinandan/Desktop/vllm-learning/vendor/vllm-main` (clean, read-only) |
| Confirmed by | `git ls-remote origin refs/heads/main` and `gh api repos/vllm-project/vllm/commits/main` |

The sibling checkout `vendor/vllm` is a stale fork (`d90f0eade5`) and was not
used. The worktree is a shallow clone (depth 1), so "NOT PRESENT" means absent at
this commit, not that it never existed. `scripts/verify_anchors.py` re-derives
every backticked `file.py:LINE` claim from the checkout and from this repository:
557 anchors resolve to real files with in-range line numbers, and none are
unresolved. 36 of them go through a basename that appears in several directories,
so the line was checked against each candidate.

Five basenames exist in both trees, and `runner.py` is the trap: vLLM has
`benchmarks/attention_benchmarks/runner.py` with 573 lines, so a repo-local
`runner.py` claim used to be checked against the wrong file. A repo-local basename
is now resolved against this repository first, and any anchor that resolves in
both trees is named in the output instead of being trusted silently. The last run
reports none.

Two places still contain em dashes, both on purpose. `research/` holds RFC bodies
and GitHub threads exactly as fetched, and `diagnostics/upstream_nccl_2gpu.py` is
vendored from upstream, so it keeps upstream's own comments. The rest of the
repository has none.

## Layout

```
src/rolloutcore/
  versions.py        WeightVersion (rc-N generation) + WeightIdentity (declared source)
  evidence.py        Postconditions reported by the adapter
  errors.py          IllegalTransition vs InvariantViolation vs EvidenceNotReady vs Taint
  lifecycle.py       The 9-state machine, no I/O
  port.py            LifecycleAdapter: the typed adapter contract
  weight_transfer.py WeightTransferDriver: the trainer-side NCCL seam
  runner.py          Drives a cycle; taints on ambiguous effects
  trajectory.py      Trajectory records + TrajectoryRecorder (JSONL)
  replay.py          Token/logprob replay validator, pure, no engine
  demo.py            Runnable no-GPU demonstration
  adapters/          fake_engine.py, fake.py (reference adapter),
                     http.py (lifecycle control plane over a real vLLM),
                     nccl.py (the real weight-transfer driver)
tests/               326 tests, incl. the full 9x9 illegal-transition matrix
docs/                Design, results, runbooks, replay-validator, upstream-contribution
results/             Committed JSON artifacts for every GPU phase
scripts/live_control_plane_smoke.py  Phase 3A real-server harness (JSON artifact)
scripts/validate_replay.py           Item 8 CLI: trajectories vs replays, exit code
scripts/test.sh            Dependency-free check runner
scripts/verify_anchors.py  Re-derives the docs' vLLM file:LINE claims
tests/fake_dev_server.py   Stub vLLM dev server, so 3A is rehearsed off-GPU
diagnostics/   Standalone 2-GPU harnesses: upstream's NCCL example and Phases 3C to 6,
               each writing its own results/ artifact
```
