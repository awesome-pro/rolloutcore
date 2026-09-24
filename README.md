# RolloutCore

A versioned RL rollout runtime for vLLM.

RolloutCore sits around a vLLM inference engine and cycles it through

```
rollout(vN) → drain → update(vN+1) → invalidate stale cache → resume → rollout(vN+1)
```

while binding every trajectory to the committed generation *and* the declared
weight source that produced it.

> **Status: Phases 1–6 complete.** The lifecycle state machine, the typed adapter
> port, a stdlib HTTP adapter for the lifecycle **control plane**, and a full
> `READY → … → READY` cycle against an in-memory fake engine. Everything that
> names hardware was measured on a real pod at the audited commit:
> **Phase 3A passed** — 16/16 control-plane checks against a real `vllm serve`
> (`docs/phase3a-results.md`); **Phase 3C passed** — real weights broadcast over
> NCCL and the whole update driven through `LifecycleRunner.run_cycle()`
> (`docs/phase3c-results.md`); **Phase 4A passed** — 12/12, a generation spanned
> the update and stayed on the version it was admitted to; **Phase 4B passed** —
> 21/21 failure paths over three engine lifetimes; **Phase 4C passed** — 8/8, no
> cross-version KV reuse, with a negative control that makes the reuse happen;
> **Phase 4D passed** — 8/8, a SIGKILL mid-update detected in 0.177 s and
> tainted; **Phase 5 passed** — 16/16, trajectories that name the training step;
> **Phase 6 passed** — 11/11 at `opt-125m` and 6/6 at `Qwen3-8B`, a hot update
> costing 130.63× and 8.71× less than a restart.
>
> Item 8, the token/logprob **replay validator**, is implemented and runs on a
> laptop (`scripts/validate_replay.py`, `docs/replay-validator.md`). Its logprob
> lane has not yet run against a real engine, because the Phase 5 artifact
> captured tokens only — say so before quoting a logprob number from it.

---

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

# Item 8 on a laptop, against the committed Phase 5 records (token lane only --
# they carry no logprobs), with the synthetic replay file kept in the repo so the
# command runs as written:
python3 scripts/validate_replay.py \
    --trajectories results/phase5-trajectories.jsonl \
    --replays results/phase5-replay-synthetic.jsonl --no-json
```

Pure stdlib on Python ≥ 3.11 — nothing to install. `scripts/test.sh` picks up
`.venv` if present, and degrades gracefully when pytest/ruff/mypy are absent.

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

---

## Documents

| File | What it is |
|---|---|
| `PROJECT.md` | Project brief: why it exists, invariants, roadmap, upstream-contribution policy |
| `docs/state-machine.md` | **Design** — states, transitions, failure policy, adapter port, fake-engine fidelity |
| `docs/phase3a-results.md` | **Results** — the real-GPU Phase 3A run: environment, measurements, what it proves and does not |
| `docs/phase3a-runbook.md` | **Runbook** — one-GPU pod setup, the exact commands, checkpoints, pass/fail criteria, troubleshooting |
| `docs/phase3b-notes.md` | Phase 3B design — the target-aware weight-sync client and the `finish_weight_update` version gap |
| `docs/phase3b-runbook.md` | **Runbook** — what NCCL is, the two-GPU pod, proving upstream's path first, the NCCL hang checklist, and (§10–11) the four failure modes found later plus the thread-local device trap |
| `docs/phase3c-results.md` | **Results** — the first real cycle end to end on two GPUs, and the identity limit it exposed |
| `docs/phase4a-results.md` | **Results** — one version per rollout under a live update: a generation spans the update and stays on its version |
| `docs/phase4b-results.md` | **Results** — the failure paths against a real engine, and the two places a correctly-detected failure has nowhere to go |
| `docs/phase4c-results.md` | **Results** — no cross-version KV reuse, measured, and then deliberately made to happen |
| `docs/phase4d-results.md` | **Results** — the engine dies mid-update: detected in 0.177 s, nothing published |
| `docs/phase5-results.md` | **Results** — trajectories that name the training step, and the engine's last word that is not the binding |
| `docs/phase6-results.md` | **Results** — the hot-update benchmark against a restart at two sizes, and the kill inside a long collective that never returns |
| `docs/replay-validator.md` | **Design** — the token/logprob replay validator: what it proves, what it cannot, and how to produce replays on a GPU box |
| `docs/upstream-contribution.md` | **Proposal** — the upstream contribution extracted from the findings, each defect assigned a verdict; nothing has been filed |
| `docs/plan-delta.md` | Original plan vs. source-map findings vs. implementation amendments |
| `source-map-vllm-main.md` | Source-level map of vLLM `main`: 9 subsystems, every claim anchored to `file:LINE`, plus an RFC cross-check |
| `mvp-plan.md` | The minimal real-vLLM cycle: exact HTTP call sequence, integration surface, guardrails |
| `old-plan.md` | The original project plan (kept for reference) |
| `research/` | RFC bodies and comment threads fetched from GitHub, verbatim |

Suggested read order: `PROJECT.md` → `docs/plan-delta.md` → `docs/state-machine.md`
→ `source-map-vllm-main.md` for the evidence behind any claim.

---

## Why this exists

Two guarantees matter most for RL correctness, and **neither has engine-side
support in vLLM `main`** as of the commit audited (details in `PROJECT.md`):

1. **One weight version per rollout.** PR #49040 added a `weight_version` query
   API but *deliberately removed* binding a version to `Request`/`RequestOutput`,
   because one request may span multiple versions. The contract is still open
   (RFC #48306 §2.2). A generation number is a lifecycle position, not a weight
   identity, so RolloutCore carries both: `WeightVersion` for ordering and a
   `WeightIdentity` over the manifest **and** the declared trainer provenance.
2. **No cross-version cache reuse.** Prefix-cache keys carry no weight
   generation (`vllm/v1/core/kv_cache_utils.py:610-646`), `finish_weight_update`
   invalidates no cache (`vllm/v1/worker/gpu_worker.py:1488-1505`; its only
   cleanup is `reset_lora_state()`), and the
   encoder-cache fix (PR #48762) was closed unmerged.

Both are therefore the caller's to provide. That is what RolloutCore is.

---

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
| Confirmed by | `git ls-remote origin refs/heads/main` **and** `gh api repos/vllm-project/vllm/commits/main` |

The sibling checkout `vendor/vllm` is a stale fork (`d90f0eade5`) and was not
used. The worktree is a shallow clone (depth 1), so **"NOT PRESENT" means absent
at this commit, not never existed**. `scripts/verify_anchors.py` re-derives every
backticked `file.py:LINE` claim from the checkout **and** from this repository:
**561 anchors resolve to real files with in-range line numbers, 0 unresolved**
(36 of them via a basename that appears in several directories, so the line was
checked against each candidate).

Cross-tree basenames get a rule of their own, because five names exist in both
trees — `runner.py` (vLLM's is `benchmarks/attention_benchmarks/runner.py`, 573
lines) is the trap. A repo-local basename is resolved against this repo first,
and any anchor that resolves in both trees, or only upstream while a same-named
local file exists, is **named in the output** rather than silently trusted. Both
trees are clean of that class as of the last run.

---

## Layout

```
src/rolloutcore/
  versions.py        WeightVersion (rc-N generation) + WeightIdentity (declared source)
  evidence.py        Postconditions reported by the adapter
  errors.py          IllegalTransition vs InvariantViolation vs EvidenceNotReady vs Taint
  lifecycle.py       The 9-state machine — no I/O
  port.py            LifecycleAdapter: the typed adapter contract
  weight_transfer.py WeightTransferDriver: the trainer-side NCCL seam
  runner.py          Drives a cycle; taints on ambiguous effects
  trajectory.py      Trajectory records + TrajectoryRecorder (JSONL)
  replay.py          The token/logprob replay validator — pure, no engine
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
diagnostics/   Standalone 2-GPU harnesses: upstream's NCCL example and Phases 3C–6,
               each writing its own results/ artifact
```
