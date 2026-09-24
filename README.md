# RolloutCore

A versioned RL rollout runtime for vLLM.

RolloutCore sits around a vLLM inference engine and cycles it through

```
rollout(vN) → drain → update(vN+1) → invalidate stale cache → resume → rollout(vN+1)
```

while binding every trajectory to the committed generation *and* the declared
weight source that produced it.

> **Status: Phases 1–2 complete, plus the pre-GPU review patch** — the lifecycle
> state machine, the typed adapter port, a stdlib HTTP adapter for the lifecycle
> **control plane**, and a full `READY → … → READY` cycle against an in-memory
> fake engine. Real weight transfer (NCCL) is **not** implemented, and real-GPU
> **Phase 3A passed on real hardware** (2026-09-24): 16/16 control-plane checks
> against a real `vllm serve` at the audited commit — `docs/phase3a-results.md`.
> **Phase 3B is implemented and its transport proved**: real weights broadcast
> over NCCL from a trainer to a live vLLM on a 2× RTX 3090 node, with the
> target-aware version handshake (`src/rolloutcore/adapters/nccl.py`).
> **Phase 3C passed** (2026-09-24): the whole update driven through
> `LifecycleRunner.run_cycle()` on two real GPUs, `READY(rc-0) → … → READY(rc-1)`
> in one server process — `docs/phase3c-results.md`, `results/phase3c.json`.
> **Phase 4A passed** (2026-09-24): a 256-token generation spanned the update and
> came out of it on the version it was admitted to — 12/12 checks, so "one version
> per rollout" now holds under real concurrency, not just by construction —
> `docs/phase4a-results.md`, `results/phase4a.json`.
> **Phase 4B passed** (2026-09-24): the failure paths against a real engine —
> a recoverable drain failure, an identity mismatch that fails before any
> mutation, a SIGKILLed engine that taints, and recovery by restart. 21/21 checks
> over three engine lifetimes — `docs/phase4b-results.md`, `results/phase4b.json`.

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
| `docs/phase3b-runbook.md` | **Runbook** — what NCCL is, the two-GPU pod, proving upstream's path first, and the NCCL hang checklist |
| `docs/phase3c-results.md` | **Results** — the first real cycle end to end on two GPUs, and the identity limit it exposed |
| `docs/phase4a-results.md` | **Results** — one version per rollout under a live update: a generation spans the update and stays on its version |
| `docs/phase4b-results.md` | **Results** — the failure paths against a real engine, and the two places a correctly-detected failure has nowhere to go |
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
   invalidates nothing (`vllm/v1/worker/gpu_worker.py:1488-1505`), and the
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
backticked `file.py:LINE` claim from the checkout: **414 anchors resolve to real
files with in-range line numbers, 0 unresolved** (134 of them via a basename that
appears in several directories, so the line was checked against each candidate).

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
  demo.py            Runnable no-GPU demonstration
  adapters/          fake_engine.py, fake.py (reference adapter),
                     http.py (lifecycle control plane over a real vLLM)
tests/               234 tests, incl. the full 9x9 illegal-transition matrix
docs/                Design and plan-delta documents
scripts/live_control_plane_smoke.py  Phase 3A real-server harness (JSON artifact)
scripts/test.sh            Dependency-free check runner
scripts/verify_anchors.py  Re-derives the docs' vLLM file:LINE claims
tests/fake_dev_server.py   Stub vLLM dev server, so 3A is rehearsed off-GPU
diagnostics/   Vendored upstream NCCL example, adapted for the 2-GPU pod
```
