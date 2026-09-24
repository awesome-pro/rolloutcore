# RolloutCore

A versioned RL rollout runtime for vLLM.

RolloutCore sits around a vLLM inference engine and cycles it through

```
rollout(vN) → drain → update(vN+1) → invalidate stale cache → resume → rollout(vN+1)
```

while binding every trajectory to the exact weight version that produced it.

> **Status: Phase 1 complete** — the pure-Python lifecycle state machine and its
> tests (103 tests, no dependencies). The real-vLLM adapter is Phase 2, gated on
> review of `docs/state-machine.md`.

---

## Quick start

```bash
./scripts/test.sh
```

Phase 1 is pure stdlib on Python ≥ 3.11 — nothing to install. With pytest
available it uses that instead.

```python
from rolloutcore import LifecycleController, WeightVersion, BootstrapEvidence

ctrl = LifecycleController()
ctrl.initialize(BootstrapEvidence(
    observed_engine_label="rc-0", weight_transfer_initialised=True,
    seeded=True, pre_seed_label="default",
))

binding = ctrl.admit_rollout("req-1")     # bound to rc-0, frozen
ctrl.finish_rollout("req-1")

ctrl.begin_drain()                              # READY -> DRAINING
ctrl.confirm_drained(drain_evidence)            # requires proven zero active work
ctrl.begin_update(WeightVersion(1))
ctrl.confirm_updated(update_evidence)
ctrl.confirm_invalidated(invalidate_evidence)   # all three caches, or taint
ctrl.confirm_validated(validate_evidence)       # commits rc-1
```

---

## Documents

| File | What it is |
|---|---|
| `PROJECT.md` | Project brief: why it exists, invariants, roadmap |
| `docs/state-machine.md` | **Phase 1 design, for review** — states, transitions, failure policy, open questions |
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
support in vLLM `main`** as of the commit audited:

1. **One weight version per rollout.** PR #49040 added a `weight_version` query
   API but *deliberately removed* binding a version to `Request`/`RequestOutput`,
   because one request may span multiple versions. The contract is still open
   (RFC #48306 §2.2).
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
at this commit, not never existed**. 396 anchors were machine-verified to
resolve to real files with in-range line numbers.

---

## Layout

```
src/rolloutcore/
  versions.py    WeightVersion: the counter vLLM does not keep, + cache_salt
  evidence.py    Postconditions reported by the (future) adapter
  errors.py      InvariantViolation (retryable) vs EngineTaintedError (terminal)
  lifecycle.py   The state machine
tests/           103 tests — legal transitions, the full 8x8 illegal matrix, I1-I9
docs/            Design and plan-delta documents
scripts/test.sh  Dependency-free test runner
```
