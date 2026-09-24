# RolloutCore lifecycle state machine — design for review

**Phase 1 deliverable.** Pure Python, no vLLM import, no I/O, no GPU.
**Status:** awaiting review before the Phase 2 real-vLLM adapter.

Source: `src/rolloutcore/` · Tests: `tests/` (103 tests, 98.9% statement coverage of the package)
Run: `./scripts/test.sh` (no dependencies — stdlib `unittest`)

---

## 1. Why the machine looks like this

The controller owns **state**; an adapter owns **effects**. No method in
`LifecycleController` performs I/O. Instead, each transition takes an *evidence*
object describing what some external system observed, checks it against that
state's postcondition, and then either advances, raises, or taints.

That split is what makes the two load-bearing claims testable:

| Claim | How it becomes testable |
|---|---|
| "No update begins until drain has **proven** zero active work" | `DrainEvidence` is a required argument to `confirm_drained`; there is no path to `QUIESCED` without one |
| "Cross-version cache reuse is forbidden" | `InvalidateEvidence` requires three independent booleans; a missing one taints |

---

## 2. States

```
                  initialize(evidence)
   UNINITIALIZED ──────────────────────► READY ◄───────────────────┐
                                          │                         │
                          begin_drain()   │                         │
                                          ▼                         │
                                      DRAINING                      │
                                          │                         │
                     confirm_drained(evidence)                      │
                                          │                         │
                                          ▼                         │
                                      QUIESCED                      │
                                          │                         │
                        begin_update(target)                        │
                                          │                         │
                                          ▼                         │
                                      UPDATING                      │
                                          │                         │
                     confirm_updated(evidence)                      │
                                          ▼                         │
                                   INVALIDATING                     │
                                          │                         │
                   confirm_invalidated(evidence)                    │
                                          ▼                         │
                                    VALIDATING                      │
                                          │                         │
                    confirm_validated(evidence) ────────────────────┘
                                                            (commit)

   any state except TAINTED ──taint(reason)──► TAINTED   (absorbing)
```

| State | Meaning | Phase 2 mapping |
|---|---|---|
| `UNINITIALIZED` | Constructed, not bound to a proven engine | — |
| `READY` | Serving. **The only admitting state.** | engine resumed, `weight_version == rc-N` |
| `DRAINING` | Drain requested; active rollouts must reach zero | `POST /pause?mode=wait` in flight |
| `QUIESCED` | Drain proven: zero active work, engine paused | pause returned; `has_work() == False` |
| `UPDATING` | Weights in flight over the native data plane | `/start_weight_update` … `/finish_weight_update` |
| `INVALIDATING` | Dropping every cache holding previous-version content | `/reset_prefix_cache` + `/reset_encoder_cache` + `/reset_mm_cache` |
| `VALIDATING` | Proving the engine serves the target before publishing | `/resume`, `GET /weight_info`, `GET /is_paused` |
| `TAINTED` | **Terminal in V1.** Never resumed, never rolled back | — |

Seven forward edges, `taint` legal from all eight states (no-op at `TAINTED`).
The full 8×8 = 64-pair matrix is enumerated in
`lifecycle.enumerate_transition_matrix()` and asserted in
`tests/test_illegal_transitions.py`; 15 pairs are legal (7 edges + 8 taints) and
49 are illegal.

---

## 3. Invariants → enforcing code → test

| # | Invariant | Enforced by | Test |
|---|---|---|---|
| **I1** | Version atomicity: an update is completely committed or not visible | `_current_version` is written in exactly two places: `initialize` and `confirm_validated` | `TestI1VersionAtomicity` |
| **I1b** | Versions are monotonic; no downgrade, no no-op update | `begin_update` requires `target > current` | `test_update_target_must_be_strictly_greater`, `test_lower_target_is_rejected` |
| **I2** | A rollout binds to exactly one committed version, fixed at admission | `RolloutBinding` is frozen; `admit_rollout` only legal in `READY` | `TestI2AdmissionBinding` |
| **I3** | No cross-version cache reuse | `InvalidateEvidence` requires prefix + encoder + MM; `cache_salt` is version-scoped | `TestI3NoCrossVersionCacheReuse` |
| **I4** | A trajectory identifies the exact weights that produced it | binding captures `(version, cache_salt, admitted_seq)` | `TestI4ReplayIdentity` |
| **I5** | No active old-version work when update begins | `confirm_drained` requires `active_rollouts == 0` | `TestI5DrainCorrectness` |
| **I6** | A failed update never publishes its target | every `confirm_*` failure routes to `_taint`, which clears `pending_version` | `TestI6FailureSafety` |
| **I7** | RolloutCore is the single lifecycle/weight writer | no public setters; `__slots__`; seeding only over an *unmanaged* label | `TestI7SingleWriter` |
| **I8** | A version mismatch is a hard failure, never auto-reconciled | `confirm_validated` compares labels for equality and taints | `TestI8NoAutomaticReconciliation` |
| **I9** | `TAINTED` is terminal in V1 | no edge leaves `TAINTED`; `test_no_edge_leaves_tainted` | `TestI9TaintedIsTerminal` |

### I2 in detail: why DRAINING rejects new rollouts

The old plan (§13) framed this as an explicit policy choice: **queue** or
**reject/retry**. This design chooses **reject**.

Queueing is unsafe in a way that is easy to miss. With `mode="wait"`, vLLM sets
`PAUSED_NEW` and blocks the WAITING queue while still stepping RUNNING requests
(`vllm/v1/core/sched/scheduler.py:868`, `:2668-2669`). A request added during the
drain therefore sits in the engine's queue and is scheduled *after* `resume` —
under version **N+1** — while RolloutCore had bound it to N. That is exactly the
mixed-version response I2 forbids, and the engine cannot detect it, because
PR #49040 deliberately removed per-request version binding.

Rejecting is the only choice that keeps the binding true. The caller retries
after `READY`, or holds the request itself.

---

## 4. Failure policy: raise vs. taint

This is the one rule a reviewer should push back on if they disagree.

| Kind of failure | Behaviour | Rationale |
|---|---|---|
| **Local precondition** — e.g. confirming a drain while rollouts are still active | `InvariantViolation`, **state unchanged** | The engine is healthy; the drain just has not finished. Tainting would destroy a working engine because a caller was early. |
| **Engine-side evidence** contradicts a postcondition — e.g. `prefix_cache_reset=False`, resume not acknowledged | `EngineTaintedError`, **→ TAINTED** | We can no longer prove the engine's state. Continuing risks serving stale or mixed weights. |

Consequences worth stating explicitly:

- `TAINTED` is **absorbing** and **terminal**. V1 does not attempt rollback or
  resume. The remedy is to restart the engine and construct a new controller.
- `taint()` is **idempotent** and keeps the *first* reason, because later ones
  are cascades of the root cause (`test_taint_from_tainted_is_idempotent_and_keeps_first_reason`).
- `abort_rollout` during a drain is **not** a taint: it is the normal way a
  straggler is cleared.

### Why no rollback

The old plan (§14) allowed two outcomes for a failed update: *safely restore
`SERVING(N)`* **or** *enter `FAILED`*. The amendment removes the first option,
and the source map explains why that is the right call rather than a scope cut:

- `finish_weight_update` invalidates nothing and reports no per-tensor outcome
  (`vllm/v1/worker/gpu_worker.py:1488-1505`).
- There is no engine-side commit certificate; `GET /weight_info` returns the
  last value *set*, not a proof of what is loaded.
- RFC #48312's fail-closed storage contract (#48478) is still open.

So "restore `SERVING(N)`" would assert something we cannot verify. Claiming it
would be worse than admitting the engine is untrusted.

---

## 5. The bootstrap exception

A fresh `vllm serve` reports `weight_version == "default"`
(`vllm/v1/engine/core.py:137`). RolloutCore must write `rc-0` before it can
assert anything, which is otherwise forbidden. `BootstrapEvidence` scopes the
exception precisely:

- the adapter may set `seeded=True` **only** if the pre-seed label was *not*
  already RolloutCore-managed (`rc-*`);
- the post-seed read must confirm `rc-0`.

Seeding over an existing `rc-N` fails with *"another RolloutCore writer may own
this engine"* — which is how I7 is defended against a second orchestrator, not
just against a second thread.

After `initialize` returns, **any** mismatch is a hard failure. There is no code
path that writes a version label in response to an observed disagreement.

---

## 6. What `cache_salt` does and does not do

The amendment is explicit and the source map agrees: **`cache_salt` is
prefix-cache isolation only.**

It reaches `generate_block_hash_extra_keys`
(`vllm/v1/core/kv_cache_utils.py:632-634`), i.e. the prefix KV block hash at
block 0. It does **not** isolate:

| Cache | Key | Covered by `cache_salt`? |
|---|---|---|
| Prefix KV blocks | `(parent_hash, token_ids, extra_keys)` incl. `cache_salt` | **yes** |
| Encoder cache | `dict[mm_hash, Tensor]` (`vllm/v1/worker/gpu/mm/encoder_cache.py:8-13`) | no |
| MM processor cache | `model_id` + content + processor kwargs | no |
| LoRA adapter caches | adapter *id* (`vllm/lora/model_manager.py:115-120`) | no |

Hence `InvalidateEvidence` demands all three resets and taints if any is
missing. A design that relied on `cache_salt` alone would be silently wrong for
any multimodal or LoRA workload — which is precisely the failure class RFC
#48312 category 7 tracks.

---

## 7. Control plane vs. data plane

Stated here because it is easy to conflate:

- **HTTP is the lifecycle control plane.** Every state maps to vLLM dev
  endpoints (`VLLM_SERVER_DEV_MODE=1`): `/pause`, `/resume`,
  `/start_weight_update`, `/update_weights`, `/finish_weight_update`,
  `/reset_prefix_cache`, `/reset_encoder_cache`, `/reset_mm_cache`,
  `/weight_info`, `/is_paused`, `/init_weight_transfer_engine`, `/get_world_size`.
- **Weights travel over the native trainer-side NCCL data plane**, out of band,
  driven by `WeightTransferTrainerFactory.trainer_init(...)` and the
  `send_weights()` loop (`examples/rl/rlhf_async_new_apis.py:152-154`,
  `:138-150`). The HTTP `POST /update_weights` carries only *metadata*
  (`names`, `dtype_names`, `shapes`) and blocks while the data plane streams.

RolloutCore never moves a tensor over HTTP.

**Target for the MVP: the `nccl` backend only**, following vLLM's current
two-GPU `Qwen/Qwen3-1.7B-Base` → `Qwen/Qwen3-1.7B` example
(`rlhf_async_new_apis.py:61-62`). One deliberate deviation from that example:
it uses `pause_generation(mode="keep")`, which this design rejects in favour of
`mode="wait"` (see §3).

---

## 8. Public API

```python
from rolloutcore import LifecycleController, LifecycleState, WeightVersion

ctrl = LifecycleController()
ctrl.initialize(BootstrapEvidence(observed_engine_label="rc-0",
                                  weight_transfer_initialised=True,
                                  seeded=True, pre_seed_label="default"))

b = ctrl.admit_rollout("req-1")      # -> RolloutBinding(version=rc-0, cache_salt="rc-0")
ctrl.finish_rollout("req-1")

ctrl.begin_drain()                    # READY -> DRAINING
ctrl.confirm_drained(DrainEvidence(active_rollouts=0, engine_pause_confirmed=True))

ctrl.begin_update(WeightVersion(1))   # -> CyclePlan(steps=[...]) for the adapter
ctrl.confirm_updated(UpdateEvidence(...))
ctrl.confirm_invalidated(InvalidateEvidence(True, True, True))
ctrl.confirm_validated(ValidateEvidence(...))   # commits rc-1
```

`CyclePlan.steps` is the adapter's to-do list; it exists so the HTTP calls are
named in one place and asserted in tests rather than scattered through the
adapter.

---

## 9. Test suite

| File | Covers |
|---|---|
| `test_legal_transitions.py` | all 7 forward edges individually + full cycle + multi-cycle + journal |
| `test_illegal_transitions.py` | all 64 (state, event) pairs; strong non-mutation assertions; admission matrix across all 8 states |
| `test_invariants.py` | I1–I9 plus the old plan's §13 mixed-version experiment |
| `test_evidence.py` | every `failure_reason` branch of all five evidence types |
| `test_versions.py` | label round-trip, ordering, foreign-label rejection, `cache_salt` schema constraint |

103 tests, 98.9% statement coverage. The three unhit lines are deliberate
defence-in-depth branches (an abstract base raising, and two redundant re-checks
that `_require_legal` makes unreachable).

---

## 10. Open questions for review

1. **`InvariantViolation` vs taint for a drain with active rollouts.** I chose
   *raise and stay in DRAINING*. The alternative reading is that a non-zero
   count after the engine confirmed a pause is a bookkeeping disagreement and
   should taint. Current behaviour: local count non-zero → raise; local zero but
   *evidence* non-zero → taint. Is that split right?
2. **Reject vs. queue during DRAINING.** Chosen: reject. If the intended
   integration wants the engine to queue and simply binds late, the admission
   rule would need to move — but then a trajectory's version is decided by the
   engine's scheduler, not by RolloutCore.
3. **`confirm_validated` ordering.** `POST /resume` is modelled as the first
   action of VALIDATING, so `ValidateEvidence` covers "resumed *and* correct".
   The alternative is a separate `RESUMING` state. Eight states matches the
   amendment, so this was folded in rather than added.
4. **Whether `initialize` should tolerate an already-`rc-0` engine** without
   seeding. Currently yes (`seeded=False, pre_seed_label=None`), which makes
   controller restarts against a live engine possible. Worth confirming that
   restart-adoption is desired in V1, given I7.
