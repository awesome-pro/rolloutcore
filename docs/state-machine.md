# RolloutCore lifecycle state machine — design

**Design for Phases 1 and 2 (fake-engine cycle)**, and still the description of
the controller as it stands through Phase 6. Pure Python, no vLLM import in the
controller, no GPU.

Source: `src/rolloutcore/` · Tests: `tests/` (326 tests)
Run: `./scripts/test.sh` (no dependencies; also runs ruff + mypy when present)

---

## 1. Why the machine looks like this

The controller owns **state**; an adapter owns **effects**. No method in
`LifecycleController` performs I/O. Each transition takes an *evidence* object
describing what some external system observed, checks it against that state's
postcondition, then advances, raises, or taints.

That split is what makes the load-bearing claims testable:

| Claim | How it becomes testable |
|---|---|
| "No update begins until drain has **proven** zero active work" | `DrainEvidence` is required by `confirm_drained`; there is no path to `QUIESCED` without one |
| "No cross-version cache reuse" | `InvalidateEvidence` requires three independent booleans; a missing one taints |
| "Nothing is resumed before it is validated" | The resume is issued only on entry to `RESUMING`, which is reachable only via `confirm_validated` |
| "A trajectory identifies its declared weight source" | `UpdateTarget` carries a `WeightIdentity` (manifest **plus** declared provenance), not just a generation number |
| "A refused bootstrap cannot corrupt the engine" | The adapter raises `AlreadyManagedEngineError` after reading `/weight_info` and before any write; the test asserts the label is unchanged |
| "An ambiguous effect is never left as a safe-looking state" | The runner classifies every adapter call and taints on foreign exceptions from mutating ones |

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
                                          ▼                         │
                                      QUIESCED                      │
                                          │                         │
                       begin_update(target)                         │
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
                                          │  (engine STILL paused)  │
                     confirm_validated(evidence)                    │
                                          ▼                         │
                                     RESUMING                       │
                                          │                         │
                      confirm_resumed(evidence) ────────────────────┘
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
| `INVALIDATING` | Dropping every cache holding previous-generation content | `/reset_prefix_cache` + `/reset_encoder_cache` + `/reset_mm_cache` |
| `VALIDATING` | Proving the target **while still paused** | `GET /weight_info`, `GET /is_paused` — **no `/resume`** |
| `RESUMING` | Resume issued; confirming the engine is actually serving | `POST /resume`, `GET /is_paused` |
| `TAINTED` | **Terminal in V1.** Never resumed, never rolled back | — |

Eight forward edges, `taint` legal from all nine states (no-op at `TAINTED`).
The full 9×9 = 81-pair matrix is enumerated by
`lifecycle.enumerate_transition_matrix()` and asserted in
`tests/test_illegal_transitions.py`; 17 pairs are legal (8 edges + 9 taints) and
64 are illegal.

### Why `VALIDATING` runs before `RESUMING` (amendment 1)

Validation answers "are the right weights installed and are the caches clean?".
Nothing about that answer requires the engine to be serving — and asking it
while paused means a failure leaves the engine **unable to serve the unverified
weights at all**, rather than serving them while we discover the problem.

A second reason: a resume is an observable side effect. If it happened inside
`VALIDATING`, that state would sometimes leave the engine serving and sometimes
not, depending on where it failed. Making the resume its own transition keeps
each state's exit condition a pure predicate.

`PAUSED_STATES` (`lifecycle.py`) encodes this and is asserted in tests:
`VALIDATING` is a paused state; `RESUMING` and `READY` are not.

---

## 3. Invariants → enforcing code → test

| # | Invariant | Enforced by | Test |
|---|---|---|---|
| **I1** | Version atomicity: an update is completely committed or not visible | `_current_target` is written in exactly two places: `initialize` and `confirm_resumed` | `TestI1VersionAtomicity` |
| **I1b** | Versions are monotonic; no downgrade, no no-op update | `begin_update` requires `target.version > current.version` | `test_update_target_must_be_strictly_greater` |
| **I2** | A rollout binds to exactly one generation *and* identity, fixed at admission | frozen `RolloutBinding`; `admit_rollout` legal only in `READY` | `TestI2AdmissionBinding` |
| **I3** | No cross-version cache reuse | `InvalidateEvidence` requires prefix + encoder + MM; `cache_salt` is generation-scoped | `TestI3NoCrossVersionCacheReuse` |
| **I4** | A trajectory identifies the **exact declared weight source** that produced it | `UpdateTarget` carries `WeightIdentity` (manifest + declared provenance); the binding copies it | `TestI4ReplayIdentity` |
| **I5** | No active old-version work when an update begins | `confirm_drained` requires engine proof **and** zero local actives | `TestI5DrainCorrectness` |
| **I6** | A failed update never publishes its target | every failing `confirm_*` routes to `_taint_with`, which clears `pending_target` | `TestI6FailureSafety` |
| **I7** | RolloutCore is the single lifecycle/weight writer | no public setters; `__slots__`; the adapter refuses a managed engine **before writing** | `TestI7SingleWriter`, `TestI9BootstrapIsSingleOwner`, `test_managed_engine_is_refused_before_any_write` |
| **I8** | A version mismatch is a hard failure, never auto-reconciled | `confirm_validated` / `confirm_resumed` compare labels for equality and taint | `TestI8NoAutomaticReconciliation` |
| **I9** | V1 bootstrap is single-owner, and the refusal precedes the write | adapter-side `AlreadyManagedEngineError`; `BootstrapEvidence` re-checks the unmanaged pre-seed label | `TestI9BootstrapIsSingleOwner` |
| **I10** | `TAINTED` is terminal in V1 | no edge leaves `TAINTED` | `TestI10TaintedIsTerminal` |

---

## 4. Failure policy: raise, wait, or taint

Three outcomes, and getting the boundaries right is most of the design:

| Kind | Behaviour | Rationale |
|---|---|---|
| **Illegal event** for the current state | `IllegalTransitionError`, state unchanged | Programming error; the state machine says what is allowed |
| **Local precondition** — e.g. an update target that is not greater than the committed one | `InvariantViolation`, state unchanged | The engine is healthy; the call was wrong |
| **Observation incomplete** — e.g. the drain is still running | `EvidenceNotReady`, state unchanged, **retryable** | Normal operation. Tainting here would destroy a healthy engine because a caller polled early |
| **Engine evidence** contradicts a postcondition — e.g. `prefix_cache_reset=False`, resume unacknowledged | `EngineTaintedError`, **→ TAINTED** | We can no longer prove the engine's state |
| **Ambiguous mutating failure** — an adapter effect raised before reporting | `EngineTaintedError`, **→ TAINTED** | The engine may have half-applied the effect; see below |
| **Refused ownership** — the engine already reports `rc-*` | `AlreadyManagedEngineError`, state unchanged, controller stays `UNINITIALIZED` | Nothing was written and the engine is healthy — it is someone else's. Tainting would demand restarting a working engine |
| **No effect attempted** — no usable transfer driver | `WeightTransferNotConfiguredError`, state unchanged | No side effect occurred, so there is nothing ambiguous to taint |

Consequences:

- `TAINTED` is **absorbing** and **terminal**. V1 does not roll back or resume.
- `taint()` is **idempotent** and keeps the *first* reason, because later ones are
  cascades of the root cause.
- `abort_rollout` during a drain is **not** a taint: it is how a straggler is
  cleared.

### Ambiguous effects (review item 3)

The controller state alone is not enough once an effect has been *attempted*.
After `POST /start_weight_update` has been sent, "worker 0 initialized reload /
worker 1 failed" is indistinguishable from "nothing happened". Leaving the
controller in `UPDATING` would mean it believes something it cannot support, so
the runner classifies each adapter call deliberately:

| Adapter call | Class | On a foreign exception |
|---|---|---|
| `bootstrap` | refuse-before-write, then mutating | `AlreadyManagedEngineError` / any `RolloutCoreError` propagates; anything else taints |
| `begin_drain` | mutating (issues the pause) | taint |
| `await_drain` | observation | propagates, retryable |
| `start_weight_update` | mutating | taint |
| `complete_weight_update` | mutating, interleaved with a collective | taint |
| `invalidate_caches` | mutating | taint |
| `validate_pre_resume` | observation (engine still paused) | propagates, retryable |
| `resume` | mutating | taint |

`RolloutCoreError` passes through untouched from either path: those errors
already encode recoverability, and a controller-raised `EngineTaintedError` is
idempotent with the taint the runner would raise.

### Drain disagreement (amendment 2)

Precisely, in `confirm_drained`:

* `engine_drain_completed is False` → `EvidenceNotReady`. Non-zero local active
  rollouts are **expected** here; the drain simply has not finished.
* `engine_drain_completed is True` while local active rollouts remain →
  **`DrainDisagreementError` (taint)**. The engine asserts quiescence while
  RolloutCore counts live work; one bookkeeping system is wrong and we cannot
  tell which.
* `engine_drain_completed is True` with `engine_active_requests > 0` →
  taint: the engine's own evidence is self-contradictory.

The *adapter* side has its own state machine (review item 5), because "the
pause call has not returned" and "the pause call failed" need different
responses:

```
IDLE ──begin_drain──▶ IN_FLIGHT ──ok──▶ COMPLETED
                           │
                           └──error──▶ FAILED ──abort + reissue──▶ IN_FLIGHT
                                            └──budget spent──▶ DrainFailedError
```

`await_drain()` is **non-blocking** in every state and returns promptly; the
per-attempt socket timeout is `drain_timeout`, and the *caller's* deadline is the
runner's `drain_polls × drain_interval`, checked against `time.monotonic`. A
failed attempt is aborted and reissued at most `drain_reissues` times, then
`DrainFailedError` stops the poll loop. Without this, a permanently failed
`/pause` was polled 600 times against an attempt nobody was going to restart.
`DrainFailedError` is **not** a taint: `pause` sets `PAUSED_NEW` before it waits
(`vllm/v1/engine/core.py:1984-2026`), so the engine is known-paused and cannot
admit new work, and the controller stays in `DRAINING`, which cannot admit
rollouts either.

### Why no rollback

The original plan allowed *"safely restore `READY(N)`"* as an alternative to
failing. vLLM provides no way to verify that:

- `finish_weight_update` reports no per-tensor outcome and invalidates no cache
  (`vllm/v1/worker/gpu_worker.py:1488-1505`);
- the engine's label is written *before* caches are touched, so it is not a
  commit marker (demonstrated in `demo.py` scenario 4);
- RFC #48312's fail-closed storage contract (#48478) is still open.

Asserting "restored to N" would be unverifiable. `TAINTED`-terminal is the
honest outcome.

### Taint forensics (amendment 5)

`_taint_with` does **not** discard active bindings. It moves them to
`_orphaned`, each carrying the state it was tainted in, the reason, and a
sequence number. Whatever was in flight when the engine became untrustworthy is
exactly the evidence an operator needs to decide what to do with those
trajectories.

---

## 5. Generation number vs. weight identity (amendments 3 and 6)

These are deliberately separate types:

| | `WeightVersion` | `WeightIdentity` |
|---|---|---|
| Meaning | *when*, for lifecycle ordering | *what declared source*, for provenance |
| Serialised to the engine | yes, as `rc-<n>` | no — the engine has no field for it |
| Used for | monotonicity, `cache_salt`, admission ordering | invariant I4, replay, cache-coherence comparison |
| Equality implies | same update round | same declared weight source |

`UpdateTarget` bundles them, and `begin_update` takes an `UpdateTarget` rather
than a bare version — so a caller cannot open an update without saying which
weights it intends to install. `RolloutBinding` copies both.

**Why generation alone is insufficient for I4**: two controllers can publish
different weights under the same `rc-N` (a rebuilt controller, a restarted
engine, a re-run of the same step). "Generation 3" is a lifecycle position, not
a description of weights. `TestI4ReplayIdentity` asserts this directly.

### Two tiers, and why the first one was not enough (review item 4)

| Tier | Computed from | Distinguishes | `exactness` |
|---|---|---|---|
| Manifest-only | names, dtypes, shapes | architectures | `"manifest-only"` |
| Provenance-qualified | the manifest **plus** `WeightProvenance(checkpoint, run_id, step)` | training steps | `"declared-source"` |

A manifest digest is an *architecture* identity. Two checkpoints with the same
tensors' names, dtypes and shapes — step 100 and step 500 of one run, or a base
model and its fine-tune — produce the **same** manifest digest. That is the
normal case in RL, not an edge case, so a manifest digest alone cannot support
the claim "this trajectory came from step 500". `WeightProvenance` supplies the
missing part, is rejected if empty (an empty source would silently degrade to a
manifest-only identity), and is folded into the digest. `WeightIdentity.describe()`
and `exactness` make the tier visible wherever it is logged.

**What I4 claims, and what it does not.** A trajectory identifies the exact
*declared* weight source: generation, plus the declared provenance, plus the
digest over them. It does **not** claim the engine's memory byte-for-byte
contains those tensors, and no tier here is a hash of tensor contents. The
correspondence between a declaration and engine bytes is a differential/replay
question, deliberately out of scope for V1 (`TestWeightProvenance` covers the
provenance semantics; §12 item 3 keeps content hashing open).

`UpdateEvidence.observed_identity` is the one place the declaration is checked
against something independent: the driver computes the manifest identity of the
tensors it actually staged, and a mismatch against `target.identity` taints.

---

## 6. What `cache_salt` does and does not do

**`cache_salt` is prefix-cache isolation only.** It reaches
`generate_block_hash_extra_keys`
(`vllm/v1/core/kv_cache_utils.py:632-634`), i.e. the prefix KV block hash at
block 0. It is *not* a weight identity and *not* universal cache versioning:

| Cache | Key | Covered by `cache_salt`? |
|---|---|---|
| Prefix KV blocks | `(parent_hash, token_ids, extra_keys)` incl. `cache_salt` | **yes** |
| Encoder cache | `dict[mm_hash, Tensor]` (`vllm/v1/worker/gpu/mm/encoder_cache.py:8-13`) | no |
| MM processor cache | `model_id` + content + processor kwargs | no |
| LoRA adapter caches | adapter *id* (`vllm/lora/model_manager.py:115-120`) | no |

Hence `InvalidateEvidence` demands all three resets and taints if any is
missing. A design relying on `cache_salt` alone would be silently wrong for any
multimodal or LoRA workload — the failure class RFC #48312 category 7 tracks.

### The invalidation boundary (amendment 7)

**The drain does not clear caches, and neither does the update. `INVALIDATING`
is the only place a cycle drops them.** The pause is issued as
`POST /pause?mode=wait&clear_cache=false` (`src/rolloutcore/adapters/http.py`),
and the fake adapter mirrors it.

The reason is not symmetry. The pause happens **before** the mutation, so a clear
there discards KV that is still valid at that instant and, worse, hides a broken
`INVALIDATING`: with `clear_cache=true` every cycle would still come out clean,
and Phase 4C's guarantee could not tell the two clears apart. After the change
the order is

```
pause(clear_cache=false) → update → INVALIDATING (all three) → validate → resume
```

and every later step runs while the engine is still paused, so there is no window
in which a request can be served from a cache the mutation invalidated.

What pins this is Phase 4C's two lanes (`results/phase4c.json`): with
`clear_cache=false` and **no** reset, the engine handed back 32 prefix-cache hits
computed under the previous weights and the tokens differed; with the reset, the
same prompt scored 0 hits. The Phase 1 run of that harness predates this
amendment and its pause also cleared, which is precisely why the boundary is now
singular: the run can no longer be read as evidence about `INVALIDATING` alone,
and the next run of it can.

`tests/test_cycle.py::test_invalidating_is_the_only_place_caches_are_dropped`
pins the whole sequence: caches dirtied before the drain stay dirty through the
pause, the update and validation, and only the `/reset_*` triple empties them.

---

## 7. Control plane vs. data plane

- **HTTP is the lifecycle control plane.** Every state maps to vLLM dev
  endpoints (`VLLM_SERVER_DEV_MODE=1`).
- **Weights travel over the native trainer-side NCCL data plane**, out of band,
  through a **`WeightTransferDriver`** (`rolloutcore/weight_transfer.py`), which
  wraps `WeightTransferTrainerFactory.trainer_init(...)` and `send_weights()`.

The split is not cosmetic, and it is why the driver owns more than the tensor
push. Upstream's trainer engine posts `/init_weight_transfer_engine` itself
during `trainer_init` — concurrently with opening the trainer endpoint, because
the workers block inside that call until the rendezvous completes
(`nccl_engine.py:272-318`) — and `send_weights()` drives
`/start_weight_update` → `/update_weights` → `/finish_weight_update`
**concurrently** with the broadcast (`base.py:617-627`;
`examples/rl/rlhf_http_nccl.py:193-196`). `POST /update_weights` blocks while the
workers receive, so it cannot be issued before the collective is in flight. An
adapter that posted those three calls itself *and* asked a driver to push
tensors would deadlock, and one that posted them with empty metadata would be
lying.

**What is deliberately not implemented.** `HttpVLLMAdapter` implements the
control plane only. `LifecycleOnlyDriver` (the default) moves nothing and
raises `WeightTransferNotConfiguredError` on any attempt to install weights, so
a controller bootstrapped with it can never publish an update it did not
perform. `NCCLWeightTransferDriver` is a documented Phase 3B landing zone whose
methods raise; writing an untested NCCL wrapper would be a guess dressed up as
an implementation. Two payloads that are **not** real and are no longer
constructed: `{"init_info": {"backend": "nccl"}}` — `NCCLWeightTransferInitInfo`
(`nccl_common.py:71`) requires `rank_offset`, `world_size`, and exactly one of
`master_address`+`master_port` or `nccl_unique_id_b64` — and
`{"names": [], "dtype_names": [], "shapes": []}`.

No tensor crosses HTTP. Target backend: **`nccl` only**, following vLLM's
two-GPU `Qwen/Qwen3-1.7B-Base` → `Qwen/Qwen3-1.7B` example
(`examples/rl/rlhf_async_new_apis.py:61-62`) — with one deliberate deviation:
that example uses `pause_generation(mode="keep")`, which this design rejects in
favour of `mode="wait"`.

---

## 8. The typed adapter port

`CyclePlan.steps` is **human-readable narration only** — for logs, error
messages and test failure output. Adapters must not parse it. The typed
interface is `rolloutcore.port.LifecycleAdapter`:

| Method | vLLM operation group |
|---|---|
| `bootstrap()` | refuse if managed → driver `initialize()` (which posts `/init_weight_transfer_engine`) → `/update_weight_version` `rc-0` → `/weight_info` + `/get_world_size` |
| `begin_drain()` | `POST /pause?mode=wait&clear_cache=false` (non-blocking) |
| `await_drain()` | polls the attempt state; **never blocks**; abort + reissue on failure |
| `start_weight_update(target)` | validates a driver exists; the driver posts `POST /start_weight_update` from inside `transfer` |
| `complete_weight_update(target)` | `driver.transfer(target)` → `/start_weight_update` + `/update_weights` ×N + `/finish_weight_update`, interleaved with the collective |
| `invalidate_caches(target)` | the `/reset_*` triple |
| `validate_pre_resume(target)` | `GET /weight_info` + `GET /is_paused` — **must not resume** |
| `resume(target)` | `POST /resume` + `GET /is_paused` |

`WeightTransferDriver` is the second port, with two implementations and one
documented stub:

| Driver | `moves_tensors` | Status |
|---|---|---|
| `LifecycleOnlyDriver` | `False` | default; bootstrap-only, refuses to install weights |
| `NCCLWeightTransferDriver` | `True` | Phase 3B; raises, wrapping `WeightTransferTrainerFactory` |

`CyclePlan` also carries machine-readable `tags` for assertions, but tags are
never a dispatch mechanism either.

Two `LifecycleAdapter` implementations ship:

- `adapters/fake.py` — `FakeVLLMAdapter` over an in-memory `FakeVLLMEngine` that
  reproduces the vLLM behaviours the design depends on (see §10). It also
  refuses a managed engine before writing, and reports
  `weight_transfer_driver="fake-in-process"`.
- `adapters/http.py` — `HttpVLLMAdapter`, stdlib `urllib`, injectable
  `Transport` and `WeightTransferDriver`. Handles the traps: `/pause` has no
  server-side timeout, so it runs on a background thread with an explicit
  attempt state machine; `/reset_prefix_cache` returns HTTP 200 with
  `{"success": false}` while blocks are held, so a 200 is not success; and
  bootstrap refuses an `rc-*` engine before writing anything.

---

## 9. Running the cycle

```python
from rolloutcore import LifecycleController, LifecycleRunner
from rolloutcore.adapters import FakeVLLMAdapter, FakeVLLMEngine, manifest_identity

engine = FakeVLLMEngine()
engine.seed_fresh_with(manifest_identity("A"))
ctrl = LifecycleController()
runner = LifecycleRunner(ctrl, FakeVLLMAdapter(engine), sleep=lambda _s: None)

runner.bootstrap()                                  # -> READY at rc-0
binding = ctrl.admit_rollout("R1")                  # bound to rc-0 + identity A
ctrl.finish_rollout("R1")

result = runner.install_next(manifest_identity("B"))  # -> READY at rc-1
```

`python -m rolloutcore.demo` runs this end to end and prints the state journal,
the engine calls, and three failure scenarios.

---

## 10. Fake engine fidelity

`FakeVLLMEngine` is not a stub that rubber-stamps: it reproduces the specific
behaviours the design depends on, each traced to its vLLM source.

| Behaviour | Why it matters | vLLM anchor |
|---|---|---|
| `weight_version` starts as the literal `"default"` and is never auto-incremented | bootstrap must seed `rc-0` | `core.py:137`, `:1043` |
| `pause(mode="wait")` sets `PAUSED_NEW` **first**, then reports "not complete" while requests are active | a still-running drain is `EvidenceNotReady`, and the engine is already blocking new work | `core.py:2017-2018` |
| `pause(clear_cache=False)` leaves all three caches populated | which is why the drain must not be trusted to clear: INVALIDATING is the boundary | `core.py:877-882` → `:861-875` |
| `finish_weight_update` writes the version but invalidates **no cache** | the engine label is not a commit marker | `async_llm.py:1284-1288`; `gpu_worker.py:1488-1505` |
| `finish_weight_update` failure leaves the label unwritten | the label is written only *after* the RPC returns | `async_llm.py:1284-1288` |
| `reset_prefix_cache` can return `false` while blocks are held | a 200 is not success | `block_pool.py:831-838` |

---

## 11. Test suite

| File | Tests | Covers |
|---|---|---|
| `test_legal_transitions.py` | 18 | all 8 forward edges individually, full cycle, multi-cycle, journal, resume ordering |
| `test_illegal_transitions.py` | 14 | all 81 (state, event) pairs; strong non-mutation; admission matrix across 9 states |
| `test_invariants.py` | 52 | I1–I10, taint forensics, the plan's §13 mixed-version experiment |
| `test_evidence.py` | 35 | every `failure_reason` branch of all six evidence types, driver consistency, immutability |
| `test_versions.py` | 45 | label round-trip, ordering, `WeightIdentity` canonicity, both identity tiers, `cache_salt` constraint |
| `test_cycle.py` | 25 | **Phase 2**: full cycle over the fake engine, pause ordering, drain polling, pre-write bootstrap refusal, effect-vs-observation failure classification, a dead engine during a drain |
| `test_http_adapter.py` | 39 | call sequences, query params, real vs. fabricated payloads, managed-engine refusal, drain reissue/budget, stub-driver round trip, port conformance |
| `test_live_smoke.py` | 8 | the Phase 3A harness against `tests/fake_dev_server.py`: full pass, drain really waited, post-resume drift caught, warm-server refusal, in-process `mode=wait` rejection, and zero weight-transfer calls |
| `test_nccl_driver.py` | 26 | the real driver's identity/declare seam, the transfer round trip against a fake engine, and the pure `device_mismatch` rule (no GPU, no torch) |
| `test_trajectory.py` | 20 | `Trajectory`/`TrajectoryRecorder`: provenance, `replay_ready`, admission agreement, JSONL round trip, logprob shape, and the committed Phase 5 artifact |
| `test_replay.py` | 35 | the item 8 validator: protocol violations, token and logprob lanes, tolerance, the greedy/trace-forced modes, and the CLI end to end |
| `test_controller_threading.py` | 4 | the reentrant controller lock, and the duplicate-sequence failure that removing it produces |
| `test_script_hygiene.py` | 5 | the rules the GPU harnesses must not break: status-only `/health` probes, `os.killpg` only with `start_new_session=True` |

326 tests. `ruff check`, `ruff format --check` and `mypy --strict` all pass on
the package; CI runs them on Python 3.11/3.12/3.13. The frozen-evidence
immutability test asserts on a real dataclass *field* rather than a
monkey-patched method, because on 3.11/3.12 the non-field path in a
frozen+slots `__setattr__` raises `TypeError` instead of `FrozenInstanceError` —
the guarantee held, the assertion was version-dependent.

---

## 12. Open questions

1. **`invoke`-style ergonomics for `EvidenceNotReady`.** `LifecycleRunner` polls
   and retries on `EvidenceNotReady`; other callers must handle it too. Should
   the controller expose a `wait_for_drain(adapter)` helper, or is the runner the
   only sanctioned driver?
2. **Identity for the bootstrap weights.** `BootstrapEvidence.weight_identity` is
   supplied by the adapter, because the engine cannot report one. For a real
   `vllm serve`, where should it come from — vLLM's trainer-side
   `WeightSource.metadata()` plus a `WeightProvenance`, the model loader's
   `ParamMeta`, or the checkpoint's revision metadata? Phase 3A begins to answer
   this empirically by recording what the adapter can actually observe.
3. **Content hashing.** Even a provenance-qualified identity does not prove what
   the engine's memory holds: it identifies the *declared* source. Is per-tensor
   content hashing worth its cost in V2, or is a differential/replay check the
   better instrument?
4. **`RESUMING` retry policy.** A failed `/resume` currently taints. The engine
   may be resumable by a plain retry; should `RESUMING` allow a bounded retry
   before tainting, given the engine is paused (and therefore safe) throughout?
5. **A control-plane-only cycle.** The review's Phase 3A wants
   `drain → cache reset → validate → resume` with no weight change at all. The
   current table cannot express that (`QUIESCED` has exactly one outgoing edge,
   to `UPDATING`). Phase 3A therefore runs at the **adapter** level and the
   controller is deliberately tainted at the end, because the adapter resumed an
   engine the controller cannot vouch for — see `docs/phase3a-runbook.md` and
   `scripts/live_control_plane_smoke.py`. The open question stands: should V1 add
   a revalidation edge (same committed target, caches dropped, no update), or
   should every cache-revalidation be a generation bump with a real no-op
   transfer?
