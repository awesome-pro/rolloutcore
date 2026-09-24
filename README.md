# RolloutCore

A versioned RL rollout runtime for vLLM.

RolloutCore sits around a vLLM inference engine and cycles it through

```
rollout(vN) → drain → update(vN+1) → invalidate stale cache → validate → resume → rollout(vN+1)
```

binding every trajectory to the generation and the declared weight source that
produced it.

## Why this exists

Two guarantees matter most for RL correctness, and neither has engine-side
support in vLLM `main` at commit `00b7847c` (2026-09-24).

1. **One weight version per rollout.** PR #49040 added a `weight_version` query
   API and removed binding a version to `Request`/`RequestOutput`, because one
   request may span multiple versions. The contract is still open (RFC #48306
   §2.2).
2. **No cross-version cache reuse.** Prefix-cache keys carry no weight
   generation: `extra_keys` is only the LoRA name, multimodal hashes,
   `cache_salt` and prompt-embed hashes (`vllm/v1/core/kv_cache_utils.py:610-646`).
   `finish_weight_update` invalidates no cache
   (`vllm/v1/worker/gpu_worker.py:1488-1505`), and the encoder-cache fix
   (PR #48762) was closed unmerged.

Both guarantees have to come from the caller, which is what RolloutCore is.
Nothing in the design requires a change to vLLM.

## Status

Phases 1 to 6 are complete. The lifecycle state machine, the adapter port, the
HTTP control plane, the NCCL weight-transfer driver, trajectory records and the
replay validator live in `src/rolloutcore`. The GPU evidence is in `results/`,
written up in `docs/`.

| Phase | What it establishes | Result | Evidence |
|---|---|---|---|
| 3A | Control plane against a real `vllm serve` on one A6000: fresh label, bootstrap to `rc-0`, a refused second bootstrap, deterministic generation, `pause(mode="wait")` that waited 2.13 s for a 2.02 s request, all three cache resets, validation, resume, byte-identical output afterwards | 16/16 | `docs/phase3a-results.md` |
| 3C | One `run_cycle()` takes a server from dummy weights to real `opt-125m` weights over NCCL, `rc-0` to `rc-1`, in one process, and reproduces Phase 3A's text token for token | 5/5 | `docs/phase3c-results.md`, `results/phase3c.json` |
| 4A | An update requested while a 256-token generation is in flight is held off for 1.375 s until the generation returns, so no rollout sees two versions. The journal shows `finish_rollout` before `begin_update` | 12/12 | `docs/phase4a-results.md`, `results/phase4a.json` |
| 4B | Failure paths against real vLLM: a timed-out drain that recovers in place, an identity mismatch refused before any write, a SIGKILLed engine that taints, and recovery by restart. It also found two places where a correctly detected failure has nowhere to go | 21/21 | `docs/phase4b-results.md`, `results/phase4b.json` |
| 4C | Invariant I3, which vLLM gives no mechanism for: after an update, the first request whose KV came from the old weights scores zero new prefix-cache hits, from vLLM's own metrics. A second phase drives the update out of band and makes the reuse happen (32 hits, different text) | 8/8 | `docs/phase4c-results.md`, `results/phase4c.json` |
| 4D | The engine is SIGKILLed the moment the controller enters `UPDATING`. The failure is detected in 0.177 s and taints, and `rc-2` is never published | 8/8 | `docs/phase4d-results.md`, `results/phase4d.json` |
| 5 | Trajectories that name the training step. A rollout admitted at `rc-0` was in flight when the update was requested, and the record keeps `rc-0` while the engine's label after the cycle is `rc-1`. The provenance is declared, so nothing verifies weight bytes. One field in the committed artifact (`engine_version_at_completion`) was read at the wrong moment; the harness is fixed, the artifact is left as measured, and `docs/phase5-results.md` documents the defect | 16/16 | `docs/phase5-results.md`, `results/phase5.json` |
| 6 | Hot update against a restart: 0.2299 s against 30.03 s at `opt-125m` (130.63x) and 4.2544 s against 37.04 s at `Qwen3-8B` (8.71x). The same phase killed the engine inside the 8B broadcast and the trainer never returned within a 120 s budget | 11/11, 6/6 | `docs/phase6-results.md`, `results/phase6*.json` |

Two notes on layers the table does not cover.

Tensors move over the trainer-side data plane through a `WeightTransferDriver`.
`NCCLWeightTransferDriver` (`src/rolloutcore/adapters/nccl.py`) wraps vLLM's
`WeightTransferTrainerFactory` and computes its identity from the trainer's own
manifest, so a mismatch fails before anything is mutated. The other shipped
driver, `LifecycleOnlyDriver`, moves nothing and refuses to install weights, so a
controller bootstrapped against it can be exercised through drain, cache reset,
validation and resume, but cannot publish an update it did not perform.

Trajectory records are in `src/rolloutcore/trajectory.py`. A record carries the
binding (version and identity) rather than the engine's last reported label, and
its provenance is declared: the caller states checkpoint, run id and step, and
nothing verifies the weight bytes. What the identity buys is that two versions
cannot collide, which the manifest digest alone cannot do. Phase 3C measured the
same digest for dummy and real weights, and Phase 4C measured it again for a
corrupted checkpoint.

## How it works

```
   UNINITIALIZED ──initialize──► READY ◄─────────────────────┐
                                  │                          │
                    begin_drain() │                          │
                                  ▼                          │
                              DRAINING                       │
                                  │ confirm_drained          │
                                  ▼                          │
                              QUIESCED                       │
                                  │ begin_update(target)     │
                                  ▼                          │
                              UPDATING                       │
                                  │ confirm_updated          │
                                  ▼                          │
                           INVALIDATING                      │
                                  │ confirm_invalidated      │
                                  ▼                          │
                            VALIDATING                       │
                                  │   (engine STILL paused)  │
                            confirm_validated                │
                                  ▼                          │
                             RESUMING                        │
                                  │ confirm_resumed          │
                                  └──────────────────────────┘
                                                       (commit)

   any state except TAINTED ──taint──► TAINTED   (absorbing, terminal in V1)
```

The controller owns state. An adapter owns effects. No method performs I/O: each
transition takes an evidence object, checks it against that state's
postcondition, then advances, raises or taints.

Three orderings carry the design:

- **Validation happens while the engine is still paused.** A failure leaves the
  engine unable to serve weights that could not be verified, rather than serving
  them while the problem is discovered.
- **The commit point is the resume**, not the finalize. The engine's own
  `weight_version` advances before caches are touched, so it is not a commit
  marker. `demo.py` scenario 4 shows this.
- **`INVALIDATING` is the only place a cycle drops caches.** The drain pauses
  with `clear_cache=false`, because a clear before the mutation is not a
  correctness boundary and would hide a broken invalidation. The reset runs after
  the mutation while the engine is still paused, so nothing can be served from a
  cache the update invalidated. See "The invalidation boundary" in
  `docs/state-machine.md`; `results/phase4c.json` measures both directions.

`TAINTED` is terminal. V1 does not roll back or resume from it, because vLLM
gives no way to prove the weights were restored.

### Invariants

| # | Invariant |
|---|---|
| I1 | An update is completely committed or not visible; versions are monotonic |
| I2 | A rollout binds to exactly one generation and identity, fixed at admission |
| I3 | KV created under generation N is never consumed under N+1 |
| I4 | A trajectory identifies the declared weight source that produced it: generation, declared provenance (checkpoint/run/step), and a digest over both |
| I5 | No active old-version work when an update begins |
| I6 | A failed update never publishes its target version |
| I7 | RolloutCore is the single lifecycle and weight writer |
| I8 | A version mismatch is a hard failure and is never auto-reconciled |
| I9 | V1 bootstrap is single-owner: only a fresh, unmanaged engine may be seeded, and the refusal happens before any write |
| I10 | `TAINTED` is terminal in V1, and preserves in-flight bindings for forensics |

## Quick start

```bash
./scripts/test.sh          # tests + ruff + mypy (uses .venv if present)
./scripts/test.sh --fast   # tests only
python3 -m rolloutcore.demo # full cycle against the fake engine, no GPU

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

## Testing

326 tests run under both `pytest` and plain `unittest`, with no third-party
packages needed for the latter, plus `ruff`, `mypy --strict` over `src/` and
`scripts/`, and the fake-engine demo. CI runs the matrix on Python 3.11, 3.12 and
3.13.

The GPU work lives in `diagnostics/`: one standalone harness per phase, each
booting a real `vllm serve`, printing its checks as it goes and writing a JSON
artifact to `results/`. `diagnostics/README.md` documents them and the rules they
share.

Tests use `facebook/opt-125m` because vLLM's own NCCL example uses it
(`examples/rl/rlhf_http_nccl.py:43`), it runs in seconds, and its greedy
continuation is stable enough to check by eye. None of the claims depend on
parameter count. Later work needs different models: `Qwen/Qwen3-1.7B-Base` to
`Qwen/Qwen3-1.7B` for a realistic base/instruct story, a multimodal model for the
encoder lane, and tensor parallelism above 1 for the transfer term.

## Repository layout

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
diagnostics/         Standalone 2-GPU harnesses, one per phase
scripts/             test.sh, verify_anchors.py, live_control_plane_smoke.py,
                     validate_replay.py
```

## Roadmap

| # | Deliverable | Status |
|---|---|---|
| 0 | Map the current vLLM RL lifecycle | done |
| 1 | Lifecycle state machine + tests | done |
| 2 | Typed port + HTTP adapter + fake-engine cycle | done |
| 3A | Real `vllm serve`: bootstrap, generate, drain(wait), cache reset, validate, resume | done, 16/16, see `docs/phase3a-results.md` |
| 3B | Real hot weight update: 2 GPUs, trainer-side NCCL, target-aware sync client | done, `src/rolloutcore/adapters/nccl.py` |
| 3C | First real cycle, dummy to real weights through `run_cycle` | done, 5/5, see `docs/phase3c-results.md` |
| 4A | Mixed-version rollout prevention under an update | done, 12/12, see `docs/phase4a-results.md` |
| 4B | Failure injection against a real engine | done, 21/21, see `docs/phase4b-results.md` |
| 4C | No cross-version KV reuse, plus the negative control | done, 8/8, see `docs/phase4c-results.md` |
| 4D | An engine that dies inside the update | done, 8/8, see `docs/phase4d-results.md` |
| 5 | Trajectory records from the trainer's declared provenance | done, 16/16, see `docs/phase5-results.md` |
| 6 | Benchmark against a restart baseline | done, 11/11 at 125M and 6/6 at 8B, see `docs/phase6-results.md` |
| 7 | Encoder and MM cache lane | deferred: needs a multimodal checkpoint; `/reset_encoder_cache` and `/reset_mm_cache` are no-ops on text-only models |
| 8 | Token/logprob replay validator | done: `src/rolloutcore/replay.py`, `scripts/validate_replay.py`, `docs/replay-validator.md`. The logprob lane has not yet run against a real engine |

Deferred: sleep/wake, IPC, multi-node, async overlap, dashboards, and new vLLM
APIs beyond the dev endpoints the design uses.

## Documents

| File | What it is |
|---|---|
| `docs/state-machine.md` | Design: states, transitions, failure policy, adapter port, fake-engine fidelity |
| `docs/source-map-vllm-main.md` | The evidence base: vLLM `main` mapped subsystem by subsystem, every claim anchored to `file:LINE`, plus an RFC cross-check |
| `docs/phase3a-runbook.md` | Runbook for a one-GPU pod: setup, exact commands, checkpoints, pass/fail criteria, troubleshooting |
| `docs/phase3b-runbook.md` | Runbook for the two-GPU pod, the NCCL hang checklist, and (sections 10 and 11) the four failure modes found later plus the thread-local device trap |
| `docs/phase3a-results.md` | Results from the Phase 3A run: environment, measurements, what it does and does not prove |
| `docs/phase3b-notes.md` | Phase 3B design: the target-aware weight-sync client and the `finish_weight_update` version gap |
| `docs/phase3c-results.md` | Results: the first full cycle on two GPUs, and the identity limit it exposed |
| `docs/phase4a-results.md` | Results: the update is held off until the in-flight generation finishes, so no rollout spans a mutation |
| `docs/phase4b-results.md` | Results: the failure paths, and the two places where a correctly detected failure has nowhere to go |
| `docs/phase4c-results.md` | Results: no cross-version KV reuse, measured, and then made to happen on purpose |
| `docs/phase4d-results.md` | Results: the engine dies mid-update and is detected in 0.177 s, with nothing published |
| `docs/phase5-results.md` | Results: trajectories that name the training step, and why the engine's last word is not the binding |
| `docs/phase6-results.md` | Results: hot update against a restart at two model sizes, and the kill inside a long collective that never returns |
| `docs/replay-validator.md` | Design: the token/logprob replay validator, what it proves, what it cannot, and how to produce replays on a GPU box |
| `diagnostics/README.md` | The GPU harnesses: one section per phase, the shared hygiene rules, and the pod the artifacts came from |

## Provenance of the vLLM analysis

Every `file:LINE` anchor in `docs/source-map-vllm-main.md` was verified against
the upstream repository at this commit:

| | |
|---|---|
| Repo | `https://github.com/vllm-project/vllm.git` |
| Commit | `00b7847c8036b667742b4efb21aab1de51fd4721` |
| Subject | *"[Perf] Use Conv3dLayer for MiniMax M3 patch embedding (#58512)"* |
| Committer date | `2026-09-24T07:41:36Z` |
| Confirmed by | `git ls-remote origin refs/heads/main` and `gh api repos/vllm-project/vllm/commits/main` |
| Re-check it | `git clone https://github.com/vllm-project/vllm /tmp/vllm && git -C /tmp/vllm checkout 00b7847c && python3 scripts/verify_anchors.py --vllm /tmp/vllm` |

The clone used for the audit is shallow (depth 1), so "NOT PRESENT" means absent
at this commit, not that it never existed.

## License

Apache License 2.0. See `LICENSE`.
