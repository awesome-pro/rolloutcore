# Diagnostics

The GPU harnesses. Each one is a **standalone script** that boots a real
`vllm serve`, drives it, prints every check as it happens, and writes one JSON
artifact to `results/`. They are the evidence behind the results documents in
`docs/` — `docs/phase3c-results.md`, `phase4a`–`phase4d`, `phase5`, `phase6` — so
what this directory is *not* is a set of throwaway probes. It is the measuring
apparatus.

They are deliberately not part of the `rolloutcore` package and nothing imports
them: a harness has to run on a pod with `torch`, `transformers`, `vllm` and
`openai` installed, none of which a laptop or CI has. `scripts/test.sh` runs
`ruff` over this directory but not `mypy`, and `tests/test_script_hygiene.py`
enforces the two rules below by AST, per file.

## The rules every harness follows

| Rule | Why |
|---|---|
| **Standalone, duplicated helpers** | each script must run on a pod with nothing else present; a shared module would also hide per-file hygiene failures from the AST test |
| **Probe `/health` by status only** | the endpoint returns **200 with an empty body** (`serve/instrumentator/health.py:28`); parsing it as JSON once cost a 900 s silent hang |
| **`os.killpg` only with `start_new_session=True`** | without it the kill reaches the harness's own process group and the script terminates itself |
| **Bound our own waits, then `os._exit`** | the trainer's collective has no timeout at all (runbook §10.2), so a hang must cost the harness's budget, not the pod |
| **Set the CUDA device at the top of every thread that touches the transfer** | `torch.cuda`'s current device is **thread-local**; a spawned thread starts on device 0 and NCCL reports only `invalid resource handle` (runbook §11) |
| **Write the artifact in `finally`** | a run that fails part-way still leaves the checks it reached |
| **Scope `git status` to the code dirs** | an untracked scratch file must not make an artifact look `dirty` |

Every script prints its checks as it goes, so the console is the live view and
the JSON is the record. `results/*.json` and `results/*.jsonl` are committed;
`results/*.log` is not.

## The harnesses

| File | Phase | What it establishes |
|---|---|---|
| `upstream_nccl_2gpu.py` | 3B | vLLM's own NCCL path works on this pod — the environment, before any RolloutCore code is involved |
| `rolloutcore_cycle_2gpu.py` | 3C | the first full RolloutCore cycle: dummy weights → real weights over NCCL, `READY(rc-0) → READY(rc-1)` in one server process |
| `rolloutcore_concurrency_2gpu.py` | 4A | an update requested while a 256-token generation is in flight is **held off** until it finishes — no rollout spans a mutation |
| `rolloutcore_failures_2gpu.py` | 4B | the failure paths: a drain that times out, an identity mismatch, a SIGKILLed engine, recovery — over three engine lifetimes |
| `rolloutcore_cache_2gpu.py` | 4C | no cross-version KV reuse, from the engine's own metrics, plus a negative control that *makes* the reuse happen |
| `rolloutcore_updatekill_2gpu.py` | 4D | SIGKILL at the instant `UPDATING` begins, and how the failure surfaces |
| `rolloutcore_trajectory_2gpu.py` | 5 | a trajectory that names the training step, with the engine label read at the request's own completion |
| `rolloutcore_benchmark_2gpu.py` | 6 | hot cycle vs restart baseline at two model sizes, plus a kill aimed inside a multi-second broadcast |

## The pod the committed artifacts came from

Two NVIDIA RTX 3090s (24 GB, compute capability 8.6), driver 595.71.05, CUDA
13.2, one node, **P2P disabled by topology** with the NCCL transport falling back
to `SHM/direct/direct` (runbook §11). `opt-125m` runs on GPU 0 with TP=1 and the
bf16 trainer on GPU 1; the 8B runs use the same layout with
`--gpu-memory-utilization 0.75 --packed-num-buffers 1`.

An 8B broadcast over that fabric is a host-path number, which is why
`rolloutcore_benchmark_2gpu.py` now records the whole environment —
`detail["environment"]`: GPU names and memory, driver, compute capability, torch,
CUDA, NCCL, per-pair `p2p_peer_access`, and the raw `nvidia-smi topo -m` — before
it starts the server. The earlier artifacts predate that block; they are left as
measured, and `docs/phase6-results.md` carries the retrospective table.

## `upstream_nccl_2gpu.py`

Vendored from upstream's `examples/rl/rlhf_http_nccl.py` at
`00b7847c8036b667742b4efb21aab1de51fd4721` (SPDX header retained) and reduced from
its 3-GPU layout to a 2-GPU pod:

| Upstream | Here | Why |
|---|---|---|
| `INFERENCE_TP_SIZE = 2` | `1` | two GPUs total, not three |
| `SERVER_DEVICE_IDS = "0,1"` | `"0"` | inference takes one GPU |
| `TRAINER_DEVICE = "cuda:2"` | `"cuda:1"` | trainer takes the other |
| `--quantization fp8` | dropped | no FP8 requirement at 125M; one less variable |
| `POST /pause` | `POST /pause?mode=wait` | matches RolloutCore's drain semantics; plain `/pause` defaults to `abort` |
| — | `VLLM_ENABLE_V1_MULTIPROCESSING=1` | `mode="wait"` is refused by an in-process engine (`vllm/v1/engine/core.py:902`) |

Everything else is upstream's: `facebook/opt-125m`, `--load-format dummy` (so the
before/after text change is visible), `--weight-transfer-config
'{"backend": "nccl"}'`.

```bash
export HF_HOME=/workspace/hf
python3 diagnostics/upstream_nccl_2gpu.py
```

It starts the server, generates against dummy weights (gibberish), broadcasts the
real weights, and generates again (plausible text). The text change is the proof.

## `rolloutcore_cycle_2gpu.py`

Phase 3C. The same two-GPU setup, but the whole update goes through
`LifecycleRunner.run_cycle()` and the real `NCCLWeightTransferDriver` — bootstrap
at `rc-0`, broadcast, invalidate, validate, resume, and a generation that
reproduces Phase 3A's text token for token. This is the run that proved the
control plane and the data plane compose; it is also where the identity limit
showed up: the trainer's manifest digest alone could not separate dummy weights
from real ones (`925369d663bc` for both).

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_cycle_2gpu.py        # → results/phase3c.json
```

## `rolloutcore_concurrency_2gpu.py`

Phase 4A — the invariant, and the claim is about **delay, not survival**. A
256-token generation (`ignore_eos`, so it cannot finish early) is admitted at
`rc-0`; 0.4 s later the cycle to `rc-1` starts; the drain waits 1.375 s for it and
the weight mutation happens only after it has returned. The evidence is the text:
the server's `rc-0` output is degenerate, the update installs real weights whose
output is coherent, and a 256-token generation whose 16-token prefix matches the
pre-update baseline ran entirely on `rc-0`. The journal settles the ordering —
`finish_rollout` comes **before** `begin_update`.

Two things the harness exists to expose:

* the rollout must be **released by the thread that owns its request**, the moment
  it returns: `confirm_drained` taints if the engine reports quiescence while
  RolloutCore still counts live work, and the engine goes idle milliseconds before
  the client sees the response. `drain_interval=2.0` is the margin that wins the
  race (`drain_polls: 2`);
* the controller is therefore used from two threads, which is why
  `LifecycleController` holds a reentrant lock
  (`tests/test_controller_threading.py`).

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_concurrency_2gpu.py  # → results/phase4a.json
```

## `rolloutcore_failures_2gpu.py`

Phase 4B. Four scenarios over three engine lifetimes, each starting from a
known-good engine:

| | Scenario | The claim it tests |
|---|---|---|
| A | `/pause?mode=wait` times out while a 256-token generation is in flight | `DrainFailedError` is *not* a taint; the controller stays in DRAINING and a fresh attempt finishes the cycle with no restart |
| B | the target declares a different manifest than the trainer holds | `WeightIdentityMismatchError` before any mutation — asserted from the server log: `/start_weight_update` appears zero times |
| C | the engine's process group is SIGKILLed | a real `ConnectionRefusedError` from a mutating call taints; the taint is terminal and refuses locally |
| D | a third engine, a fourth controller | recovery is a fresh process; taint is controller-scoped |

A uses `drain_reissues=0` deliberately: with a reissue enabled the second `/pause`
can succeed, because the FAILED path aborts stragglers *before* reissuing, and the
engine is then already idle. That is good behaviour but it would not exercise the
failure.

C separates two paths that behave differently, which is the point of the phase:

* the **drain** path goes through `_observe`, which deliberately does not taint a
  failed read (`src/rolloutcore/runner.py:213`) — the engine is paused during
  DRAINING, so it cannot serve anything new. That reasoning still holds for a dead
  engine, so the controller is left in DRAINING, untainted, and the harness
  **records** what that costs rather than asserting it;
* the **mutating** path taints as designed, and `bootstrap()` against a dead
  engine is the deterministic way to reach it: the very first read fails with an
  unknown outcome.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_failures_2gpu.py     # → phase4b.json + one server log per lifetime
```

Takes a few minutes. Each scenario records its own checks, so a failure part-way
still leaves a useful artifact.

## `rolloutcore_cache_2gpu.py`

Phase 4C. Two lanes, because a passing "no reuse" assertion is only as good as the
proof that reuse *could* have happened.

**Phase 1 — the guarantee, and it gates.** With prefix caching on by default, `P`
is sent twice at `rc-0` to show the cache is real and measurable, a full cycle
installs `rc-1`, then `P` is sent again. The first post-update request must score
**zero** new hits (`vllm:prefix_cache_hits_total` /
`vllm:prefix_cache_queries_total`, read from the engine — not from RolloutCore's
bookkeeping), and a second request must score hits again, or "no reuse" would be
indistinguishable from caching being off.

**Phase 2 — the hazard, recorded rather than asserted.** The trainer's first
decoder layer is negated in place and the update is driven entirely out of band:
`/pause?mode=wait&clear_cache=false`, broadcast, `/resume`. No controller, no
reset. The old KV *is* handed back — 32 hits computed under the previous weights,
producing different tokens — which is what makes the reset load-bearing. The
negated weights also keep the *same* manifest digest: the manifest-only blind spot
item 5 exists to fix, measured rather than argued.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_cache_2gpu.py        # → results/phase4c.json
```

## `rolloutcore_updatekill_2gpu.py`

Phase 4D. A healthy cycle first (which also times a real update), then a second
cycle whose engine process group is SIGKILLed the instant `ctrl.state` reads
`UPDATING`. The watcher runs on the main thread; the cycle runs on a daemon thread
bounded by the harness's own budget and `os._exit`, so a blocked collective costs
the budget instead of the pod.

At 125M the whole update phase is ~100 ms, so the kill lands inside that phase but
the failure surfaces on the HTTP socket (0.177 s, `ConnectionResetError(104)`) —
this measurement cannot say whether the process died during the collective or in
the calls around it. Whether a kill *inside* a long collective is survivable is
the benchmark's question, not this script's. The checks gate only on the
fail-closed properties (nothing published, no rollout admissible, engine dead);
whether the outcome was a taint or a hang is recorded.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_updatekill_2gpu.py   # → results/phase4d.json
```

## `rolloutcore_trajectory_2gpu.py`

Phase 5. Phase 4A's shape with provenance added, for the one case where two weight
sets are both live: `R-long` is admitted at `rc-0` while the update to `rc-1` is
requested, the drain holds the mutation until it returns, and `R2` is admitted
afterwards at `rc-1`.

The engine label is read **in the request thread, between the response arriving
and the rollout being released** — the only moment at which it is known to be the
label the request completed under, because the drain cannot confirm (so the update
cannot begin) while RolloutCore still counts the rollout. Reading it after the
cycle instead records the *post*-update label, which is what an earlier version of
this harness did; the committed `results/phase5.json` carries that defect and
`docs/phase5-results.md` documents it.

The run needs `NCCLWeightTransferDriver.declare`, because a driver whose cached
identity cannot follow weights that changed has no way to re-declare what it is
about to install. It writes `results/phase5-trajectories.jsonl`; the provenance is
**declared**, which is the honest word — nothing here verifies a weight byte.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_trajectory_2gpu.py   # → phase5.json + phase5-trajectories.jsonl
```

## `rolloutcore_benchmark_2gpu.py`

Phase 6. Three engine lifetimes: **L1** the hot path with a state watcher so the
cycle decomposes into per-state dwell times, **L2** the restart baseline (kill,
restart on the same checkpoint, time to `/health` and to the first token), and
**L3** a kill aimed at half of L1's measured update duration — inside the
collective on a model whose broadcast is seconds wide.

L2 is deliberately generous to the baseline: it assumes the checkpoint is already
on disk and does not count the write a real restart-based update needs, so the
ratio is a floor on the hot path's advantage. At 125M that ratio is 130.63×; at
8B it is 8.71×, because the restart costs ~30 s of near-fixed startup while the
hot path scales with the payload.

In L3 the outcome is **recorded, not gated** (`detail.deep_kill.verdict`): a hang
there is an observation about vLLM, and a FAIL should mean the benchmark failed.
The fail-closed checks still gate.

| Flag | Why |
|---|---|
| `--model` | the only knob that changes the payload (default `facebook/opt-125m`) |
| `--skip-deep-kill` | L1 + L2 only — the clean 6/6 artifact |
| `--packed-buffer-gib` / `--packed-num-buffers` | receive-buffer sizing; `1` buffer is what makes an 8B fit |
| `--gpu-memory-utilization` | default 0.6, raised to 0.75 for 8B: leave the transfer room (runbook §10.3) |
| `--max-model-len` | default 4096, **clamped** to the checkpoint's own `max_position_embeddings` |
| `--transfer-budget` | default 180 s; the harness's own bound on the broadcast |

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_benchmark_2gpu.py --model Qwen/Qwen3-8B --skip-deep-kill
```

Artifacts: `results/phase6.json` (125M, 11/11), `results/phase6-8b.json` (8B,
6/6), `results/phase6-8b-with-deepkill.json` (8B, 10/11 — the pre-fix artifact
whose eleventh check was the hang, since made non-gating). Every run now also
carries `detail["environment"]`, so the artifact says what machine produced it.

## When something goes wrong

`docs/phase3b-runbook.md` is the operational document: §8 is the NCCL hang
checklist, §10 the four failure modes these phases found (a receive-side OOM that
presents as a hang, the unbounded collective, the KV cache with no transfer
headroom, and the failed-drain classification), and §11 the thread-local CUDA
device trap that cost five pod runs. Read §10 before blaming the transport.
