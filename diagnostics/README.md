# Diagnostics

Throwaway scripts that prove the *environment* before RolloutCore code is
written. Nothing here is part of the `rolloutcore` package, nothing here is
imported by it, and nothing here is covered by the test suite — these exist to
answer "is the pod's NCCL fine?" separately from "is our driver fine?".

| File | What it is |
|---|---|
| `upstream_nccl_2gpu.py` | vLLM's own `examples/rl/rlhf_http_nccl.py` at `00b7847c8036b667742b4efb21aab1de51fd4721`, adapted for a two-GPU pod. Proves the environment before any RolloutCore code is involved |
| `rolloutcore_cycle_2gpu.py` | Phase 3C: the same setup, but the update runs through `LifecycleRunner.run_cycle` and the real `NCCLWeightTransferDriver` |
| `rolloutcore_concurrency_2gpu.py` | Phase 4A: a generation spans the update, and must come out of it unchanged — the "one version per rollout" invariant, which no vLLM API enforces |
| `rolloutcore_failures_2gpu.py` | Phase 4B: the failure paths against a real engine — a failed drain, an identity mismatch, a SIGKILLed engine, and recovery — over three engine lifetimes |
| `rolloutcore_cache_2gpu.py` | Phase 4C: no cross-version KV reuse, asserted through the public prefix-cache metrics, plus a negative control that *makes* the reuse happen to show the reset step is load-bearing |
| `rolloutcore_updatekill_2gpu.py` | Phase 4D: SIGKILL the engine at the moment the controller enters `UPDATING`, and measure how the failure surfaces |
| `rolloutcore_trajectory_2gpu.py` | Phase 5: a trajectory that names the training step, not just the architecture — the record for a rollout that spans an update |
| `rolloutcore_benchmark_2gpu.py` | Phase 6: the hot-update cycle against a restart baseline, plus a kill aimed inside a multi-second broadcast |

## `upstream_nccl_2gpu.py`

Vendored verbatim (SPDX header retained) and then reduced from upstream's 3-GPU
layout to the two GPUs a Phase 3B pod has. The changes, and only these:

| Upstream | Here | Why |
|---|---|---|
| `INFERENCE_TP_SIZE = 2` | `1` | two GPUs total, not three |
| `SERVER_DEVICE_IDS = "0,1"` | `"0"` | inference takes one GPU |
| `TRAINER_DEVICE = "cuda:2"` | `"cuda:1"` | trainer takes the other |
| `--quantization fp8` | dropped | no FP8 requirement at 125M; one less variable |
| `POST /pause` | `POST /pause?mode=wait` | matches RolloutCore's drain semantics; plain `/pause` defaults to `abort` |
| — | `VLLM_ENABLE_V1_MULTIPROCESSING=1` | `mode="wait"` is refused by an in-process engine (`vllm/v1/engine/core.py:902`) |

Kept exactly as upstream has it: `facebook/opt-125m`, `--load-format dummy`
(the server starts with dummy weights so the before/after text change is visible),
and `--weight-transfer-config '{"backend": "nccl"}'`.

## Running it

On the two-GPU pod, from the repo root, with vLLM installed:

```bash
export HF_HOME=/workspace/hf
python3 diagnostics/upstream_nccl_2gpu.py
```

It starts `vllm serve` itself, generates once against dummy weights (expect
gibberish), broadcasts the real weights over NCCL, and generates again (expect
plausible text). The text change is the proof that the transport works.

If it hangs, that is diagnostic information, not a RolloutCore bug — see
`docs/phase3b-runbook.md` §8 for the NCCL hang checklist and §10–11 for the
failure modes found later. Re-run with `NCCL_DEBUG=INFO`. Do **not** reach for
`NCCL_TIMEOUT`: nothing in this path honours one, and a blocked collective has to
be bounded from outside the trainer (`docs/phase3b-runbook.md` §10.2).

## `rolloutcore_concurrency_2gpu.py`

Same two-GPU setup as Phase 3C, but the interesting event happens *during* a
generation: a 256-token rollout is admitted at `rc-0`, the cycle to `rc-1` starts
while it is still generating, and the drain has to wait for it.

The evidence is the text, not a log line. The server runs `--load-format dummy`,
so its output is degenerate; the update installs real `opt-125m` weights, whose
output is coherent and was measured in Phase 3A. A generation that *ends after*
the update but whose 16-token prefix matches the pre-update baseline is one that
ran entirely on the old weights.

Two things about it are load-bearing, and both are explained in the file:

* the rollout is released by the thread that owns its request, the moment that
  request returns — because `confirm_drained` taints if the engine reports
  quiescence while RolloutCore still counts live work. The engine goes idle a few
  milliseconds before the client sees the response, which is why the runner is
  given `drain_interval=2.0` rather than the 0.5 s default;
* the controller is therefore called from two threads at once, which is why
  `LifecycleController` now holds a reentrant lock
  (`tests/test_controller_threading.py`).

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_concurrency_2gpu.py
```

Same rules as above. The HTTP and launch helpers are duplicated between the two
RolloutCore scripts on purpose: each has to run standalone on a pod, and a shared
module would also hide the `/health` probe rule that
`tests/test_script_hygiene.py` enforces per file.

## `rolloutcore_failures_2gpu.py`

Four scenarios over three engine lifetimes, because each one has to start from a
known-good engine:

| | Scenario | The claim it tests |
|---|---|---|
| A | `/pause?mode=wait` times out while a 256-token generation is in flight | `DrainFailedError` is *not* a taint; the controller stays in DRAINING and a fresh attempt finishes the cycle with no restart |
| B | the target declares a different manifest than the trainer holds | `WeightIdentityMismatchError` before any mutation — asserted from the server log: `/start_weight_update` appears zero times |
| C | the engine's process group is SIGKILLed | a real `ConnectionRefusedError` from a mutating call taints; the taint is terminal and refuses locally |
| D | a third engine, a fourth controller | recovery is a fresh process; taint is controller-scoped |

A uses `drain_reissues=0` deliberately: with a reissue enabled the second
`/pause` can succeed, because the FAILED path aborts stragglers *before*
reissuing, and the engine is then already idle. That is good behaviour but it
would not exercise the failure.

C separates two paths that behave differently, which is the point of the phase:

* the **drain** path goes through `_observe`, which deliberately does not taint a
  failed read (`src/rolloutcore/runner.py:213`) — the engine is paused during DRAINING, so it
  cannot serve anything new. That reasoning still holds for a dead engine, so the
  controller is left in DRAINING, untainted, and the harness **records** what that
  costs rather than asserting it;
* the **mutating** path taints as designed, and `bootstrap()` against a dead
  engine is the deterministic way to reach it: the very first read fails with an
  unknown engine outcome.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_failures_2gpu.py
```

Takes a few minutes and produces `results/phase4b.json` plus one server log per
lifetime (`phase4b-server-{1,2,3}.log`). Each scenario records its own checks, so
a failure part-way still leaves a useful artifact.

## `rolloutcore_cache_2gpu.py`

Two phases in one run, because a passing "no reuse" assertion is only as good as
the proof that reuse *could* have happened.

**Phase 1 — the guarantee, and it gates.** With prefix caching on by default, `P`
is sent twice at `rc-0` to show the cache is real and measurable, a full cycle
installs `rc-1`, then `P` is sent again. The first post-update request must score
**zero** new cache hits (`vllm:prefix_cache_hits_total` /
`vllm:prefix_cache_queries_total`, read from the engine's own metrics — not from
RolloutCore's bookkeeping), and a second request must score hits again, or "no
reuse" would be indistinguishable from caching being off.

**Phase 2 — the hazard, recorded rather than asserted.** The trainer's first
decoder layer is negated in place and the update is driven entirely out of band:
`/pause?mode=wait&clear_cache=false`, broadcast, `/resume`. No controller, no
reset. If the old KV is still handed back — and it is, 32 hits — then the
`INVALIDATING` step is load-bearing rather than belt-and-braces. The negated
weights also keep the *same* manifest digest, which is the manifest-only blind
spot item 5 exists to fix, measured rather than argued.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_cache_2gpu.py
```

## `rolloutcore_updatekill_2gpu.py`

A healthy cycle first (which also times a real update), then a second cycle whose
engine process group is SIGKILLed the instant `ctrl.state` reads `UPDATING`. The
watcher runs on the main thread; the cycle runs on a daemon thread whose NCCL
session is bounded by the harness's own detection budget and `os._exit`, so a
blocked collective costs the budget instead of hanging the pod.

At 125M the whole update is ~100 ms, so the kill lands at its *edge* — the HTTP
calls around the collective — and detection is prompt (0.177 s,
`ConnectionResetError(104)`). Whether a kill *inside* a long collective is even
survivable is the benchmark's question, not this script's; the checks here gate
only on the fail-closed properties (nothing published, no rollout admissible,
engine dead), and whether the outcome arrived as a taint or a hang is recorded.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_updatekill_2gpu.py
```

## `rolloutcore_trajectory_2gpu.py`

Phase 4A's shape with provenance added, because 4A produced the case that makes
the whole record design necessary: a rollout admitted at `rc-0` that finishes
*after* the update to `rc-1`, so the engine reports `rc-1` for tokens that came
entirely from `rc-0`. The run asserts that divergence on purpose — the record
keeps `rc-0`, the engine's last word is `rc-1`, and the output (256 copies of
token id `0`) corroborates that the spanned rollout really ran on the dummy
weights.

It also needs `NCCLWeightTransferDriver.declare`, because a driver whose cached
identity cannot follow weights that changed has no way to re-declare what it is
about to install. Writes `results/phase5-trajectories.jsonl`; the provenance is
**declared**, which is the honest word — nothing here verifies a weight byte, and
`replay_ready` means the record *can support* a replay claim, not that one has
been made.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_trajectory_2gpu.py
```

## `rolloutcore_benchmark_2gpu.py`

Three engine lifetimes: **L1** the hot path with a state watcher so the cycle
decomposes into per-state dwell times, **L2** the restart baseline (kill, restart
on the same checkpoint, time to `/health` and to the first token), and **L3** a
kill aimed at half of L1's measured update duration — inside the collective on a
model whose broadcast is seconds wide.

L2 is deliberately generous to the baseline: it assumes the checkpoint is already
on disk and does not count the write a real restart-based update needs, so the
ratio is a floor on the hot path's advantage. At 125M that ratio is 130.63×; at
8B it is 8.71×, because the restart costs ~30 s of near-fixed startup while the
hot path scales with the payload.

In L3, whether the failure surfaced or hung is **recorded, not gated**
(`detail.deep_kill.verdict`): a hang there is an observation about vLLM, and a
FAIL should mean the benchmark failed. Fail-closed checks still gate.

| Flag | Why |
|---|---|
| `--model` | the only knob that changes the payload (default `facebook/opt-125m`) |
| `--skip-deep-kill` | L1 + L2 only — the clean 6/6 artifact |
| `--packed-buffer-gib` / `--packed-num-buffers` | receive-buffer sizing; `1` buffer is what makes an 8B fit |
| `--gpu-memory-utilization` | default 0.6, raised to 0.75 for 8B: leave the transfer room (see runbook §10.3) |
| `--max-model-len` | default 4096, **clamped** to the checkpoint's own `max_position_embeddings` |
| `--transfer-budget` | default 180 s; the harness's own bound on the broadcast |

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_benchmark_2gpu.py --model Qwen/Qwen3-8B --skip-deep-kill
```

Artifacts: `results/phase6.json` (125M, 11/11), `results/phase6-8b.json` (8B,
6/6), `results/phase6-8b-with-deepkill.json` (8B, 10/11 — the pre-fix artifact
whose eleventh check was the hang, since made non-gating).

This directory is deliberately outside the *type*-check scope: `scripts/test.sh`
runs `ruff` over it, but not `mypy`, because these files import `torch`,
`transformers`, `openai`, `requests` and `vllm`, none of which are installed on a
development machine or in CI.
