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
`docs/phase3b-runbook.md` §7 for the NCCL hang checklist. Re-run with
`NCCL_DEBUG=INFO` and a finite `NCCL_TIMEOUT` so a hang fails instead of
blocking forever.

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
  failed read (`runner.py:213`) — the engine is paused during DRAINING, so it
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

This directory is deliberately outside the *type*-check scope: `scripts/test.sh`
runs `ruff` over it, but not `mypy`, because these files import `torch`,
`transformers`, `openai`, `requests` and `vllm`, none of which are installed on a
development machine or in CI.
