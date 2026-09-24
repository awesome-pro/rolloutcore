# Diagnostics

Throwaway scripts that prove the *environment* before RolloutCore code is
written. Nothing here is part of the `rolloutcore` package, nothing here is
imported by it, and nothing here is covered by the test suite — these exist to
answer "is the pod's NCCL fine?" separately from "is our driver fine?".

| File | What it is |
|---|---|
| `upstream_nccl_2gpu.py` | vLLM's own `examples/rl/rlhf_http_nccl.py` at `00b7847c8036b667742b4efb21aab1de51fd4721`, adapted for a two-GPU pod. Proves the environment before any RolloutCore code is involved |
| `rolloutcore_cycle_2gpu.py` | Phase 3C: the same setup, but the update runs through `LifecycleRunner.run_cycle` and the real `NCCLWeightTransferDriver` |

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

This directory is deliberately outside the lint and type-check scope
(`scripts/test.sh` and CI check `src`, `tests` and `scripts` only): the file
imports `torch`, `transformers`, `openai`, `requests` and `vllm`, none of which
are installed on a development machine or in CI.
