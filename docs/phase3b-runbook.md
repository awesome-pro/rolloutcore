# Phase 3B runbook — real NCCL weight transfer, two GPUs

**Status:** nothing here is implemented yet. Phase 3A passed (see
`docs/phase3a-results.md`); this is the plan for the next GPU session.

---

## 0. What NCCL is, and why it is the whole point

**NCCL is the NVIDIA Collective Communications Library.** It is how one GPU
sends tensors to another GPU **directly**, over NVLink or PCIe, without routing
the data through the CPU, a Python object, or an HTTP socket. It is the standard
transport for multi-GPU training, and vLLM uses it for weight synchronisation
too.

Three words carry all the operational pain:

- **Collective** — an operation every participant must join. The one we use is a
  *broadcast*: the trainer (rank 0) sends each tensor, and every inference worker
  receives it. Nobody can start early and nobody can finish while a peer is
  missing.
- **Rendezvous** — the handshake that forms the group before any data moves.
  Participants need a shared `master_address` + `master_port` (or a pre-minted
  NCCL unique ID), the total `world_size`, and their own `rank`.
- **World size** — in our cycle it is **inference workers + 1 trainer**:
  `world_size = get_world_size(...) + 1`
  (`examples/rl/rlhf_http_nccl.py:168`). For `opt-125m` at TP=1 that is 1 + 1 = 2.

So the division of labour in Phase 3B is:

```
control plane  (HTTP)   /pause, /start_weight_update, /update_weights metadata,
                        /finish_weight_update, /resume        -> already proven in 3A
data plane     (NCCL)   the tensors themselves, GPU -> GPU    -> Phase 3B
```

The failure mode of a collective is a **hang, not an exception**: if one rank
never arrives, the others wait forever. That is why the plan insists on proving
the environment with vLLM's own example *before* writing any RolloutCore code —
otherwise a hang tells you nothing about which layer is broken.

---

## 1. One pod with two GPUs — never two pods

NCCL needs the participants on the **same host**, sharing a fast GPU interconnect.
Two separate pods are two machines; cross-machine NCCL needs an InfiniBand/RoCE
fabric that a rental pod will not have configured for you. Two 1-GPU pods will
hang or fail, and the error will look like a bug in your code.

`opt-125m` at TP=1 needs exactly **2 GPUs**:

| GPU | Process |
|---|---|
| 0 | `vllm serve` — one inference worker (`--tensor-parallel-size 1`) |
| 1 | the trainer process (a Hugging Face model + the NCCL sender) |

Upstream's example defaults to **three** GPUs (TP=2 inference on 0-1, trainer on
2). We shrink it to two, which is cheaper and removes FP8 from the equation.

## 2. Deploy the two-GPU pod

1. **Stop the Phase 3A pod first** — after pulling `results/phase3a.json`.
2. Deploy with the **same template** as 3A
   (`runpod/pytorch:…-cu1281-torch280-ubuntu2404`), container disk **≥ 100 GB**
   (two model copies + wheels), no volume needed.
3. **GPU count 2, same GPU type.** Multi-GPU machines are usually Secure Cloud and
   limited to certain datacenters; look for a type that offers a `2x` option.
   Cost is roughly 2× the single-GPU rate (~$0.7/h for 2× A6000).
4. Expose TCP **22** (SSH, for rsync/tmux) and **8888** (Jupyter). Never expose
   8000 — the dev endpoints are unauthenticated.
5. Verify **both** GPUs before anything else:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.device_names(0))"
```

**Checkpoint B1 — proceed only if** `torch.cuda.device_count() == 2` and
`nvidia-smi` lists two devices.

## 3. Same install as Phase 3A

Identical to `docs/phase3a-runbook.md` §2–§3: copy the repo over SSH with
`tar`/`rsync`, `chown -R root:root`, then

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U uv
export VLLM_COMMIT=00b7847c8036b667742b4efb21aab1de51fd4721
uv pip install vllm --torch-backend=cu130 \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cu130
python3 -c "import vllm; print(vllm.__version__)"     # must contain g00b7847c8
```

**Checkpoint B2.** Same three verification lines as 3A, plus
`torch.cuda.device_count() == 2`.

## 4. Step 1 — prove upstream's own path first (this is the real "second step")

Do **not** start with RolloutCore code. Adapt vLLM's own example so that the
only thing under test is the environment:

```bash
cd /workspace/rolloutcore
cp /workspace/vllm/examples/rl/rlhf_http_nccl.py scripts/_upstream_nccl_2gpu.py
```

If you installed vLLM from a wheel there is no source tree; fetch just the file:

```bash
mkdir -p scripts && curl -sL -o scripts/_upstream_nccl_2gpu.py \
  https://raw.githubusercontent.com/vllm-project/vllm/00b7847c8036b667742b4efb21aab1de51fd4721/examples/rl/rlhf_http_nccl.py
```

Then make exactly these edits:

| Line | Upstream | Change to | Why |
|---|---|---|---|
| `:43` | `MODEL_NAME = "facebook/opt-125m"` | keep | same model as Phase 3A |
| `:48` | `INFERENCE_TP_SIZE = 2` | `1` | two GPUs total, not three |
| `:50` | `SERVER_DEVICE_IDS = "0,1"` | `"0"` | inference gets one GPU |
| `:51` | `TRAINER_DEVICE = "cuda:2"` | `"cuda:1"` | trainer gets the other |
| serve args | `--quantization fp8` | **delete** | no FP8 requirement; one less variable |
| serve args | `--load-format dummy` | **keep** | this is what makes the demo visible |
| serve args | `--weight-transfer-config '{"backend": "nccl"}'` | **keep** | the server needs a transfer engine (3A deliberately had none) |
| `pause_generation` | `requests.post(f"{base_url}/pause")` | `…/pause?mode=wait` | matches RolloutCore's drain semantics; plain `/pause` defaults to `abort` |
| env in `start_vllm_server` | `VLLM_SERVER_DEV_MODE=1` | add `VLLM_ENABLE_V1_MULTIPROCESSING=1` | `mode="wait"` requires the out-of-process engine |

Then, in tmux:

```bash
export HF_HOME=/workspace/hf
python3 scripts/_upstream_nccl_2gpu.py
```

**What success looks like:**

```
[trainer] Loading training model: facebook/opt-125m on cuda:1
[server] Launching: vllm serve facebook/opt-125m --tensor-parallel-size 1 --device-ids 0 ...
[transfer] Rendezvous at 10.x.x.x:PORT, world_size=2 (1 trainer + 1 vLLM worker)
BEFORE weight sync (dummy weights): ' the the the of of of ...'      <- gibberish
[sync] Broadcasting weights via NCCL...
[sync] Weight broadcast complete.
AFTER weight sync (real weights): ' the capital of France is Paris'   <- plausible
```

The **before/after text change is the proof**: dummy weights in, real weights
over NCCL, same server process, no restart. That is the milestone Phase 3A could
not reach.

**Checkpoint B3 — this is the go/no-go for the whole phase.** Paste back:
`nvidia-smi` (both GPUs), `world_size` and the rendezvous line, both generations,
and `NCCL_DEBUG` output if it hung. If NCCL fails here, the problem is the pod /
CUDA / NCCL / vLLM — not RolloutCore, and RolloutCore code should not be written
yet.

## 5. Step 2 — implement `NCCLWeightTransferDriver` (Mac-side, no GPU needed)

Once B3 passes, the remaining work is code, and the design is already written in
`docs/phase3b-notes.md`:

- `initialize()` wraps `WeightTransferTrainerFactory.trainer_init(...)`
  (`vllm/distributed/weight_transfer/factory.py:167`), which posts
  `/init_weight_transfer_engine` concurrently with opening the trainer endpoint.
- `transfer(target)` wraps `engine.send_weights()`
  (`vllm/distributed/weight_transfer/nccl_engine.py:319`) — **through a wrapper
  client**, because upstream calls `finish_weight_update()` with no version
  (`nccl_engine.py:361`) while the endpoint accepts one (`clients.py:89`). Without
  the wrapper the engine keeps reporting the old label and RolloutCore's
  VALIDATING state taints on every successful transfer.
- Before anything is sent, compute the identity from `ModuleSource.metadata()`
  plus a `WeightProvenance` and compare it with `target.identity`. A mismatch must
  fail **before** the rendezvous, which is why
  `WeightTransferNotConfiguredError` is a `RolloutCoreError` (no taint: nothing
  happened).

## 6. Step 3 — the first real RolloutCore cycle (Phase 3C)

Only after the driver exists. Same two GPUs, same `--load-format dummy` trick:

```
READY(rc-0) --dummy weights--> generate gibberish
   -> DRAINING -> QUIESCED -> UPDATING (NCCL, real weights)
   -> INVALIDATING -> VALIDATING (engine reports rc-1 while paused)
   -> RESUMING -> READY(rc-1)
   -> generate -> sensible output, same vLLM PID
```

That is when `LifecycleRunner.run_cycle()` becomes real for the first time — the
function Phase 3A deliberately never called.

## 7. When NCCL hangs (the expected first failure)

A hang is the normal symptom, so work through this list rather than guessing:

| Symptom | Likely cause |
|---|---|
| Both sides hang at rendezvous, before any transfer | `world_size` mismatch — it must be `1 + get_world_size()`, i.e. trainer + every inference worker |
| Trainer hangs opening its endpoint | `master_port` already in use, or `master_address` is not reachable from the worker (on one host, use the host IP from `get_ip()`, not a container-internal alias) |
| Transfer starts then stalls | the worker never entered the collective: check that `/init_weight_transfer_engine` actually returned on the server side |
| OOM right after starting the trainer | the trainer landed on GPU 0 alongside vLLM — check `TRAINER_DEVICE` and `--device-ids` |
| Works on one node but not another | PCIe topology; `NCCL_P2P_DISABLE=1` is the standard workaround (slower, still correct) |

Diagnostics: `NCCL_DEBUG=INFO` (add `NCCL_DEBUG_SUBSYS=INIT,COLL` for less noise),
and set a finite `NCCL_TIMEOUT` so a hang fails instead of blocking forever.

## 8. Cost, and what to send back

Two A6000-class GPUs are ~$0.7/h. Step 1 (the upstream proof) is a one-hour
session if it works first time. The driver implementation (step 2) costs **no GPU
time** — it is Mac-side work against a documented API — so the sequence is:
prove NCCL on the pod, stop the pod, implement, then rent again for 3C.

Send back from the 2-GPU session:

1. `nvidia-smi` with both GPUs, and `torch.cuda.device_count()`.
2. The rendezvous line: `world_size`, `master_address`, `master_port`.
3. Both generations (before/after) verbatim.
4. `NCCL_DEBUG=INFO` output if anything hung.
5. Whether `--quantization fp8` was dropped and `mode=wait` used.
