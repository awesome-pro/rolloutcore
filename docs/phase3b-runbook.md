# Phase 3B runbook: real NCCL weight transfer, two GPUs

**Status: steps 1 and 2 are done.** Upstream's NCCL path was proved on a 2× RTX
3090 node (driver 595.71.05, CUDA 13.2): real weights broadcast over NCCL,
dummy→coherent output, no restart. The driver is implemented in
`src/rolloutcore/adapters/nccl.py` with 20 GPU-free tests. What remains is
Phase 3C: running `diagnostics/rolloutcore_cycle_2gpu.py` so the update goes
through `LifecycleRunner.run_cycle`.

---

## 0. What NCCL is, and why it is the whole point

**NCCL is the NVIDIA Collective Communications Library.** It is how one GPU
sends tensors to another GPU **directly**, over NVLink or PCIe, without routing
the data through the CPU, a Python object, or an HTTP socket. It is the standard
transport for multi-GPU training, and vLLM uses it for weight synchronisation
too.

Three words carry all the operational pain:

- **Collective**: an operation every participant must join. The one we use is a
  *broadcast*: the trainer (rank 0) sends each tensor, and every inference worker
  receives it. Nobody can start early and nobody can finish while a peer is
  missing.
- **Rendezvous**: the handshake that forms the group before any data moves.
  Participants need a shared `master_address` + `master_port` (or a pre-minted
  NCCL unique ID), the total `world_size`, and their own `rank`.
- **World size**: in our cycle it is **inference workers + 1 trainer**, i.e.
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
the environment with vLLM's own example *before* writing any RolloutCore code;
otherwise a hang tells you nothing about which layer is broken.

---

## 1. One pod with two GPUs: never two pods

NCCL needs the participants on the **same host**, sharing a fast GPU interconnect.
Two separate pods are two machines; cross-machine NCCL needs an InfiniBand/RoCE
fabric that a rental pod will not have configured for you. Two 1-GPU pods will
hang or fail, and the error will look like a bug in your code.

`opt-125m` at TP=1 needs exactly **2 GPUs**:

| GPU | Process |
|---|---|
| 0 | `vllm serve`: one inference worker (`--tensor-parallel-size 1`) |
| 1 | the trainer process (a Hugging Face model + the NCCL sender) |

Upstream's example defaults to **three** GPUs (TP=2 inference on 0-1, trainer on
2). We shrink it to two, which is cheaper and removes FP8 from the equation.

## 2. Deploy the two-GPU pod

1. **Stop the Phase 3A pod first**, after pulling `results/phase3a.json`.
2. Deploy with the **same template** as 3A
   (`runpod/pytorch:…-cu1281-torch280-ubuntu2404`), container disk **≥ 100 GB**
   (two model copies + wheels), no volume needed.
3. **GPU count 2, same GPU type.** Multi-GPU machines are usually Secure Cloud and
   limited to certain datacenters; look for a type that offers a `2x` option.
   Cost is roughly 2× the single-GPU rate (~$0.7/h for 2× A6000).
4. Expose TCP **22** (SSH, for rsync/tmux) and **8888** (Jupyter). Never expose
   8000: the dev endpoints are unauthenticated.
5. **Check the driver before you deploy, not after.** RunPod's GPU card shows an
   **"Available CUDA versions"** field; that is the host driver's maximum, and
   it decides your wheel variant:

   | Card says | Driver | Variant |
   |---|---|---|
   | `13.0` | ≥ R580 | `cu130` |
   | `12.9` | ≥ R575 | `cu129` |
   | `12.8` or older | < R575 | **neither ships natively** |

   CUDA 13 needs R580+ (`gpu.cuda.inc.md:327`). A host on driver 570 reports
   CUDA 12.8 and cannot run a cu130 build at all (observed on an A40 host). Two
   remedies: pick a machine whose card advertises 13.0, or use the CUDA
   forward-compatibility route (`gpu.cuda.inc.md:325-340`), which vLLM's own
   image bundles for pro/datacenter GPUs. Redeploying is usually cheaper.

   Hosts actually seen on RunPod (driver varies per machine, not per GPU type,
   so read the field, do not assume from the card name):

   | Host GPU | Driver | Max CUDA | cu130? |
   |---|---|---|---|
   | 1× A6000 | 580.159.03 | 13.0 | yes: Phase 3A ran here |
   | 2× A40 | 570.195.03 | 12.8 | **no**, below the R580 floor |
   | 2× RTX 2000 Ada | (card advertised 13.0) | 13.0 | yes (not deployed) |

   VRAM is not the constraint: `opt-125m` is 250 MB, so a 2× 16 GB machine is as
   good as a 2× 48 GB one and often cheaper.

6. Verify **both** GPUs, and prove a kernel actually launches: `is_available()`
   alone is not enough, because it can report `True` while context creation
   fails on an older driver:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python3 -c "import torch; a=torch.randn(8, device='cuda:0'); print('kernel ok', float((a+a).sum())); print('devices', torch.cuda.device_count())"
```

**Checkpoint B1. Proceed only if** `torch.cuda.device_count() == 2`, the kernel
test prints a value, and `nvidia-smi` lists two devices.

## 3. Same install as Phase 3A

Identical to `docs/phase3a-runbook.md` §2 and §3: copy the repo over SSH with
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

## 4. Step 1: prove upstream's own path first (this is the real "second step")

Do **not** start with RolloutCore code. Adapt vLLM's own example so that the
only thing under test is the environment:

The adapted file is already in the repository at
`diagnostics/upstream_nccl_2gpu.py`: vendored from upstream at the audited
commit and reduced to two GPUs before anyone rents a machine, so no editing
happens on paid GPU time. For reference, these are the only changes it makes:

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
python3 diagnostics/upstream_nccl_2gpu.py
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

**Checkpoint B3. This is the go/no-go for the whole phase.** Paste back:
`nvidia-smi` (both GPUs), `world_size` and the rendezvous line, both generations,
and `NCCL_DEBUG` output if it hung. If NCCL fails here, the problem is the pod /
CUDA / NCCL / vLLM, not RolloutCore. RolloutCore code should not be written
yet.

## 5. Step 2: implement `NCCLWeightTransferDriver` (Mac-side, no GPU needed)

Once B3 passes, the remaining work is code, and the design is already written in
`docs/phase3b-notes.md`:

- `initialize()` wraps `WeightTransferTrainerFactory.trainer_init(...)`
  (`vllm/distributed/weight_transfer/factory.py:167`), which posts
  `/init_weight_transfer_engine` concurrently with opening the trainer endpoint.
- `transfer(target)` wraps `engine.send_weights()`
  (`vllm/distributed/weight_transfer/nccl_engine.py:319`): **through a wrapper
  client**, because upstream calls `finish_weight_update()` with no version
  (`nccl_engine.py:361`) while the endpoint accepts one (`clients.py:89`). Without
  the wrapper the engine keeps reporting the old label and RolloutCore's
  VALIDATING state taints on every successful transfer.
- Before anything is sent, compute the identity from `ModuleSource.metadata()`
  plus a `WeightProvenance` and compare it with `target.identity`. A mismatch must
  fail **before** the rendezvous, which is why
  `WeightTransferNotConfiguredError` is a `RolloutCoreError` (no taint: nothing
  happened).

## 6. Step 3: the first real RolloutCore cycle (Phase 3C)

Only after the driver exists. Same two GPUs, same `--load-format dummy` trick:

```
READY(rc-0) --dummy weights--> generate gibberish
   -> DRAINING -> QUIESCED -> UPDATING (NCCL, real weights)
   -> INVALIDATING -> VALIDATING (engine reports rc-1 while paused)
   -> RESUMING -> READY(rc-1)
   -> generate -> sensible output, same vLLM PID
```

That is when `LifecycleRunner.run_cycle()` becomes real for the first time: the
function Phase 3A deliberately never called.

### Running it

Kill anything left over from an earlier attempt first. The harness starts its own
server with `start_new_session=True`, so a Ctrl-C'd run can leave one holding
port 8000, and the next run then dies with "vLLM exited before becoming ready":

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_cycle_2gpu.py
```

Server output goes to `results/phase3c-server.log`, **not** the terminal. That is
deliberate: `vllm serve` logs every readiness poll, and in the first attempt that
buried the harness's own output so completely that a healthy server looked like a
hang. The script now prints its own progress and dumps the log tail on failure.

Expect:

```
[server] ready after 41.2s (pid 5176)
[trainer] loading facebook/opt-125m on cuda:1
[identity] rc-1: 197 tensors, ... (manifest)
[cycle] bootstrap
[before] ' the the the of of of ...'
[cycle] run_cycle -> rc-1
[after]  ' the capital of France is Paris'

=== Phase 3C ===
  [PASS] committed_rc1
  [PASS] engine_label_rc1
  [PASS] engine_resumed
  [PASS] output_changed
  [PASS] server_never_restarted
  [report] /workspace/rolloutcore/results/phase3c.json
```

`committed_rc1` proves the commit point is the resume, and `engine_label_rc1`
proves the version handshake: upstream's `send_weights()` calls
`finish_weight_update()` with no version (`nccl_engine.py:361`), so without
`RolloutCoreWeightSyncClient` supplying one, VALIDATING would taint on a
*successful* transfer.

If the run goes quiet again, do not wait out the timeout: `tail -f
results/phase3c-server.log` shows what the server is actually doing.

## 7. Phase 4A: one version per rollout, under a live update

Phase 3C proved the cycle works. Phase 4A asks the question the cycle exists for:
**what happens to a rollout that is already generating when an update is
requested?**

vLLM cannot answer that itself. PR #49040 added a `weight_version` query API and
deliberately *removed* binding a version to a request, because one request may
span two versions. So the guarantee is RolloutCore's to keep, and this is the run
that checks it under real concurrency.

```bash
pkill -f "vllm serve"; sleep 3
python3 diagnostics/rolloutcore_concurrency_2gpu.py
```

One 256-token generation (`ignore_eos`, so it cannot finish early) is admitted at
`rc-0`; 0.4 s later the cycle to `rc-1` starts; the drain has to wait for it, and
the weight mutation happens only after it returns.

**Why the text is the evidence.** The server runs `--load-format dummy`, so `rc-0`
produces degenerate output. The update installs real `opt-125m` weights, whose
output is coherent and was measured in Phase 3A. So a generation whose 16-token
prefix matches the pre-update baseline is one that ran entirely on the old
weights. The two texts are impossible to confuse.

```
=== Phase 4A ===
  [PASS] pre_update_binding_is_rc0
  [PASS] rollout_released_cleanly
  [PASS] inflight_overlapped_the_cycle   (named ..._update in the committed artifact)
  [PASS] inflight_ran_to_length
  [PASS] inflight_used_rc0_weights
  [PASS] inflight_is_not_rc1
  [PASS] drain_outlasted_the_request
  [PASS] committed_rc1
  [PASS] post_update_binding_is_rc1
  [PASS] engine_label_rc1
  [PASS] engine_resumed
  [PASS] server_never_restarted
  rc-0 baseline : '<s><s><s>...'
  in-flight     : '<s><s><s>...'      <- 256 tokens, all produced at rc-0
  rc-1 after    : ' the capital of the French Republic.\n\n...'
  timing        : request 2.1s, 1.6s of it under the cycle, cycle 2.4s over 2 poll(s)
```

`inflight_used_rc0_weights` is the claim; `drain_outlasted_the_request` is why it
held. The journal settles the ordering: `finish_rollout` comes **before**
`begin_update`, so the rollout spanned the *drain*, not the mutation. If the drain
had *not* waited, the request would have been aborted and
`rollout_released_cleanly` / `inflight_ran_to_length` would fail instead.

**If it reports a taint instead**, `DrainDisagreementError` means the engine said
"drained" while a rollout was still counted: the release lost the race against
the drain poll. That is the sharp edge the script documents rather than hides:
see `docs/phase3c-results.md` and the `drain_interval` note in the script.

## 8. When NCCL hangs (the expected first failure)

A hang is the normal symptom, so work through this list rather than guessing:

| Symptom | Likely cause |
|---|---|
| Both sides hang at rendezvous, before any transfer | `world_size` mismatch: it must be `1 + get_world_size()`, i.e. trainer + every inference worker |
| Trainer hangs opening its endpoint | `master_port` already in use, or `master_address` is not reachable from the worker (on one host, use the host IP from `get_ip()`, not a container-internal alias) |
| Transfer starts then stalls | the worker never entered the collective: check that `/init_weight_transfer_engine` actually returned on the server side |
| OOM right after starting the trainer | the trainer landed on GPU 0 alongside vLLM; check `TRAINER_DEVICE` and `--device-ids` |
| Works on one node but not another | PCIe topology; `NCCL_P2P_DISABLE=1` is the standard workaround (slower, still correct) |
| `NCCL WARN Cuda failure 400 'invalid resource handle'` | **not** a transport or topology problem: the calling thread's current CUDA device. See §11. |
| Trainer blocked forever, engine still answers `/health` | a receive-side OOM, or a kill inside a long collective. See §10. |

Diagnostics: `NCCL_DEBUG=INFO` (add `NCCL_DEBUG_SUBSYS=INIT,COLL` for less noise).

**There is no `NCCL_TIMEOUT` to set.** This section used to recommend one; Phase 6
disproved it. The trainer forms the group through vLLM's own
`PyNcclCommunicator` (`nccl_engine.py:156` → `nccl_common.py:181`), the only
timeout in that path is on *teardown* (`pynccl.py:237`), and there is no
`NCCL_TIMEOUT`-style knob anywhere in the tree. A blocked collective does not fail;
it blocks. Bound it from outside (the harnesses here use their own budget plus
`os._exit`), and never plan on the trainer noticing.

## 9. Cost, and what to send back

Two A6000-class GPUs are ~$0.7/h. Step 1 (the upstream proof) is a one-hour
session if it works first time. The driver implementation (step 2) costs **no GPU
time** (it is Mac-side work against a documented API), so the sequence is:
prove NCCL on the pod, stop the pod, implement, then rent again for 3C.

Send back from the 2-GPU session:

1. `nvidia-smi` with both GPUs, and `torch.cuda.device_count()`.
2. The rendezvous line: `world_size`, `master_address`, `master_port`.
3. Both generations (before/after) verbatim.
4. `NCCL_DEBUG=INFO` output if anything hung.
5. Whether `--quantization fp8` was dropped and `mode=wait` used.

---

## 10. Addendum: findings from Phases 4C to 6 (all measured on the 2-GPU pod)

Four ways this setup fails that are not in the list above, because they were not
known until the later phases measured them. The first three are properties of the
concurrent broadcast; the fourth is a property of our own code (see §11).

| Symptom | What it actually is | What to do |
|---|---|---|
| Trainer blocked forever, engine alive and answering `/health` | a **receive-side OOM** raised in the server, or a kill inside a long collective | read the server log for `CUDA out of memory`; leave memory headroom (§10.1) |
| A kill during the update never returns | there is **no timeout** on the trainer's collective (§10.2) | bound it from outside the trainer |
| OOM only on models bigger than ~1 GB | the **KV cache is sized with no headroom** for transfer buffers (§10.3) | `--gpu-memory-utilization 0.6`, `--packed-num-buffers 1` |
| A drain fails and the controller sits in `DRAINING` untainted | a failed read is deliberately not a taint (`src/rolloutcore/runner.py:213`) | check `/health` before assuming the engine is merely paused (`docs/phase4b-results.md`) |

### 10.1 An OOM in a concurrent broadcast presents as a hang

The 8B run's first failure, verbatim from the engine log:

```
CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has 531.00 MiB free.
```

That 1.16 GiB is the receive buffer for the 1.24 GB embedding tensor. The engine
logs the OOM and **stays alive** (`/health` keeps returning 200, the process does
not exit) while the trainer, already inside the collective, blocks forever.
`complete_weight_update` never returns, so RolloutCore is never told anything
happened and the controller sits in `UPDATING` with no taint. There is no
client-side error to catch; the only evidence is the server log.

If a transfer hangs, read the engine log **first**, before the NCCL checklist
above.

### 10.2 The collective has no timeout

The trainer joins through vLLM's `PyNcclCommunicator` (`nccl_engine.py:156` →
`nccl_common.py:181`). The only timeout in that path is on teardown, and the
source's own comment says a failed join "leaves the peer blocked in
`ncclCommInitRank` until timeout" (`pynccl.py:237`). Nothing bounds the collective
itself, so the observable behaviour depends on the length of the broadcast:

| Broadcast | Kill inside it | Observed result |
|---|---|---|
| 0.10 s (125M) | kill at 0.154 s, aimed at 0.050 s | `EngineTaintedError` in **0.1738 s**, controller `TAINTED` |
| 4.13 s (8B) | kill at 2.174 s, aimed at 2.063 s (≈53% through) | **never returned**: 120.2254 s was the harness's own budget, `outcome: null`, controller stuck in `UPDATING`, `tainted: false` |

The 8B number is not a failure that took 120 s. It is the harness terminating
itself with `os._exit` after its budget, because the path underneath has no bound
of its own. Phase 4D's fast 0.1767 s detection is the same code path in the narrow
regime; do not generalise it to bigger models.

**Operational rule:** any process that drives a broadcast needs its own watchdog
and its own way to exit. Do not wait for the trainer to fail, and do not expect a
taint: if the broadcast never returns, RolloutCore never learns that anything
changed, which is exactly the state that needs an operator.

### 10.3 The KV cache is sized with no room for the transfer

vLLM allocates the KV cache to fill `--gpu-memory-utilization` at startup and
reserves **nothing** for the weight-transfer receive buffers, which the packed
consumer allocates lazily per chunk (`torch.empty(packing_tensor_sizes[buffer_idx])`).
At 125M this is invisible: a ~250 MB model fits inside one sub-buffer chunk. A
16 GB checkpoint streams ~16 chunks of ~1 GiB, and the first chunk that does not
fit turns into §10.1.

The harnesses now start the server with `--gpu-memory-utilization 0.6` and
`--max-model-len` clamped to the checkpoint's own `max_position_embeddings`; the
8B runs additionally use `--packed-num-buffers 1` so only one receive buffer is
live at a time. This is configuration, not a fix in vLLM: the engine has no way
to know a trainer is about to land gigabytes on it.

### 10.4 Checklist before blaming the environment

1. Engine log: any `CUDA out of memory`, any traceback.
2. Is `/health` still 200 while the trainer is blocked? Then the engine is alive
   and the problem is downstream of the collective, not the transport.
3. How long was the broadcast supposed to take? Compare the kill/observation time
   against it: under a second, expect an error; multi-second, expect a hang.
4. Free GPU memory on the inference card right after startup.
5. Only then the transport: `NCCL_DEBUG=INFO`, ports, `world_size`.

## 11. The thread-local device trap

Five consecutive pod runs (two transports, three model sizes, one fresh
container) failed identically:

```
NCCL WARN Cuda failure 400 'invalid resource handle'
```

It was the harness, and the error names neither the device nor the thread.

**Cause.** PyTorch's current CUDA device is **thread-local**. A newly spawned
thread starts on device 0. The harness runs the cycle on a daemon thread (to bound
a hang), and that thread defaulted to device 0 while the model, the KV cache and
the NCCL communicator had been created on device 1. vLLM's packed producer builds
its CUDA streams from `torch.accelerator.current_device_index()`
(`packed_tensor.py:23-24`), so a GPU-1 broadcast was issued on GPU-0 streams. NCCL
reports `invalid resource handle` and nothing else.

**Two theories that were wrong, retracted here so nobody re-runs them:** a
`/dev/shm` sizing problem, and `NCCL_P2P_DISABLE`. P2P is disabled by topology on
this host and always was; the transport line (`SHM/direct/direct`, or
`NET/Socket/0` with `NCCL_SHM_DISABLE=1`) was never the issue.

**The rule.** Any thread that touches CUDA must set its own device first:

```python
torch.cuda.set_device(TRAINER_DEVICE_INDEX)   # at the top of the thread
```

`torch.cuda.set_device` is thread-local too, so this must be done *in* the thread,
not just on the main thread before spawning it.

**The guard.** `NCCLWeightTransferDriver.initialize()` records the device it
rendezvoused on; `transfer()` compares it with the calling thread's and raises
`WeightTransferNotConfiguredError` naming both devices, the cause and the fix,
before any request is sent. The comparison is a pure function,
`device_mismatch(initialized_on, current)`, so it is tested with no GPU and no
torch, and `tests/test_nccl_driver.py::TestThreadDeviceRule` pins the thread rule.
The harnesses set the device at the top of both transfer threads.
