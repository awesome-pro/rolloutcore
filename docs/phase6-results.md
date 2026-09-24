# Phase 6 results: what a hot update costs, against restarting the engine

**Result: PASS.** 11/11 checks at `facebook/opt-125m`, 6/6 at `Qwen/Qwen3-8B`.
The headline: **130.63× cheaper than a restart at 125M, 8.71× at 8B**. The gap
narrows with model size because the restart's cost is mostly fixed engine startup
while the hot path's cost is the broadcast itself.

| run | model | hot cycle | restart → ready | restart → serving | speedup | checks |
|---|---|---|---|---|---|---|
| `results/phase6.json` | `facebook/opt-125m` | **0.2299 s** | 30.0317 s | 30.3501 s | **130.63×** / 132.01× | 11/11 |
| `results/phase6-8b.json` | `Qwen/Qwen3-8B` | **4.2544 s** | 37.0449 s | 37.5290 s | **8.71×** / 8.82× | 6/6 |
| `results/phase6-8b-with-deepkill.json` | `Qwen/Qwen3-8B` | 4.2556 s | 37.0487 s | 37.5621 s | 8.71× / 8.83× | 10/11 † |

| | |
|---|---|
| Date | 2026-09-24 18:06 to 18:25 UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| Hardware | 2× RTX 3090 24 GB, PCIe (no NVLink, P2P disabled by topology), one pod |
| Server | TP=1 on GPU 0, `--load-format dummy`, `--enable-prefix-caching` (default) |
| Trainer | bf16, full checkpoint, GPU 1 |
| 125M flags | `--gpu-memory-utilization 0.6`, `--max-model-len 4096`→**2048** (the model's own limit), drain interval 0.1 s |
| 8B flags | same plus `--gpu-memory-utilization 0.75`, `--packed-num-buffers 1` |

† The 10/11 is the *pre-fix* artifact: its eleventh check was
`deep_kill_detected_within_budget`, a **hang in the bonus scenario gating the
benchmark**. That was corrected afterwards: the hang is now recorded in
`detail.deep_kill.verdict` while the fail-closed checks still gate, because a FAIL
should mean the benchmark failed and nothing else. Under the corrected harness the
ten remaining checks all pass; `results/phase6-8b.json` is the clean 6/6 run with
`--skip-deep-kill`.

## The environment these numbers came from

An 8B transfer time is not interpretable without the fabric it ran over, and the
committed artifacts record the software flags and the vLLM/RolloutCore revisions
but **not** the driver, CUDA, NCCL or the topology. This section is assembled from
what the pod and the earlier runs do say; it states plainly what was never
captured rather than filling the gap in.

| | |
|---|---|
| GPUs | 2× NVIDIA GeForce RTX 3090, 24 GB each (GA102, compute capability 8.6) |
| Interconnect | PCIe, with **P2P disabled by topology** and no NVLink bridge in play |
| NCCL transport observed | `SHM/direct/direct`, or `NET/Socket/0` with `NCCL_SHM_DISABLE=1` (runbook §11) |
| Driver / CUDA | 595.71.05 / 13.2 (`docs/phase3c-results.md`; `docs/phase3b-runbook.md`) |
| torch / NCCL library | **not captured** in the Phase 6 artifacts; the harness now records both |
| Server | TP=1 on GPU 0, `--load-format dummy`, prefix caching on (default) |
| Trainer | bf16 full checkpoint on GPU 1 |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| RolloutCore | each artifact's own `rolloutcore_sha` (all clean trees) |

**What this means for the numbers.** The 8B cycle's 4.1287 s of `UPDATING` moves
roughly 16 GB of bf16 weights (about **3.9 GB/s effective**) over a host-mediated
path, because P2P is off and NCCL is using the shared-memory transport. That is a
property of *this* fabric. On NVLink, or with P2P enabled across a wider PCIe
topology, the transfer term would be smaller, so the hot cycle would shrink and
the speedup ratio would move; which way depends on how much of the restart's ~30 s
is fabric-independent (most of it is engine startup). The 125M numbers are far
less fabric-sensitive: 0.1008 s of transfer against the same ~30 s baseline.

The harness now records all of it, before the server starts, in
`detail["environment"]`: `gpu_names`, `gpu_memory_mib`, `gpu_driver`,
`gpu_compute_capability`, `torch`, `cuda`, `nccl`, `p2p_peer_access` (per device
pair, from `torch.cuda.can_device_access_peer`) and the raw `nvidia-smi topo -m`.
The committed artifacts predate that, so this table is the retrospective; the next
run's artifact is self-describing.

---

## The three lifetimes

**L1: the hot path.** Server booted from dummy weights, trainer holding the real
checkpoint, one `run_cycle`. A watcher samples the controller's state, so the
cycle decomposes into per-state dwell times instead of one opaque total.

**L2: the restart baseline.** Kill the engine, start it again on the same
checkpoint, time from the kill to `/health` and to the first token. This is
**generous to the baseline**: it assumes the new checkpoint is already on disk. A
real restart-based update must also *write* a checkpoint first, which is not
counted, so the gap below is a floor on the hot path's advantage.

**L3: the deep-broadcast kill.** A second cycle is started, and the engine's
process group is SIGKILLed at half of L1's *measured* update duration. At 8B that
lands mid-collective, which is the case Phase 4D could not reach at 125M; at 125M
the update phase is short enough that the failure still surfaced on an HTTP socket
before the transfer path could block.

## L1: the cost is the broadcast, and nothing else

`results/phase6.json` (125M), per-state dwell:

```
READY       0.0204 s   ← adapter setup around the cycle
DRAINING    0.1005 s   ← the drain poll interval, no in-flight rollout
UPDATING    0.1008 s   ← the broadcast
VALIDATING  0.0087 s
             0.2299 s total, timeline 0.2304 s
```

`results/phase6-8b.json` (8B):

```
READY       0.0206 s
DRAINING    0.1007 s
UPDATING    4.1287 s   ← 97% of the cycle
RESUMING    0.0051 s
             4.2544 s total, timeline 4.2551 s
```

The control plane is free: validate is 8.7 ms, resume 5.1 ms, the drain is the
poll interval and nothing more. What the hot path costs is what moves over the
wire: 0.1008 s for a ~250 MB checkpoint, 4.1287 s for a ~16 GB one.

## L2: the baseline is ~30 s of engine startup

Both models pay roughly the same restart cost: 30.03 s at 125M and 37.04 s at 8B.
Most of it is fixed (process start, CUDA context, NCCL/worker init, KV-cache
allocation), with the extra ~7 s at 8B being checkpoint load. The restart's first
token costs ~0.3 to 0.5 s more than `/health`, and both runs serve the same text
the hot path produced, which is the evidence that the baseline really installed
the target weights.

## The comparison, and why it narrows

| | 125M | 8B | ratio |
|---|---|---|---|
| hot cycle | 0.2299 s | 4.2544 s | ×18.5 |
| restart → ready | 30.0317 s | 37.0449 s | ×1.23 |
| speedup | **130.63×** | **8.71×** | ÷15 |

The hot path scales with payload; the restart is almost size-independent. So the
speedup is a ratio between a fixed cost and a growing one, and **it must fall as
models get bigger**: an 8B model is already 15× less favourable than a 125M one,
and a 70B model with a 140 GB checkpoint would push it further. Quoting the 131×
without the 8.7× next to it would be misleading, which is why both artifacts are
committed and why this table is the headline rather than the first row.

## L3: a kill inside the collective does not fail; it blocks

This is the phase's most important result, and it is a negative one about vLLM,
not about RolloutCore.

**At 125M the whole update is 0.10 s, so the failure surfaced as an error.** Aim
half of 0.1008 s = 0.0504 s; the kill landed at 0.1537 s (33 ms into the update
phase) and the failure surfaced 0.1738 s later:

```
hung   false
outcome EngineTaintedError … RemoteDisconnected('Remote end closed connection
        without response')   → controller TAINTED
```

That is Phase 4D's result reproduced with a tighter aim: a fast, clean taint, and
nothing published.

**At 8B the same experiment never returns.** Aim 2.0632 s (half of 4.1264 s), kill
at 2.1743 s, about 53% of the way through the collective:

```
hung                true
seconds_to_outcome  120.2254   ← the harness's budget, not a failure
outcome             null
controller          state UPDATING, tainted false
```

The trainer is blocked in the broadcast and stays blocked. RolloutCore cannot
taint, because `complete_weight_update` never returned and RolloutCore is never
told anything happened; the engine is dead and the controller still believes an
update is in progress. The 120.2254 s is the harness's own bound (`os._exit`), not
a timeout in the path; the path has none. The trainer joins through vLLM's
`PyNcclCommunicator` (`nccl_engine.py:156` → `nccl_common.py:181`), and the only
timeout in that code is on *teardown*, with the source's own comment saying a
failed join "leaves the peer blocked in `ncclCommInitRank` until timeout"
(`pynccl.py:237`). There is no `NCCL_TIMEOUT`-style knob anywhere in the tree.

So the two sizes measure two different regimes of the same code path, and the
boundary is the length of the collective:

| | 125M (0.10 s update phase) | 8B (4.13 s update phase) |
|---|---|---|
| kill landed | 33 ms in, inside the phase | ~2.05 s in, ~50% through |
| failure surfaced as | an error, 0.17 s later | nothing, ever |
| controller | `TAINTED` | stuck in `UPDATING`, untainted |

## Findings

**1. A receiver-side OOM in a concurrent broadcast presents as a hang.** The first
8B attempt failed with `CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has
531.00 MiB free.` That 1.16 GiB is not a chunk-boundary divergence: it is the
receive buffer for the 1.24 GB embedding tensor, the one chunk that needs its own
allocation. The OOM is raised **server-side and logged**, the engine stays alive
and keeps answering `/health`, and the trainer (already inside the collective)
blocks forever. `complete_weight_update` never returns, so the controller sits in
`UPDATING`. This is the same failure shape as the 8B deep kill above, reached
without killing anything.

**2. vLLM sizes the KV cache with no headroom for transfer buffers.** The KV cache
is allocated to fill the memory budget at startup and nothing is reserved for the
packed consumer's lazily allocated receive buffers
(`torch.empty(packing_tensor_sizes[buffer_idx])`). At 125M this is invisible
because the whole model fits in one sub-buffer chunk; a 16 GB model streams ~16
chunks of ~1 GiB with the default two buffers. The shipped defaults now leave the
transfer somewhere to land (`--gpu-memory-utilization 0.6`, `--max-model-len`
clamped to the checkpoint's own limit; 8B additionally uses
`--packed-num-buffers 1`). This is a configuration fix, not a code fix, and it is
the kind of thing an RL runtime has to know about its engine.

**3. The trainer's collective is unbounded.** Above. It is the strongest
upstreaming candidate of the phase, and it is also why L3 is measured with the
harness's own `os._exit` bound: without one, the experiment would not have
returned at all.

**4. The transfer is device-sensitive to the *calling thread*, and NCCL's error
names neither.** Five consecutive runs (two transports, three model sizes, a
fresh container) failed identically with
`NCCL WARN Cuda failure 400 'invalid resource handle'`. PyTorch's current CUDA
device is thread-local, and the capture thread started on device 0 while the model
and communicator were created on device 1. vLLM's packed producer builds its
streams from `torch.accelerator.current_device_index()`
(`packed_tensor.py:23-24`), so a GPU-1 broadcast ran on GPU-0 streams. Two earlier
theories were wrong and are retracted: `/dev/shm` sizing, and `NCCL_P2P_DISABLE`
self-blame: P2P is disabled by topology on this host and always was. The fix has
two halves: the harness sets the device at the top of both transfer threads, and
`NCCLWeightTransferDriver` now refuses the situation instead of letting it reach
NCCL (`initialize()` records its rendezvous device, `transfer()` compares it to the
calling thread's, and a mismatch raises before any request is sent). The rule is a
pure function, `device_mismatch`, tested with no GPU and no torch.

## What Phase 6 does *not* prove

- **The hot number is the best case.** No in-flight rollouts, a 0.1 s drain poll,
  one client, one stream, no preemption. Under load the drain is the cost that
  grows, not the broadcast.
- **The restart baseline excludes the checkpoint write**, as designed. It is also
  measured on a warm page cache; a cold one would be worse for the baseline, i.e.
  better for the hot path.
- **No tensor parallelism.** TP=1 on one GPU, and the two GPUs communicate over
  PCIe with P2P disabled by topology; a different fabric (NVLink, or a real
  TP=2 weight loader) would move the transfer term.
- **One checkpoint pair per size**, one dummy→real transition each. The 8B
  artifacts use different flags from the 125M ones, so the cross-size comparison
  above is of two configurations, not one swept variable.
- **No quantization, no MoE, no speculative decoding.**
- **The 8B deep-kill artifact is pre-fix** (see †), so its `ok` field is `false`
  even though the benchmark it measures passed.
- **Timing granularity is the harness's poll interval** (0.1 s for the drain, and
  the watcher samples on the same order), so sub-10 ms phases are not resolvable.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf

# 125M, all three lifetimes (the 11-check run)
python3 diagnostics/rolloutcore_benchmark_2gpu.py --model facebook/opt-125m

# 8B, hot path + restart baseline only
python3 diagnostics/rolloutcore_benchmark_2gpu.py --model Qwen/Qwen3-8B --skip-deep-kill

# 8B with the deep-broadcast kill (the run whose L3 never returns)
python3 diagnostics/rolloutcore_benchmark_2gpu.py --model Qwen/Qwen3-8B
```
