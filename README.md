# RolloutCore

**A versioned RL rollout control plane for vLLM.**

RolloutCore coordinates live policy updates around a running vLLM engine:

```text
rollout(vN)
    ↓
drain active work
    ↓
install weights vN+1 over NCCL
    ↓
invalidate stale caches
    ↓
validate while paused
    ↓
resume
    ↓
rollout(vN+1)
```

The goal is simple: **update an inference policy without restarting the engine, while preventing rollouts or KV state from silently crossing weight versions.**

RolloutCore does not patch vLLM. It builds a correctness layer around vLLM's existing lifecycle and weight-transfer primitives.

---

## Results

All headline results below come from real `vllm serve` processes and committed GPU artifacts in [`results/`](results/).

| Experiment | Result |
|---|---:|
| **Qwen3-8B hot weight transition** | **4.254 s** |
| Qwen3-8B full restart → ready | **37.045 s** |
| Restart / hot-update ratio | **8.71×** |
| OPT-125M hot weight transition | 0.230 s |
| OPT-125M restart → ready | 30.032 s |
| Cross-version cache negative control | **32 stale prefix-cache hits + different generated text** |
| Mid-update Qwen3-8B engine SIGKILL | target version not published; NCCL sender remained blocked **>120 s** |
| Test suite | **326 tests**, Python 3.11–3.13 CI |

### Qwen3-8B

On a single node with **2× RTX 3090 24 GB**, TP=1, trainer on one GPU and vLLM on the other:

```text
Hot update:        4.2544 s
Restart → ready:  37.0449 s

restart / hot update = 8.71×
```

The inference process is not restarted during the hot-update path.

At 8B, ~97% of the RolloutCore transition time is the actual weight-update phase, rather than controller overhead.

Hardware and benchmark details are recorded in [`docs/phase6-results.md`](docs/phase6-results.md).

> This is a model-transition benchmark, not an inference-throughput claim.
> The 8.71× ratio is specific to the measured single-node configuration.

---

## Why this exists

The design came from a source-level audit of vLLM `main` at:

```text
00b7847c8036b667742b4efb21aab1de51fd4721
```

Two gaps matter for strict RL rollout updates.

### 1. Weight versions are not bound to requests

vLLM exposes an opaque `weight_version`, but the audited version does not attach it to `Request` or `RequestOutput`.

A request may legitimately span versions in vLLM's asynchronous RL flow.

RolloutCore chooses a stricter contract:

> **A rollout admitted under version N must finish before weights can mutate to N+1.**

It enforces that structurally by draining active work before the update begins.

### 2. Weight updates do not make KV caches version-aware

In the audited vLLM version:

- `finish_weight_update` does not invalidate the prefix, encoder, or multimodal caches;
- the prefix-cache key contains no weight generation.

That means KV computed under old weights can be unsafe after a model update unless the caller establishes a cache-coherence boundary.

RolloutCore makes cache invalidation an explicit lifecycle state after the mutation and before serving resumes.

The source-level evidence is documented in [`docs/source-map-vllm-main.md`](docs/source-map-vllm-main.md).

---

## Architecture

```text
                           ┌──────────────┐
                           │    READY     │
                           └──────┬───────┘
                                  │ update requested
                                  ▼
                           ┌──────────────┐
                           │   DRAINING   │
                           └──────┬───────┘
                                  │ zero active rollouts
                                  ▼
                           ┌──────────────┐
                           │   QUIESCED   │
                           └──────┬───────┘
                                  │ preflight passed
                                  ▼
                           ┌──────────────┐
                           │   UPDATING   │
                           └──────┬───────┘
                                  │ NCCL install complete
                                  ▼
                         ┌──────────────────┐
                         │   INVALIDATING   │
                         └────────┬─────────┘
                                  │ caches reset
                                  ▼
                          ┌───────────────┐
                          │  VALIDATING   │
                          │ engine paused │
                          └───────┬───────┘
                                  │ target observed
                                  ▼
                           ┌──────────────┐
                           │   RESUMING   │
                           └──────┬───────┘
                                  │
                                  ▼
                               READY

Any ambiguous mutation failure
            ↓
         TAINTED
```

`TAINTED` is terminal in V1.

RolloutCore does not attempt to "roll back" a partially applied model update because the current vLLM interfaces do not provide enough evidence to prove the old weights were restored.

---

## Core invariants

RolloutCore is built around a small set of explicit invariants.

| Invariant | Meaning |
|---|---|
| **Version-pure rollout** | a rollout binds to one committed generation at admission |
| **Drain before mutation** | weight mutation cannot begin while an old-version rollout is active |
| **Post-update cache boundary** | KV/cache state from generation N is not served under N+1 |
| **Commit after validation** | the new generation becomes usable only after update, invalidation, validation and resume |
| **Single writer** | one RolloutCore instance owns lifecycle and weight mutations |
| **Fail closed** | ambiguous mutation failures do not silently return the engine to service |
| **Trajectory provenance** | records carry the generation and declared trainer/checkpoint provenance used for the rollout |

The complete state-machine contract is in [`docs/state-machine.md`](docs/state-machine.md).

---

## Real GPU experiments

### Live NCCL weight update

RolloutCore wraps vLLM's existing trainer-side weight-transfer stack rather than implementing NCCL itself.

The real path is:

```text
trainer model
    ↓
vLLM WeightTransferTrainerFactory
    ↓
NCCL broadcast
    ↓
running vLLM worker
    ↓
finish weight update
    ↓
cache invalidation
    ↓
validation
    ↓
resume
```

Phase 3C demonstrated a complete:

```text
dummy weights / rc-0
        ↓
real NCCL install
        ↓
real OPT-125M weights / rc-1
```

without restarting the vLLM process.

See [`docs/phase3c-results.md`](docs/phase3c-results.md).

---

### Update requested while a rollout is active

A 256-token rollout was already generating when an update was requested.

RolloutCore entered `DRAINING`, but the weight mutation did **not** begin until that rollout finished:

```text
admit rollout @ rc-0
        ↓
update requested
        ↓
DRAINING
        ↓
rollout finishes @ rc-0
        ↓
QUIESCED
        ↓
begin weight update to rc-1
```

Measured drain delay:

```text
1.375 s
```

The journal records `finish_rollout` before `begin_update`.

See [`docs/phase4a-results.md`](docs/phase4a-results.md).

---

### Cross-version KV reuse

A passing cache test is not useful unless reuse can be made to happen.

So Phase 4C includes two lanes.

#### Safe lane

RolloutCore performs:

```text
weights N
   ↓
update N+1
   ↓
INVALIDATING
   ↓
same prompt
```

The first post-update request records:

```text
old-version prefix-cache hits = 0
```

#### Negative control

The update is deliberately driven outside RolloutCore and cache invalidation is skipped.

Result:

```text
stale prefix-cache hits: 32
fresh-cache hits:         0

stale-cache continuation:
' Madrid Madrid Madrid Madrid ...'

fresh-cache continuation:
' a few few,,,,, and the the first...'
```

So old-weight KV was not merely retained in memory; it was actually reused and changed generation behavior.

That experiment is described in [`docs/phase4c-results.md`](docs/phase4c-results.md).

---

### Failure behavior

RolloutCore distinguishes safety from liveness.

In one Qwen3-8B experiment, the inference engine was killed approximately halfway through a multi-second NCCL update.

Observed:

```text
target version published:  no
engine alive:              no
trainer returned:          no, after >120 s
controller state:          UPDATING
```

The safety property held — the target version was never committed — but the trainer-side collective did not return within the experiment's 120-second budget.

That is a liveness failure outside the controller's ability to classify until control returns.

See [`docs/phase6-results.md`](docs/phase6-results.md).

---

## Weight versions and provenance

RolloutCore separates two concepts.

### `WeightVersion`

```text
rc-0
rc-1
rc-2
...
```

This represents **when** a model generation was published.

### `WeightIdentity`

This represents **what declared source** the update is intended to contain:

```text
checkpoint
trainer run
training step
parameter manifest
```

The distinction matters because two checkpoints of the same architecture have the same parameter names, shapes and dtypes.

A manifest digest alone therefore cannot distinguish:

```text
step 100
from
step 500
```

Trajectory records store both the version binding and declared provenance.

This is intentionally not presented as a byte-level proof of GPU memory contents.

See [`docs/phase5-results.md`](docs/phase5-results.md).

---

## Implementation

The core package is intentionally small.

```text
src/rolloutcore/
├── lifecycle.py        state machine and invariants; no I/O
├── runner.py           effect orchestration
├── evidence.py         typed observations/postconditions
├── versions.py         WeightVersion + WeightIdentity
├── trajectory.py       trajectory records + JSONL recorder
├── replay.py           token/logprob replay validator
├── port.py             LifecycleAdapter contract
├── weight_transfer.py  trainer-side transfer interface
└── adapters/
    ├── fake.py         reference fake adapter
    ├── http.py         real vLLM lifecycle control plane
    └── nccl.py         real trainer-side NCCL driver
```

The controller never performs external I/O directly.

Instead:

```text
adapter performs effect
        ↓
evidence object
        ↓
controller verifies postcondition
        ↓
state advances / waits / taints
```

This keeps lifecycle policy separate from vLLM-specific effects.

---

## Quick start

The state machine, fake adapter and replay tooling run without a GPU.

```bash
git clone https://github.com/awesome-pro/rolloutcore.git
cd rolloutcore

./scripts/test.sh
python3 -m rolloutcore.demo
```

Fast tests only:

```bash
./scripts/test.sh --fast
```

The core `src/rolloutcore` package uses only the Python standard library.

### Replay validator

```bash
python3 scripts/validate_replay.py \
  --trajectories results/phase5-trajectories.jsonl \
  --replays results/phase5-replay-synthetic.jsonl \
  --no-json
```

The committed Phase 5 records contain tokens but not real rollout logprobs, so the checked-in example exercises the token lane. Real-engine logprob replay remains future work.

---

## Reproducing the GPU work

GPU experiments live under [`diagnostics/`](diagnostics/).

Each harness:

- starts a real `vllm serve`;
- drives one controlled experiment;
- prints checks as they execute;
- writes a structured JSON artifact to [`results/`](results/).

Examples:

```bash
# real hot-weight cycle
python3 diagnostics/rolloutcore_cycle_2gpu.py

# rollout/update concurrency
python3 diagnostics/rolloutcore_concurrency_2gpu.py

# cache-coherence + negative control
python3 diagnostics/rolloutcore_cache_2gpu.py

# hot update vs restart benchmark
python3 diagnostics/rolloutcore_benchmark_2gpu.py
```

See [`diagnostics/README.md`](diagnostics/README.md) for environment requirements and the exact pod configuration used for the committed artifacts.

---

## Benchmark environment

The Qwen3-8B results were measured on:

```text
2 × NVIDIA RTX 3090 24 GB
single node
TP = 1
GPU 0 = vLLM inference
GPU 1 = bf16 trainer
PCIe
no NVLink
P2P disabled by topology
NCCL transport falling back to host/shared-memory path
```

The 8B server used:

```text
--enforce-eager
--gpu-memory-utilization 0.75
--packed-num-buffers 1
```

The benchmark compares:

```text
RolloutCore:
drain → NCCL weight install → cache invalidate → validate → resume

against

restart:
terminate vLLM → start with target checkpoint → /health ready
```

The checkpoint was already available locally for the restart baseline.

This benchmark therefore measures **model-transition downtime in this configuration**, not general inference throughput or a universal vLLM speedup.

---

## Testing

Current CI covers:

```text
Python 3.11
Python 3.12
Python 3.13
ruff
mypy --strict
fake-engine end-to-end demo
```

The repository contains **326 tests**, including the full state-transition matrix, invariant tests, adapter behavior, trajectory/provenance handling, drain/failure semantics and script hygiene.

The latest `main` CI is green.

---

## Limitations

RolloutCore V1 deliberately does not attempt to solve every RL serving problem.

Current scope:

```text
single vLLM engine
single node
NCCL backend
strict blocking updates
text-only cache experiments
TP=1 for committed GPU results
vLLM dev-mode lifecycle endpoints
```

Not yet demonstrated:

- asynchronous/overlapped RL where requests intentionally span updates;
- multi-node or multi-replica fan-out;
- tensor-parallel weight updates above TP=1;
- encoder/MM cache correctness with a multimodal checkpoint;
- sleep/wake lifecycle;
- IPC or `sharded_rdt` transfer backends;
- real-engine token/logprob replay validation;
- byte-level verification that declared trainer provenance exactly matches GPU memory.

Those are intentionally not hidden behind the current claims.

---

## Detailed evidence

The README is the summary. The experiment history is kept separately.

| Document | Purpose |
|---|---|
| [`docs/source-map-vllm-main.md`](docs/source-map-vllm-main.md) | source-level vLLM audit behind the design |
| [`docs/state-machine.md`](docs/state-machine.md) | lifecycle states, invariants and failure policy |
| [`docs/phase3a-results.md`](docs/phase3a-results.md) | real control-plane validation |
| [`docs/phase3c-results.md`](docs/phase3c-results.md) | first complete NCCL hot-update cycle |
| [`docs/phase4a-results.md`](docs/phase4a-results.md) | in-flight rollout + update ordering |
| [`docs/phase4b-results.md`](docs/phase4b-results.md) | failure injection |
| [`docs/phase4c-results.md`](docs/phase4c-results.md) | cache-coherence experiment + negative control |
| [`docs/phase4d-results.md`](docs/phase4d-results.md) | engine death during an update |
| [`docs/phase5-results.md`](docs/phase5-results.md) | trajectory provenance |
| [`docs/phase6-results.md`](docs/phase6-results.md) | Qwen3-8B benchmark + mid-collective failure |
| [`docs/replay-validator.md`](docs/replay-validator.md) | replay semantics and current limits |
| [`diagnostics/README.md`](diagnostics/README.md) | GPU harnesses and environment |

Raw experiment artifacts are committed under [`results/`](results/).

---

## vLLM provenance

The source audit and all `file:LINE` anchors were derived against:

```text
repository: vllm-project/vllm
commit:     00b7847c8036b667742b4efb21aab1de51fd4721
date:       2026-09-24
```

The audit can be rechecked against a local vLLM checkout:

```bash
python3 scripts/verify_anchors.py --vllm /path/to/vllm
```

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).