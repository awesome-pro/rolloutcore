# Phase 3C results — the first real RolloutCore cycle

**Result: PASS. 5/5 checks**, on two real GPUs, with a real NCCL weight transfer,
driven end to end by `LifecycleRunner.run_cycle()`.

| | |
|---|---|
| Date | 2026-09-24 15:43 UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` (exact match, read back from `/version`) |
| RolloutCore | `9967c2132390dd70160bc006d7024e6cd7eda470` |
| Hardware | 2× NVIDIA RTX 3090, one pod (`root@213.192.2.117`, port 40107) |
| Driver / CUDA | 595.71.05 / 13.2 |
| NCCL | 2.29.7 (`pynccl.py:148`) |
| Model | `facebook/opt-125m` |
| Topology | vLLM TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Transfer group | `world_size=2` (1 inference worker + 1 trainer), rank 0 = trainer |
| Artifact | `results/phase3c.json` |

---

## What was exercised

The function Phase 3A deliberately never called, with nothing faked:

```
READY(rc-0) --dummy weights--> generate gibberish
  -> DRAINING -> QUIESCED -> UPDATING (real NCCL broadcast)
  -> INVALIDATING -> VALIDATING -> RESUMING -> READY(rc-1)
  -> generate -> coherent, same vLLM pid
```

```
=== Phase 3C ===
  [PASS] committed_rc1            controller committed rc-1 only after /resume
  [PASS] engine_label_rc1         engine reports rc-1 -- the version handshake
  [PASS] engine_resumed           /is_paused is false afterwards
  [PASS] output_changed           dummy gibberish -> real weights
  [PASS] server_never_restarted   same process (pid 6778) throughout
  before: '<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
  after : ' the capital of the French Republic.\n\nThe capital of France is the capital'
  states: READY -> DRAINING -> QUIESCED -> UPDATING -> INVALIDATING -> VALIDATING -> RESUMING -> READY
```

Two checks carry the weight:

- **`committed_rc1`** — the commit point is the resume, not the finalize. The
  controller does not advance its own version until `/resume` is acknowledged.
- **`engine_label_rc1`** — the engine reports `rc-1` *while still paused*. This
  only happens because `RolloutCoreWeightSyncClient` supplies the version
  upstream omits: `NCCLTrainerWeightTransferEngine.send_weights()` calls
  `finish_weight_update()` with no argument (`nccl_engine.py:361`) although the
  client accepts one (`clients.py:89`). Without the wrapper, VALIDATING would
  taint on a *successful* transfer.

Both generations ran on the **same server process**; the trainer was a second
process on the other GPU. Nothing was restarted, and no weight update was faked
to reach the end of the state machine.

## The strongest single piece of evidence

The `after` text is identical, token for token, to Phase 3A's generation — and in
Phase 3A nothing was transferred at all:

> `' the capital of the French Republic.\n\nThe capital of France is the capital'`

Phase 3A's server loaded these weights through vLLM's own loader; Phase 3C's
server started from garbage and received them over NCCL. Same text, same prompt,
same `temperature=0`. So the bytes that arrived over the wire are the bytes the
loader would have produced — which is what "the transfer works" has to mean.

The `before` output is the control: 16 consecutive BOS tokens (id 2), which is
what `--load-format dummy` predicts for a model whose embeddings are uninitialized.

## The limitation this run exposes

`[identity] 925369d663bc (manifest-only)` — and the *same* digest appears as the
target for `rc-1`:

```
[cycle] run_cycle -> rc-1 (925369d663bc (manifest-only))
```

`WeightIdentity` is derived from `WeightSource.metadata()`: parameter names,
dtypes and shapes. For `opt-125m` that manifest is identical for *any* weights of
that architecture — the dummy ones and the real ones included. So the identity
proves structure, not values; only the engine's opaque `rc-*` label distinguishes
the two versions here.

This is exactly the point the pre-GPU review raised (use *declared* provenance,
never claim "exact weights"), and Phase 3C is the run that makes it concrete
rather than theoretical. It is also why the next honest step is either a content
digest or a provenance-carried identity from the trainer's checkpoint/step, and
not a stronger claim about the current digest.

## What Phase 3C does *not* prove

- **Nothing about values.** See above: two different weight sets share one digest.
- **Nothing about concurrent rollouts.** One request was in flight at a time. The
  "one version per rollout" and "no cross-version cache reuse" guarantees still
  have no engine-side support and no live test.
- **Not a benchmark.** 30.0 s to ready, one cycle, `opt-125m`, no timing claim
  about drain latency or transfer size beyond what `results/phase3c.json` records.
- **Single-node only.** Both processes are on one host; the rendezvous used
  `get_ip()` + `get_open_port()`. Multi-node is untested.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3          # a Ctrl-C'd run can leave one on :8000
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_cycle_2gpu.py
```

Server output goes to `results/phase3c-server.log`; the harness prints its own
progress. Ready in ~30 s on a warm HF cache.
