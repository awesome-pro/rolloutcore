# Phase 4C results: no cross-version cache reuse, measured and then made to happen

**Result: PASS. 8/8 checks.** Two phases against real vLLM: the guarantee
asserted through the public prefix-cache metrics, and then a negative control
that *deliberately breaks it* to show the reset step is load-bearing rather than
belt-and-braces.

| | |
|---|---|
| Date | 2026-09-24 16:34 UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| RolloutCore | `78941f5b` (see `results/phase4c.json`) |
| Hardware | 2× NVIDIA RTX 3090, one pod |
| Model | `facebook/opt-125m`, TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Artifact | `results/phase4c.json` |

This is invariant I3 (*KV created under generation N is never consumed under
N+1*), and it is the half of the project that vLLM has no mechanism for. The
prefix cache maps token blocks to KV blocks and knows nothing about which weights
produced them, `finish_weight_update` invalidates no cache
(`gpu_worker.py:1488-1505`), and `reset_prefix_cache` is called only by explicit
engine APIs (`llm_engine.py:361`, `async_llm.py:1089`, `core_client.py:398`), never
by the weight-update path. So the reset is the caller's job, and this phase is
where that job is either discharged or shown to be theatre.

**One amendment postdates this run.** The cycle now pauses with
`clear_cache=false`, so `INVALIDATING` is the only place caches are dropped
(`docs/state-machine.md`, "The invalidation boundary"). The run below was made
when the pause also cleared, which is the ambiguity that amendment removes:
Phase 1's zero-hit result cannot by itself separate the pause's clear
from the reset's, and Phase 2 is what pins the reset, because it runs with
`clear_cache=false` and no reset at all.

---

## Phase 1: the guarantee, and it gates

Prefix caching is on by default (`vllm/config/cache.py:142`), so the lane is
live without any flag. The sequence is: send prompt `P`, send `P` again to prove
the cache works, run a full cycle to `rc-1`, send `P` a third time.

```
pre_update   cold text      '<s><s>…'  (16 tokens)
             hits after 1st   0.0
             hits after 2nd   32.0     ← the cache is real and measurable
             queries          68.0
post_update  first text     ' Lisbon. The capital of the United Kingdom is…'
             first hit delta   0.0     ← every token was recomputed
             second hit delta 32.0     ← and caching still works afterwards
             weight_version  rc-1
```

The first post-update request must score **zero** new hits: every one of its
tokens was computed under `rc-0`, so a hit would mean a trajectory assembled from
two weight sets. The second request must score hits again; otherwise "no reuse"
is indistinguishable from "caching silently switched off", which is the failure
mode that a naive zero-delta assertion cannot tell apart. Both hold, and the
counts come from vLLM's own `vllm:prefix_cache_hits_total` /
`vllm:prefix_cache_queries_total`, not from RolloutCore's bookkeeping.

The cycle itself took 2.1404 s and visited all eight states.

## Phase 2: the hazard, made to happen on purpose

A passing assertion that "no reuse occurred" is only as good as the claim that
reuse *could* have occurred. So the second phase removes RolloutCore from the
loop and drives the update out of band: `/pause?mode=wait&clear_cache=false`,
broadcast, `/resume`. No controller, no reset, no cache invalidation. The
trainer's first decoder layer is negated in place, so `rc-2` is a checkpoint that
is byte-different and semantically broken.

```
pause_cleared_cache          false
reset_skipped                true
engine_version               rc-2
hits_with_stale_cache        32.0      ← the old KV was handed straight back
hits_with_fresh_cache         0.0      ← after an explicit reset, none of it
text_with_stale_cache        ' Madrid Madrid Madrid Madrid Madrid Madrid - - - - - - - - - -'
text_with_fresh_cache        ' a few few,,,,, and the the first.,,,'
texts_differ                 true
kv_reused_across_weights     true
```

Three things follow, and they are the finding:

1. **The reuse happens.** The engine does not merely keep stale KV around; it
   serves it. 32 hits against a cache built by different weights.
2. **The reuse changes the output.** The two continuations differ, which is what
   makes this a correctness hazard rather than a memory-efficiency question. The
   tokens sampled from a prompt prefilled by `rc-0` weights are not the tokens the
   `rc-2` weights would have produced.
3. **The reset is sufficient.** The fresh-cache run scores `0.0` hits on the same
   prompt and the same weights, so the 32 hits in the stale run are exactly what
   RolloutCore's `INVALIDATING` step removes. The step is load-bearing.

The verdict string the harness writes is deliberately blunt: *"REUSE HAPPENED: KV
computed before the weight change was handed back after it. RolloutCore's reset
step is load-bearing."*

## Finding: the corrupted checkpoint kept the manifest digest

`corrupted_weights_share_the_manifest_digest` passes, and it is the cleanest
statement of the blind spot item 5 exists to fix. The negated checkpoint, the
real `opt-125m`, and the dummy-initialised `opt-125m` all report
`925369d663bc`: the digest covers parameter names, dtypes and shapes, and a sign
flip changes none of them.

Phase 3C found the same digest for dummy and real weights. That was enough to
show `WeightIdentity.parse` cannot separate two checkpoints of one architecture;
this run shows it again for a *corrupted* one, which is the case that would
actually hurt: a bad update would be indistinguishable from a good one by digest
alone. Declared provenance (checkpoint/run/step, Phase 5) is the answer: the same
manifest yields two different identities once a step is declared
(`c8c0aafd93a5` for step 0, `f7a652cf4d70` for step 1), and
`Trajectory.replay_ready` is what refuses to pretend otherwise.

## What Phase 4C does *not* prove

- **No encoder/multimodal lane.** `/reset_encoder_cache` and `/reset_mm_cache`
  are no-ops on a text-only model: Phase 3A measured that all three cache resets
  return success while two of them have nothing to clear. A multimodal checkpoint
  is the only way to exercise the encoder lane, and that is deferred.
- **`opt-125m`, one node, TP=1, one request at a time.** No interaction with
  preemption, eviction, or a cache under pressure.
- **The hazard phase is out of band by construction.** It shows what happens when
  the reset is skipped; it does not show a controller path that skips it, because
  there is not one.
- **The metric is token-based.** `vllm:prefix_cache_queries` counts tokens, not
  blocks or requests, so the absolute numbers are prompt-token counts and should
  not be read as block counts.
- **Nothing here verifies weight bytes.** The provenance for `rc-2` is declared by
  the harness, exactly as in Phase 5.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_cache_2gpu.py
```
