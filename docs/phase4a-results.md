# Phase 4A results — one version per rollout, under a live update

**Result: PASS. 12/12 checks.** An update was requested while a 256-token
generation was in flight, and RolloutCore **delayed the weight mutation until
that generation had finished**. The generation came out on the version it was
admitted to, and no rollout in this run ever saw two weight versions.

| | |
|---|---|
| Date | 2026-09-24 21:2x UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` (read back from `/version`) |
| RolloutCore | `b48ae00332de6b8badc2133a767cdbe07566c5e6` |
| Hardware | 2× NVIDIA RTX 3090, one pod |
| Model | `facebook/opt-125m`, TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Transfer group | `world_size=2` |
| Artifact | `results/phase4a.json` |

---

## Why this phase exists

Phase 3C proved the cycle works. This asks the question the cycle exists for:
**what happens to a rollout that is already generating when an update is
requested?**

vLLM cannot answer that. PR #49040 added a `weight_version` query API and
*deliberately removed* binding a version to a request — the engine hands you the
label and declines to enforce anything with it (RFC #48306 §2.2). So "one version
per rollout" is RolloutCore's guarantee or nobody's, and this is the run that
shows it holds against a live engine under real concurrency.

**The guarantee, stated precisely — and it is a claim about *delay*, not about
survival.** RolloutCore does not let a generation run across a mutation and hope
it comes out intact. It refuses to mutate while the generation is alive:

```
update requested → drain → (the in-flight rollout finishes) → mutate → invalidate → resume
```

A rollout that *did* span a mutation would be a different, weaker design (vLLM's
`mode="keep"` path, which RolloutCore deliberately cannot reach). The stronger
claim is that the mutation never lands under a live request at all, and that is
what the journal below shows.

## The setup, and why the text is the evidence

```
READY(rc-0) --admit R-long--> 256-token generation starts
   |                               |
   |                          0.4s | (it is genuinely mid-generation)
   v                               v
run_cycle(rc-1): DRAINING =====> /pause?mode=wait holds until R-long finishes
   R-long returns -> its own thread releases the rollout
   -> QUIESCED -> UPDATING (NCCL) -> INVALIDATING -> VALIDATING
   -> RESUMING -> READY(rc-1)     -> admit R-2 -> binds rc-1
```

The weight mutation happens in the `UPDATING` step, i.e. *after* R-long has
returned; the drain is what stands between the two. What the run therefore
measures is whether RolloutCore can hold an update off for as long as a real
generation takes, and whether the generation's output is consistent with the
version it was admitted to.

The server runs `--load-format dummy`, so `rc-0` output is degenerate. The update
installs `opt-125m`'s real weights, whose output for this prompt is coherent and
was measured in Phase 3A. The two are impossible to confuse, which makes the text
itself the evidence:

```
rc-0 baseline : '<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
in-flight     : '<s>' x 256          <- 256 tokens, all produced at rc-0
rc-1 after    : ' the capital of the French Republic.\n\n...'
```

`chars_matching_baseline_prefix: 48` out of `baseline_prefix_chars: 48` — the
whole 16-token baseline. A 256-token generation whose prefix reproduces the
pre-update baseline, token for token, is one that ran on the old weights
throughout. `ignore_eos` forces the 256 tokens exactly, so the request could not
finish early and quietly stop overlapping the cycle.

The in-flight record's own field says the same thing: `released_version: "rc-0"`.

## The journal is the mechanism

```
initialize, admit_rollout(R-long), begin_drain, finish_rollout,
confirm_drained, begin_update, confirm_updated, confirm_invalidated,
confirm_validated, confirm_resumed, admit_rollout(R-2), finish_rollout
```

`finish_rollout` lands **between** `begin_drain` and `confirm_drained`, and
`begin_update` comes after both. That is the ordering the invariant requires: the
engine's drain reports "quiescent" only once the request is done, RolloutCore must
already agree with that, and only then is the mutation allowed to start. Had the
release come after `confirm_drained`, the controller would have tainted
(`lifecycle.py:465-472`) — the engine claiming quiescence while RolloutCore still
counted live work.

The honest reading of this journal is worth stating, because an earlier version of
this document read it the other way: **the rollout did not span the mutation.** It
spanned the *drain*. The mutation was held off until it was gone.

The committed artifact's check `inflight_overlapped_update` predates this reading.
It asserts `cycle_started < request_ended`, i.e. that the request was still running
when the *cycle* began — the precondition for the experiment, not a claim about
the mutation. The harness now names it `inflight_overlapped_the_cycle`; as with
Phase 5's completion label, the artifact is left as measured and the rename is
recorded here.

## Measurements

| | |
|---|---|
| Cycle, with a rollout to wait for | **2.149 s** |
| — of which the drain held for R-long | 1.375 s |
| Drain polls | 2 (of 600; `drain_interval=2.0`, so ~0.2 s of margin) |
| In-flight request | 1.776 s, 1.375 s of it under the cycle (77%) |
| Phase 3C's identical cycle, nothing in flight | 0.673 s |

The contrast with Phase 3C is the point: the cycle costs 0.673 s when nobody is
generating and 2.149 s when someone is, and the difference is entirely the drain
doing its job rather than aborting the straggler.

## What this run required, and what it exposed

Three things had to be true for this to pass, and two of them were not true when
the harness was written:

1. **The rollout must be released from the thread that owns its request**, the
   instant that request returns — not after the cycle. The engine goes idle a few
   milliseconds *before* the client sees the response, so the release races the
   next drain poll. `drain_interval` is raised to 2.0 s so the release wins;
   `drain_polls: 2` shows that it did, with roughly 0.2 s to spare. This is a
   sharp edge, not a solved problem: at the 0.5 s default the margin would be
   ~0.05 s, and a loss taints. Confirm-on-second-observation would remove the race
   entirely and is the obvious fix if it ever bites.
2. **`LifecycleController` needed a lock.** This is the first phase to use it from
   two threads, which is its intended shape. Measured power: with the lock
   replaced by a no-op, 4 threads × 250 admit/release pairs produce 178 duplicate
   journal sequence numbers; with it, zero.
3. **The identity is still manifest-only** (`925369d663bc`, the same digest in
   Phase 3C for *both* weight sets). Nothing in this run depends on it — the
   version label and the text carry the claim — but it remains the reason the
   binding cannot yet mean "these exact weights".

## What Phase 4A does *not* prove

- **Not the cache lane.** This shows no rollout *saw* two versions; it says
  nothing about cross-version KV reuse. The cycle does invalidate
  (`INVALIDATING`, after the mutation), and a prefix-cache experiment
  (populate at `rc-0`, update, re-request) is a separate measurement — Phase 4C,
  which needed the prefix-cache hit metrics to be meaningful.
- **It does not show a rollout surviving a mutation**, because by design none
  does. The record type can represent that case (`spans_an_update`), and a
  synthetic record exercises it, but no GPU run here produced one: producing one
  would require the `mode="keep"` path RolloutCore forbids.
- **The delay was 1.375 s.** A generation long enough to outlast a configured
  drain timeout is a different scenario (Phase 4B's failed-drain case).
- **Two GPUs, one node, one prompt.** No multi-node, no batching pressure, no
  multimodal encoder lane.
- **Not a benchmark.** `opt-125m` at TP=1, one cycle, one rollout. The timings
  above characterise this run, not the design's throughput.
- **The rollout count is one.** The engine's drain is exercised with a single
  in-flight request; whether N concurrent rollouts behave the same is untested.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_concurrency_2gpu.py
```

Server output goes to `results/phase4a-server.log`.
