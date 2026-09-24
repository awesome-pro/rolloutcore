# The replay validator

**Item 8.** What it is: a pure-stdlib check that a recorded rollout can be
*reproduced* by the weight source it claims. What it is for: RL correctness rests
on the policy that produced a trajectory being the policy you think you are
training against, and an off-policy token is silently wrong data. This is the one
mechanism in the project that can catch that after the fact.

It runs on a laptop. `src/rolloutcore/replay.py` imports nothing but `math` and
`dataclasses`, so the comparison needs no engine, no GPU and no network:

```
python3 scripts/validate_replay.py --trajectories <jsonl> --replays <jsonl>
```

---

## 1. Why a replay, and what it can and cannot settle

RolloutCore binds every rollout to a declared weight source: a version label and
a `WeightIdentity` carrying provenance (`checkpoint`/`run_id`/`step`). That binding
is a *claim about provenance*. Nothing in the engine enforces it, because PR #49040
removed per-request version binding (one request may span two versions), so the
only way to test the claim is to reproduce the rollout and look.

The asymmetry is the point:

> **A mismatch is the strong direction.** It falsifies the claim outright.
> **Agreement only fails to falsify it.**

Nothing here proves that a checkpoint on disk is the checkpoint that ran. That
would need a content hash of the tensors, which `WeightIdentity` does not do: it
hashes names, dtypes and shapes (`src/rolloutcore/trajectory.py:159-163` records
why a manifest-only identity cannot even support the claim).

## 2. Two modes, and a mismatch means different things in each

This distinction is not cosmetic. Reading one mode's mismatch as the other's turns
a broken request into a false accusation against the weights, or hides a real
one.

| | `tokens_were_forced: true`: **trace-forced** | `tokens_were_forced: false`: **greedy** |
|---|---|---|
| How the ids are used | handed to the engine as a script | an expectation the weights must reproduce |
| What a token mismatch means | a fault in the replay (misconfigured request, diverged loop) | **the finding**: the weights did not reproduce their own argmax |
| How the validator reports it | an `echo:` **protocol violation**, and the verdict says it says nothing about the weights | `token_mismatches` / `first_divergence`, as an ordinary finding |

In both modes the position-by-position arithmetic is identical and is still
reported, so a diagnosis is possible: a trace-forced run that fails to echo also
tells you *where* it stopped echoing.

## 3. The record

A plan is built from a `Trajectory` (`ReplayPlan.from_trajectory`,
`src/rolloutcore/replay.py:98`) and is refused outright for a manifest-only
identity, because a replay of such a record could not be attributed to any
particular weight source:

```
ReplayError: trajectory 'R1' carries a manifest-only identity (925369d663bc),
which cannot separate two checkpoints of the same architecture, ...
```

An observation is one JSON object per line. Unknown keys are ignored, so a
hand-written file can annotate itself:

```json
{"request_id": "R2", "version": "rc-1", "token_ids": [5, 812, 9],
 "logprobs": [-0.12, -1.03, -2.20],
 "identity_digest": "sha256:f7a652cf4d70...", "prompt": "The capital of France is",
 "tokens_were_forced": true}
```

* `version`: what `GET /weight_info` reports **at replay time**, not at rollout time.
* `logprobs`: optional; `choices[0].logprobs.token_logprobs` from the completion
  response (`entrypoints/openai/completion/protocol.py:625`). `null` entries must be
  removed rather than dropped silently: a short tuple changes which positions the
  logprob lane covers, so the loader refuses `null` outright.
* `identity_digest` and `prompt`: optional. Absent means *unknown*, not *matching*:
  the validator only checks them when they are supplied.

## 4. The logprob lane

Token comparison catches a different argmax. It does not catch a distribution that
has drifted while the argmax held, which is the early stage of a weights mismatch
and invisible in the tokens alone. When both sides carry logprobs covering every
compared position, the validator reports:

| figure | meaning |
|---|---|
| `mean_abs_delta` | `mean abs(recorded[i] - observed[i])` |
| `max_abs_delta` | the worst position |
| `p99_abs_delta` | **nearest-rank**, `sorted[ceil(0.99n)-1]`; no interpolation, so the figure is always one of the measured deltas |
| `worst_index` | argmax, first occurrence |

The default tolerance is `1e-3` (`src/rolloutcore/replay.py:58`). It is not a
precision claim: bf16 kernels are not bit-reproducible across batch shapes, and
`VLLM_BATCH_INVARIANT=1`, which needs SM90 or newer
(`examples/rl/rlhf_async_new_apis.py:177`), is the only way to make an exact
comparison meaningful. Below the tolerance the deltas are noise.

## 5. Producing replays on the GPU side

### Option 1: in-process, trace-forced (the strong mode)

vLLM can be told exactly which ids to emit. `trace_decode_token_ids`
(`sampling_params.py:374`) "forces the engine to emit this predetermined sequence
of token IDs during decoding instead of sampling randomly. Real logprobs are
still computed" (`sampling_params.py:375-377`), it requires the engine to be
started with `--enable-trace-replay` (`config/model.py:276`) or the request is
rejected (`v1/engine/input_processor.py:169-176`), and the sampler overwrites the
sampled token while computing logprobs from the unmodified logit distribution
(`v1/worker/gpu/sample/trace_replay.py:14`).

```python
llm = LLM(model=..., enable_trace_replay=True)   # plus the usual flags
params = SamplingParams(
    max_tokens=len(record.token_ids),
    trace_decode_token_ids=list(record.token_ids),
    logprobs=1,
    temperature=0.0,
)
```

Write `"tokens_were_forced": true` for these observations.

**This is not reachable over HTTP today.** `vllm/entrypoints/` contains no
reference to `trace_decode_token_ids`, so `POST /v1/completions` cannot force a
sequence; the flag is engine-level only. That is a finding in its own right, and
it is why option 2 exists.

### Option 2: over HTTP, greedy (the weaker mode)

Re-request with `temperature=0`, `logprobs=1`, and the recorded text appended to
the prompt. Only the continuation is compared, and nothing is forced, so a
mismatch is the finding. Leave `tokens_were_forced` at its default `false`.

## 6. Worked example: the committed Phase 5 record

`results/phase5-trajectories.jsonl` holds two declared-source records with **no
logprobs**, because the Phase 5 harness asked only for `return_token_ids`. So the
committed artifact exercises the **token lane only**, and the logprob lane has
never been run against a real engine. Say so when quoting a number from it.

`results/phase5-replay-synthetic.jsonl` is a **hand-written** replay of those two
records: no engine ran, no GPU was involved, and it exists so the command below is
runnable as written. It deliberately leaves `tokens_were_forced` false, because
nothing forced anything.

```console
$ python3 scripts/validate_replay.py \
    --trajectories results/phase5-trajectories.jsonl \
    --replays results/phase5-replay-synthetic.jsonl \
    --json-out /tmp/verdicts.json
R-long AGREE: shared prefix 256, 0 token mismatch(es), token-only (both-missing)
    note: logprobs were not compared (both-missing); the verdict is token-only
R2 AGREE: shared prefix 16, 0 token mismatch(es), token-only (both-missing)
    note: logprobs were not compared (both-missing); the verdict is token-only

2/2 replay(ies) agreed over 2 trajectory record(s); 0 trace-forced / 2 greedy, lanes {'both-missing': 2}, 0 token mismatch(es) in total
PASS: every record replayed and agreed
[report] /tmp/verdicts.json
```

Change one id and the same command exits `1`, naming the position:

```console
$ python3 scripts/validate_replay.py \
    --trajectories results/phase5-trajectories.jsonl \
    --replays /tmp/phase5-replay-disagrees.jsonl --no-json
R-long AGREE: shared prefix 256, 0 token mismatch(es), token-only (both-missing)
    note: logprobs were not compared (both-missing); the verdict is token-only
R2 DISAGREE: shared prefix 3, 1 token mismatch(es), first divergence at 3, token-only (both-missing)
    note: greedy replay: the tokens were expected, not forced, so the 1 mismatch(es) are the finding -- the weights did not reproduce their own argmax
    note: logprobs were not compared (both-missing); the verdict is token-only

1/2 replay(ies) agreed over 2 trajectory record(s); 0 trace-forced / 2 greedy, lanes {'both-missing': 2}, 1 token mismatch(es) in total
FAIL: 1 problem(s): R2: disagrees
```

Both summaries above are copied from real runs, not paraphrased.

To see the logprob lane and the `echo:` violation end to end, hand-write a
two-line pair with logprobs and `"tokens_were_forced": true`; no engine is
required, which is why the validator is a pure function.

## 7. What it does not prove

1. **It does not prove which weights ran.** Only that the tokens are consistent
   with the declared source. A content hash of the tensors would be needed for
   more, and that is out of scope for `WeightIdentity`.
2. **The logprob lane has no real-engine evidence yet.** Phase 5 recorded no
   logprobs, so every figure in that lane is currently exercised only by unit
   tests and hand-written vectors.
3. **The traced path is engine-level only.** Until `trace_decode_token_ids` is
   exposed by an entrypoint, a trace-forced replay cannot be produced against
   `vllm serve`; option 2 is a strictly weaker substitute.
4. **Non-determinism is absorbed, not measured.** The `1e-3` tolerance treats
   small logprob drift as noise. It cannot separate kernel non-determinism from a
   tiny weight difference, and without `VLLM_BATCH_INVARIANT=1` on SM90+ it never
   will.
5. **One replay is one sample.** Agreement on a handful of records is evidence
   about those records, not a guarantee for a training run.
