# Phase 5 results — trajectories that name the training step

**Result: PASS. 16/16 checks.** Roadmap item 5, on real hardware: a rollout that
spans a weight update is recorded against the version it was **admitted** to, with
a declared provenance that separates two checkpoints the manifest digest cannot.

| | |
|---|---|
| Date | 2026-09-24 16:45 UTC |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` |
| RolloutCore | `3d238777` (see `results/phase5.json`) |
| Hardware | 2× NVIDIA RTX 3090, one pod |
| Model | `facebook/opt-125m`, TP=1 on GPU 0 (`--load-format dummy`), bf16 trainer on GPU 1 |
| Artifacts | `results/phase5.json`, `results/phase5-trajectories.jsonl` |

---

## The gap this closes

Two earlier phases measured the same thing from different angles:

- **Phase 3C** — a dummy-initialised `opt-125m` and the real one share the
  manifest digest `925369d663bc`. The digest covers parameter *names, dtypes and
  shapes*, so it identifies an architecture, not a training step.
- **Phase 4C** — a checkpoint with its first decoder layer negated shares that
  digest too. The blind spot survives a corrupted checkpoint, which is the case
  that would actually hurt: a bad update would look identical to a good one.
- **Phase 4A** — a rollout admitted at `rc-0` finished *after* the update to
  `rc-1`, and the engine then reports `rc-1` for a request whose tokens came
  entirely from `rc-0`. A trajectory that recorded the engine's last word would
  be wrong about which weights produced it.

PR #49040 made the second case structural: it added the `weight_version` query API
and deliberately removed binding a version to `Request`/`RequestOutput`, because
one request may span two versions. So the record a trainer needs has to live
outside the engine. That is `rolloutcore.trajectory`, and the enabling piece on
the trainer side is `NCCLWeightTransferDriver.declare` — a driver whose cached
identity cannot follow weights that changed has no way to *re*-declare what it is
about to install.

## The experiment

Phase 4A's shape, with provenance added. `R-long` is admitted at `rc-0` and
generates 256 tokens across the cycle to `rc-1`; `R2` is admitted afterwards at
`rc-1`.

| | `R-long` | `R2` |
|---|---|---|
| bound version | `rc-0` | `rc-1` |
| declared provenance | `facebook/opt-125m@dummy-init`, run `phase5-declared`, step **0** | `facebook/opt-125m`, run `phase5-declared`, step **1** |
| identity digest | `c8c0aafd93a5…` | `f7a652cf4d70…` |
| manifest digest | `925369d663bc…` | `925369d663bc…` (the same) |
| tensors | 196 | 196 |
| `exactness` | `declared-source` | `declared-source` |
| `replay_ready` | `true` | `true` |
| engine at admission | `rc-0` | `rc-1` |
| engine at completion | **`rc-1`** | `rc-1` |
| output | 256 × `<s>` (token id `0`) | `' the capital of the French Republic.…'` |

Three of those rows are the result:

1. **The two records have different identities and the same manifest.** The same
   `925369d663bc` under both is not a defect in the record; it is why provenance
   exists. `c8c0aafd93a5` vs `f7a652cf4d70` is the difference a step number makes.
2. **The engine's last word is not the binding.** `R-long`'s
   `engine_version_at_completion` is `rc-1` while its `version` is `rc-0`, and the
   record keeps `rc-0`. The check `inflight_engine_reported_a_different_version_at_completion`
   asserts that divergence on purpose — it is the Phase 4A case, now recorded.
3. **The tokens are the step-0 weights.** All 256 token ids are `0` (`<s>`), which
   is what dummy-initialised weights produce, and `R2`'s are real text. So the
   binding is not merely internally consistent: the output corroborates that the
   spanned rollout ran on the pre-update weights.

Both records are `replay_ready`, both round-trip through
`TrajectoryRecorder.write_jsonl`/`read_jsonl` byte-for-byte in meaning, and the
summary is `2 trajectories over ['rc-0', 'rc-1']; 2 replay-ready, 0 manifest-only`
— which is the blind spot from 3C and 4C closed, for records that declare a
source.

## "Declared" is the honest word

Nothing in this run verifies a single weight byte. `exactness == "declared-source"`
says the *caller* stated `checkpoint`, `run_id` and `step`; `replay_ready` says the
record **can support** a replay claim, not that one has been made. What the
identity buys is that two versions can no longer collide — the failure Phase 3C
found — and what it does not buy is proof that step 1's bytes are the ones on the
card.

The design's answer to that is the replay validator (item 8): a trajectory carries
the prompt *and* the token ids precisely so the generation can be re-run on the
same declared source and compared. That validator exists as of this phase's
follow-up, and it is the thing that converts "declared" into evidence — in one
direction only, as the document for it says.

## What Phase 5 does *not* prove

- **`seconds` is `null` for `R-long`.** The record is assembled after the cycle
  from the captured outcome, and the worker thread's elapsed time was not threaded
  into it. The field is optional and unset rather than wrong, but the in-flight
  rollout's wall time is not in the artifact.
- **No logprobs.** The harness requests plain greedy completion, so the records
  are token-only; the logprob lane of the replay validator has not been exercised
  against a real engine.
- **One span, one step.** `rc-0 → rc-1` only. Nothing here exercises several
  updates in a row, or a rollout that spans two of them.
- **`opt-125m`, one node, TP=1.**
- **`finish_reason` is `length` for both**, so neither record has a natural stop
  to check against.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_trajectory_2gpu.py
```
