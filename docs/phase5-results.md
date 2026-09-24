# Phase 5 results — trajectories that name the training step

**Result: PASS. 16/16 checks — with one field in the committed artifact known to
be wrong, corrected in the harness and documented below rather than edited out of
the measurement.** Roadmap item 5, on real hardware: a trajectory carries the
version it was **admitted** to *and* a declared provenance, which together
separate two checkpoints that the manifest digest cannot.

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
- **PR #49040** made the other half structural: it added the `weight_version`
  query API and deliberately removed binding a version to
  `Request`/`RequestOutput`, because a request *may* span two versions on the
  `mode="keep"` path. So the record a trainer needs has to live outside the
  engine.

`rolloutcore.trajectory` is that record, and the enabling piece on the trainer
side is `NCCLWeightTransferDriver.declare` — a driver whose cached identity cannot
follow weights that changed has no way to *re*-declare what it is about to
install.

## The experiment

Phase 4A's shape, with provenance. `R-long` is admitted at `rc-0` while the update
to `rc-1` is requested; the drain holds the mutation until the request returns;
`R2` is admitted afterwards at `rc-1`.

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
| engine at completion | `rc-0` *(see the defect below)* | `rc-1` |
| output | 256 × `<s>` (token id `0`) | `' the capital of the French Republic.…'` |

Three rows are the result:

1. **The two records have different identities and the same manifest.** The same
   `925369d663bc` under both is not a defect in the record; it is why provenance
   exists. `c8c0aafd93a5` vs `f7a652cf4d70` is the difference a step number makes.
2. **The binding, not the engine's last word, is what the record keeps.** After
   the cycle the engine reports `rc-1` — and that label is not evidence about
   `R-long`, whose tokens were produced entirely at `rc-0`. A record built from
   the engine's post-hoc label would attribute them to the wrong step.
3. **The tokens corroborate the binding.** All 256 token ids are `0` (`<s>`),
   which is what dummy-initialised weights produce, while `R2`'s are real text.
   The check `inflight_text_is_the_step_0_weights` asserts exactly that.

Both records are `replay_ready`, both survive a
`TrajectoryRecorder.write_jsonl`/`read_jsonl` round trip, and the summary is
`2 trajectories over ['rc-0', 'rc-1']; 2 replay-ready, 0 manifest-only` — the
blind spot from 3C and 4C closed, for records that declare a source.

## A defect in this artifact, corrected in the harness

`results/phase5.json` and `results/phase5-trajectories.jsonl` record
`engine_version_at_completion: "rc-1"` for `R-long`, and the check
`inflight_engine_reported_a_different_version_at_completion` was built to pass on
it. **That label is the engine's state after the whole cycle, not at the request's
completion.** The harness assembled the trajectory after `run_cycle` and read
`weight_info()` then.

The true completion-time label is `rc-0`. The drain cannot confirm - and the
update cannot begin - while RolloutCore still counts the rollout, and the request
thread is the only thing that releases it, so the label when `R-long` returned
must have been the version it was admitted to.

The harness now reads the label **in the request thread, between the response
arriving and the rollout being released**, and captures the elapsed seconds in the
same place. The check is now
`inflight_engine_reported_the_bound_version_at_completion`: it passes when
`spans_an_update is False`, which is the honest statement of the design - the
drain exists so that no rollout spans a mutation, so `True` is the **violation**,
not an expected event.

The committed artifacts are left exactly as measured. Rewriting a measurement to
match a later understanding would destroy the only thing an artifact is for; this
section is the correction of record, and the next run of the harness writes
`rc-0` and the new check name. The `results/phase5-trajectories.jsonl` records are
still valid inputs for the replay validator (item 8), which uses `version`, the
identity and the tokens; the field in question is not part of that comparison.

## "Declared" is the honest word

Nothing in this run verifies a single weight byte. `exactness == "declared-source"`
says the *caller* stated `checkpoint`, `run_id` and `step`; `replay_ready` says the
record **can support** a replay claim, not that one has been made. What the
identity buys is that two versions can no longer collide — the failure Phase 3C
found — and what it does not buy is proof that step 1's bytes are the ones on the
card.

The design's answer to that is the replay validator (item 8): a trajectory carries
the prompt *and* the token ids precisely so the generation can be re-run on the
same declared source and compared. That is the mechanism that converts "declared"
into evidence — in one direction only, as `docs/replay-validator.md` says.

## What Phase 5 does *not* prove

- **No rollout was observed spanning a mutation**, and none can be under this
  design. `spans_an_update` is a detector with a synthetic unit test; producing a
  real span would require the `mode="keep"` path RolloutCore deliberately cannot
  reach, which is a different experiment (and one worth running against vLLM's
  documented async RL flow).
- **No logprobs.** The harness requests plain greedy completion, so the records
  are token-only; the logprob lane of the replay validator has not been exercised
  against a real engine.
- **One step, one cycle.** `rc-0 → rc-1` only: nothing here exercises several
  updates in a row, or a chain of provenance declarations.
- **`opt-125m`, one node, TP=1.**
- **`finish_reason` is `length` for both**, so neither record has a natural stop
  to check against.
- **The artifact's completion label is wrong** (previous section) — every other
  field in it is as measured.

## Reproduce

```bash
pkill -f "vllm serve"; sleep 3
cd /workspace/rolloutcore && source .venv/bin/activate
export HF_HOME=/workspace/hf
python3 diagnostics/rolloutcore_trajectory_2gpu.py
```
