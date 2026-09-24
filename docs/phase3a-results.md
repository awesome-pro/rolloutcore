# Phase 3A results: real vLLM control plane, one GPU

**Result: PASS. 16/16 checks, 0 warnings, 0 failures**, against the audited
commit, on a real `vllm serve`.

| | |
|---|---|
| Date | 2026-09-24 |
| vLLM | `0.30.1rc1.dev60+g00b7847c8` = `00b7847c8036b667742b4efb21aab1de51fd4721` (exact match) |
| RolloutCore | `586d8c74e2bdd9e0aab79ef4a90881168fbf4eec` |
| GPU | NVIDIA RTX A6000, 49140 MiB, compute 8.6 |
| Driver / CUDA | 580.159.03 / 13.0 |
| Wheel variant | `cu130` (torch `2.13.0+cu130`) |
| Python | 3.12.3 (uv, after `pip install -U uv`) |
| Model | `facebook/opt-125m` @ `27dcfa74d334bc871f3234de431e71c6eeba5dd6` |
| Server flags | `--enforce-eager --max-model-len 512 --gpu-memory-utilization 0.6` |
| Artifact | `results/phase3a.json` |
| Wall clock | 8.4 s total, of which 2.0 s is the deliberate 256-token in-flight request |

This is an **adapter-level** test, as the plan specifies: no weight transfer is
performed and no fake update is fabricated, so `LifecycleRunner.run_cycle()` is
never called. The controller is driven through
`READY → DRAINING → QUIESCED`; the remaining three steps run through the adapter.

---

## The eleven required behaviours, and the evidence

| # | Requirement | Result |
|---|---|---|
| 1 | fresh `/weight_info == "default"` | pass |
| 2 | control-plane-only bootstrap seeds `rc-0` | `pre_seed='default' → 'rc-0'`, `driver=None`, no transfer engine, 13.5 ms |
| 3 | a second bootstrap refuses, `rc-0` unchanged | `AlreadyManagedEngineError(observed='rc-0')`, label still `rc-0`, 2.8 ms |
| 4 | deterministic generation before pause | two greedy requests, identical text **and** identical 16 token ids |
| 5 | `pause(mode="wait")` non-blocking | `begin_drain` 0.9 ms, first `await_drain` 0.0 ms, `engine_drain_completed=False` while generation continued |
| 6 | …and waits for the in-flight request | 256-token stream completed in 2.02 s; drain completed 2.13 s after the pause began |
| 7 | engine reports paused | `is_paused=true`, `weight_version='rc-0'` |
| 8 | prefix / encoder / MM resets succeed | `prefix=true encoder=true mm=true`, raw HTTP 200/200/200, 17.5 ms |
| 9 | pre-resume validation sees `rc-0` **and** paused | `weight_version='rc-0' is_paused=True`, 5.6 ms |
| 10 | resume succeeds, engine unpaused | acknowledged, `is_paused=false` (confirmed by a second independent read), 9.6 ms |
| 11 | deterministic generation still works | byte-identical to the baseline: `sha256=66f332a6…`, same 16 token ids |

### The measurement that matters most

The pause did **not** abort the in-flight request and did **not** return early:

```
in-flight request:  256 tokens in 2.0203 s
drain completed:    2.1313 s after the pause began   (+110 ms)
```

If the drain had used `mode="abort"` (or if `mode="wait"` had silently been
rejected and retried), the drain would have returned in milliseconds and the
stream would have been truncated. It did neither. This is the behaviour I2
("one weight version per rollout") depends on, verified on a real engine for the
first time.

### Lifecycle control-plane cost

| Operation | Time |
|---|---|
| bootstrap (read → seed `rc-0` → read back) | 13.5 ms |
| cache invalidation (3 endpoints) | 17.5 ms |
| pre-resume validation (2 reads) | 5.6 ms |
| resume (+ read back) | 9.6 ms |
| **total control plane, excluding the drain wait** | **≈ 46 ms** |

The drain itself is bounded by real work, not by our code: 2.13 s for a request
that took 2.02 s. That ~46 ms figure is the honest headline for the project,
and it is a *lower* bound, since Phase 3A does not transfer any weights.

### Generation determinism

```
prompt: "The capital of France is"
baseline  (t=0.889 s): 16 tokens  sha256=66f332a645dabc37a52389e3ac9ea487aa677537bb6b48e0c42f93d0096d77f2
post-resume (t=0.116 s): 16 tokens  sha256=66f332a645dabc37a52389e3ac9ea487aa677537bb6b48e0c42f93d0096d77f2
token ids: [5, 812, 9, 5, 1515, 3497, 4, 50118, 50118, 133, 812, 9, 1470, 16, 5, 812]
```

Identical text *and* identical token ids before and after a full
drain → cache-reset → validate → resume cycle. The caches were genuinely dropped
in between (requirement 8 passed), so this is recomputation agreeing with the
cached path, not a stale-cache hit. The 7.6× latency difference between the two
is warm-up (CUDA context, kernels), not caching.

---

## What Phase 3A does *not* prove

- **Nothing was updated.** The generation never advanced past `rc-0`; the engine
  and the controller agreed on one version throughout.
- **`WeightIdentity` is declared, not verified.** The identity recorded is
  `4d15b0ad0141 (declared checkpoint=facebook/opt-125m@27dcfa74…)`, computed from
  the real parameter manifest, but the engine only reports an opaque version
  string, so nothing confirms that its memory holds those bytes.
- **No mixed-version guarantee was exercised.** Without a real update there is
  no second version for a rollout to straddle; that is Phase 3B/3C.
- **The encoder lane is untested.** `opt-125m` has no vision encoder, so
  `/reset_encoder_cache` and `/reset_mm_cache` were no-ops that returned 200.

---

## Findings for the next round

Two are harness defects noticed while reading the artifact, both small, neither
affecting the verdict above:

1. **`world_size` is observed and then discarded.** `GET /get_world_size` is
   called, but `bootstrap` reports `world_size=None` whenever no transfer driver
   is configured. The engine answered. Phase 3B needs this number: the NCCL
   rendezvous is sized `1 + get_world_size()`
   (`examples/rl/rlhf_http_nccl.py:177-179`), and it should be recorded
   unconditionally.
2. **`rolloutcore_dirty: true` over-reports.** The tree had no modified tracked
   files; the flag was tripped by an untracked scratch file at the repo root.
   `git status --porcelain` should be run with `--untracked-files=no`, or the
   untracked set recorded separately.

And three observations that inform Phase 3B and Phase 6:

3. `/reset_encoder_cache` and `/reset_mm_cache` return **200 on a text-only
   model**. All three resets are therefore satisfiable on any model, but on a
   text model the encoder/MM half is a no-op, so proving the encoder lane needs a
   multimodal model (Phase 6's cache-coherence experiment).
4. The controller journal from the real run is
   `initialize → admit_rollout → finish_rollout → begin_drain → confirm_drained → taint`:
   the drain path was driven end to end by **real engine evidence** for the first
   time. The final `taint` is deliberate (see below).
5. The pod's stock `uv` 0.9.0 rejects `--torch-backend=cu130`; after
   `pip install -U uv` the cu130 install resolved cleanly. Mixing a cu130 vLLM
   wheel with cu129 torch produces `ImportError: libcudart.so.13`, so the variant
   must be chosen once, before installing anything.

## Why the controller ends `TAINTED`

The adapter resumed an engine the controller did not resume, because V1's table
has no update-free revalidation path (`docs/state-machine.md` §12 item 5). A
controller that cannot vouch for a serving engine must not claim `READY`, so it
taints. The report marks this `expected: true`, and it is the correct fail-closed
outcome rather than a workaround: the alternative would have been fabricating an
update, which the plan forbids.

## Reproduce

```bash
# on the pod, in the venv
vllm serve facebook/opt-125m --host 127.0.0.1 --port 8000 \
    --enforce-eager --max-model-len 512 --gpu-memory-utilization 0.6
python3 scripts/live_control_plane_smoke.py \
    --base-url http://127.0.0.1:8000 --model facebook/opt-125m \
    --vllm-sha 00b7847c8036b667742b4efb21aab1de51fd4721 \
    --identity-source manifest
```

Full procedure, checkpoints and troubleshooting: `docs/phase3a-runbook.md`.
