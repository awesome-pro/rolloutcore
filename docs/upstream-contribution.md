# Upstream contribution — what the integration found, and what to do with it

This document is the item-11 deliverable: the candidate upstream contributions
that fell out of building RolloutCore against vLLM `main` at
`00b7847c8036b667742b4efb21aab1de51fd4721`.

Every claim about vLLM is anchored to `file:LINE` in that checkout. Every claim
about RolloutCore's own measurements is anchored to an artifact in `results/`
or a document in `docs/`. Nothing here has been filed, and no maintainer has
seen it.

---

## 1. The policy, restated

`PROJECT.md` (section *"Upstream contribution (item 11)"*) commits to nothing in
advance. It sets a priority order and says a documented negative result is an
acceptable outcome:

1. an end-to-end weight-update / cache-coherence regression test;
2. a real bug found during integration;
3. response version spans — a design discussion, not a defect.

What the integration actually surfaced, by that order:

| # | Finding | Status against upstream |
|---|---|---|
| 1 | No test in the tree generates across a weight update, and the composite RL lane of RFC #48312 has no implementation | **New.** Primary proposal, §2 |
| 2 | The RL update endpoints pass no timeout, although the shared machinery accepts one and the sibling dev route exposes it | **New.** Strongest patch candidate, §3.4 |
| 3 | `/pause`'s `clear_cache` documentation contradicts its behaviour in every mode | **New.** Documentation defect, §3.3 |
| 4 | `finish_weight_update` resets LoRA state but no cache | **Known, unfixed.** #48762 was closed unmerged; RFC #48312 exit criterion 7 wants an equivalent fix. §3.1 is a *documented negative result* on the "patch it again" idea |
| 5 | Encoder / multimodal caches are not invalidated by a weight update | **Known, unfixed, and self-documented in the code.** RFC #48312 category 7 already tracks the related open bugs. §3.2 |
| 6 | `cache_salt` covers only the prefix-KV block hash | **Partly new as a mechanism note; already covered as a defect.** The prefix cache *is* keyed by `cache_salt`; the other caches are separate and unkeyed by it. §3.2 |
| 7 | No lease / recovery protocol for a managed engine | **New, and a design discussion rather than a patch.** §3.5 |
| 8 | KV cache is sized with no headroom for the weight-transfer receive buffers | **New.** Measured OOM. §3.6 |
| 9 | The engine's `weight_version` is not evidence of which weights produced a response | **Known design question** (RFC #48306 §2.2; maintainer thread in `research/comments_48306.md`). §1.1 supplies a measured data point |

### 1.1 The one measured contribution to the version-span discussion

RFC #48306 §2.2 is where per-request version binding was removed — recorded in
`research/comments_48306.md`: #49040 *"implemented the query/update APIs, but
intentionally removed binding a version to `Request`/`RequestOutput` after review
noted that one request may span multiple weight versions."* The same thread asks
which contract a follow-up should implement.

RolloutCore measured the case that motivated the removal. In `results/phase5.json`
(`docs/phase5-results.md`), a request was admitted while the engine reported
`rc-0`, the update ran while it was in flight, and by the time it finished the
engine reported `rc-1`:

| request | bound version | engine label at admission | engine label at completion |
|---|---|---|---|
| `R-long` | `rc-0` | `rc-0` | **`rc-1`** |
| `R2` | `rc-1` | `rc-1` | `rc-1` |

`R-long`'s tokens were produced entirely by the `rc-0` weights — its 256-token
completion is the degenerate `'<s>'` output of the dummy checkpoint, and the
Phase 4A run of the same shape reproduced the pre-update 16-token baseline
prefix exactly (`results/phase4a.json`). So the engine's post-hoc label names the
version that was *current when the request ended*, not the version that produced
it. Any contract that reads `weight_version` after the fact inherits that error;
the admission-time observation (`engine_version_at_admission`) is the one that
matched in every case we ran. This is a data point for the open question, not a
proposal to change the API.

---

## 2. Primary proposal — an end-to-end cache-coherence regression test

### 2.1 The gap, verified

**The inventory.** There is exactly one directory whose name promises lifecycle
coverage, and it contains one test module:

```console
$ find . -type d -name state_transitions
./tests/entrypoints/serve/dev/rlhf/state_transitions
$ ls tests/entrypoints/serve/dev/rlhf/state_transitions/
__init__.py  test_pause_resume.py
```

`test_pause_resume.py` is 168 lines and covers pause/resume state, mode
semantics, and — importantly — that `clear_cache` changes prefix-cache
behaviour: `test_clear_cache_preserves_output_and_controls_prefix_cache`
(`tests/entrypoints/serve/dev/rlhf/state_transitions/test_pause_resume.py:145`)
warms a prompt, asserts `cached_tokens(baseline) == 0` then
`cached_tokens(warmed) > 0` (`:155`), and repeats with `clear_cache=False`.
That test proves the *oracle* works. It never touches weights.

**The helpers that were written for the missing lane are unused.** The shared
fixture module `tests/entrypoints/serve/dev/rlhf/conftest.py` defines weight
transfer, sleep/wake, and metrics helpers that no test imports:
`start_weight_update` (`:384`), `finish_weight_update` (`:392`),
`get_world_size` (`:396`), `sleep` (`:353`), `wake` (`:359`),
`is_sleeping` (`:364`), `sleep_metrics` (`:422`), `gpu_free_bytes` (`:409`),
plus `poll_until` (`:146`) and `gen_with_logprobs` (`:194`).

*How this was checked*, since a bare-name grep lies here: `sleep` also names
functions in `vllm/entrypoints/llm.py` and `vllm/device_allocator/cumem.py`, and
`get_world_size` names one in `examples/rl/rlhf_http_nccl.py`. The check that
matters is imports, not names:

```console
$ grep -rn "serve.dev.rlhf.conftest import" --include=*.py tests/
tests/entrypoints/serve/dev/rlhf/state_transitions/test_pause_resume.py:13:
  from tests.entrypoints.serve.dev.rlhf.conftest import (
      cached_tokens, completion_with_cache_details, gen, golden_output,
      is_paused, ok, pause, resume, server, start_stream)
```

That is the *only* import of this conftest anywhere in the tree, and it imports
ten names — none of them a weight-transfer or sleep/wake helper.

**There is no test that generates across an update.** This is the claim most
likely to be wrong, so it is stated precisely, because weight-transfer tests *do*
exist:

| Test | What it actually does |
|---|---|
| `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py:241` (`test_full_weight_transfer_flow`) | init → start → update → finish → assert the *version string* changes (`:266` `"default"`, `:287` still `"default"` before finish, `:292` `"step-42"` after). In-process, **mock engine**, **never generates** |
| `tests/v1/worker/test_gpu_worker_weight_transfer.py:101` | Worker-level delegation and session-guard unit tests against a `_RecordingEngine`. No server, no generation |
| `.../state_transitions/test_pause_resume.py` | Pause/resume and cache effects. No weights |

So: update *plumbing* is tested, pause/resume *cache* effects are tested, and the
two are never joined. The mechanical check was: enumerate every test file that
contains a completions call and an update symbol.

```console
$ for f in $(grep -rln "completions" --include=*.py tests/); do \
    grep -q "weight_update\|update_weights" "$f" && echo "BOTH: $f"; done
BOTH: tests/entrypoints/serve/dev/rlhf/conftest.py      # helper definitions only
BOTH: tests/entrypoints/openai/test_openai_schema.py    # schema strings, not a flow
```

`test_full_weight_transfer_flow` is also the clearest illustration of RFC
#48312's own warning that *"HTTP success alone is not a correctness oracle"*: it
asserts a version string advanced while nothing asserts a weight changed.

**The RFC asks for exactly this.** From `research/rfc_48312.json` (RFC #48312,
state `open`), category 7's minimum regression check is:

> warm cache on A → update to B → reuse the same request → assert the cache path
> is exercised and the result matches cold-cache B

and the exit criteria include:

> **7 — Post-update cache coherence:** #48762 or an equivalent non-reverted fix
> lands; model, adapter, target, and draft updates either invalidate every
> affected cache or advance the generation used in cache identity.

> The composite `sleep → update → wake → resume` lane passes for each advertised
> model feature combination, jointly satisfying #48310 and this RFC.

(Quoted from the RFC body; the long form of the lane appears in the proposed CI
architecture section — `pause/drain → sleep → wake(weights) → update →
post-load/finalize → wake(kv_cache) → cache invalidation → resume`.)

### 2.2 What the test would do

Only existing public surface: the dev router, the `state_transitions` conftest,
and `--enable-prompt-tokens-details`, which is what makes `cached_tokens`
available. The oracle in `cached_tokens` is a frontend argument
(`vllm/entrypoints/launchers/cli_args.py:139`) and the server fixture already
passes it (`tests/entrypoints/serve/dev/rlhf/state_transitions/test_pause_resume.py:42`).

Two lanes, because the cheap lane and the complete lane have different oracles.

**Lane A — differential value refresh, 1 GPU, in-process.** This is the RFC's
*"result matches cold-cache B"* half, and it is feasible today because the
in-tree mock engine is handed the real model:
`mock_create_engine(config, vllm_config, device, model)`
(`tests/entrypoints/weight_transfer/test_weight_transfer_llm.py:96`) constructs
`MockWeightTransferEngine(config, vllm_config, device, model)`, whose `__init__`
passes `model` to `super()` (`:67`). A mock that writes a deterministic sentinel
into the named parameters therefore changes real weights with no transport at all.

1. Start an `LLM` on a tiny model with `--load-format dummy`, prefix caching on.
2. Generate with a fixed prompt at temperature 0 → `out_A`.
3. Update through the existing API: `init_weight_transfer_engine` →
   `start_weight_update` → `update_weights` → `finish_weight_update("B")`, with
   the mock writing sentinel values derived from a seed.
4. Generate the same prompt → `out_B`. Assert `out_B != out_A`.
5. Re-load the engine cold on the same sentinel weights → `out_cold_B`. Assert
   `out_B == out_cold_B`: this is what separates "the update took effect" from
   "the engine served a stale continuation from a warm cache".
6. Repeat the same-prompt request after `finish_weight_update` and assert the
   result equals `out_cold_B`, with the version advanced.

**Lane B — the composite lane, real transfer, RL CI matrix.** Lane A's mock
cannot show that the cache *path* was exercised, because `cached_tokens` is a
server-side field and a mocked transfer never populates a KV block under two
different weight generations. Lane B is the RFC exit criterion:

1. `server(extra_args=["--enable-prefix-caching", "--enable-prompt-tokens-details"])`
   from the conftest fixture (`tests/entrypoints/serve/dev/rlhf/conftest.py:81`).
2. Warm the prefix cache for a fixed prompt: `completion_with_cache_details`
   (`:316`) twice; assert `cached_tokens` goes `0` then `> 0`
   (`cached_tokens` at `:344`). This is the existing test's own setup, reused.
3. Quiesce: `pause(url, mode="wait", clear_cache=False)` (`:304`).
   `clear_cache=False` on purpose — the point is to prove the *update* is what
   invalidates, not the pause.
4. Run a real update: `start_weight_update` (`:384`) → `POST /update_weights`
   (`vllm/entrypoints/serve/dev/rlhf/api_router.py:186`) →
   `finish_weight_update` (`:392`).
5. Invalidate and resume, in the order the RFC specifies: `POST /reset_prefix_cache`
   (`vllm/entrypoints/serve/dev/cache/api_router.py:20`), `POST /reset_encoder_cache`
   (`:57`), `POST /reset_mm_cache` (`:47`), then `resume` (`:312`).
6. Reuse the *same* request. Assert (a) `cached_tokens` is `0` — the cache path
   was not reused — and (b) the result equals a cold-B run.

Step 6(a) is the assertion that makes this a coherence test. RolloutCore
validated the oracle and the failure mode on real hardware: with the reset step
present, the first post-update request scored **0** new prefix-cache hits;
with the reset deliberately skipped and the weights changed out of band, the
engine handed back **32 cached tokens** computed under the previous weights and
the output changed as a result (`results/phase4c.json`:
`kv_reused_across_weights: true`, `texts_differ: true`,
`hits_with_stale_cache: 32.0`). The same run shows the "cache works" control
(`results/phase4c.json`: the second post-update request scores hits again), which
is what stops "no reuse" from being indistinguishable from "caching silently
stopped".

### 2.3 Sketch of the test file

Marked as a sketch: names and endpoints are real, the bodies are not runnable as
written.

```python
# tests/entrypoints/serve/dev/rlhf/state_transitions/test_update_coherence.py
"""Prefix-cache coherence across a weight update. RFC #48312 category 7."""

from tests.entrypoints.serve.dev.rlhf.conftest import (
    cached_tokens,
    completion_with_cache_details,
    finish_weight_update,
    pause,
    resume,
    server,
    start_weight_update,
)

PROMPT = "Paris is the capital of France. Berlin is the capital of Germany. " * 20


def test_prefix_cache_is_not_reused_across_a_weight_update():
    with server(extra_args=["--enable-prefix-caching",
                            "--enable-prompt-tokens-details"]) as url:
        # 1. warm the cache under A
        assert cached_tokens(completion_with_cache_details(url, PROMPT)) == 0
        assert cached_tokens(completion_with_cache_details(url, PROMPT)) > 0

        # 2. quiesce without clearing: the update must be what invalidates
        assert pause(url, mode="wait", clear_cache=False) == 200

        # 3. move weights A -> B through the public update surface
        assert start_weight_update(url).status_code == 200
        #    POST /update_weights with a payload the configured backend accepts
        assert finish_weight_update(url).status_code == 200

        # 4. invalidate, then resume
        assert requests.post(f"{url}/reset_prefix_cache").status_code == 200
        assert requests.post(f"{url}/reset_encoder_cache").status_code == 200
        assert requests.post(f"{url}/reset_mm_cache").status_code == 200
        assert resume(url) == 200

        # 5. the same request, after the update
        after = completion_with_cache_details(url, PROMPT)
        assert cached_tokens(after) == 0, "stale KV was served after the update"
        assert after == cold_run_at_B(url, PROMPT)
```

Two gaps the sketch makes visible, and both are the reason this is a proposal
and not a patch: step 3 needs a *real* weight delta, which in CI means either a
synthetic in-tree trainer for one of the transfer backends, or Lane A's
mock-writes-sentinels variant; and `cold_run_at_B` needs a second engine started
on B for a genuine cold comparison.

### 2.4 Why this belongs upstream, not in RolloutCore

* It tests vLLM's own invariant. RFC #48312 requirement 7 — *"Any cache derived
  from model/adapter version N must either be invalidated or keyed by a new
  identity before version N+1 serves requests"* — is a vLLM requirement, and
  vLLM owns the caches, the reset endpoints, and the update path.
* RolloutCore can only *drive* it. Our Phase 4C harness proves the failure mode
  is reachable and that the reset is load-bearing, but it tests RolloutCore's
  policy, not vLLM's default behaviour: an operator who calls `/pause` with
  `clear_cache=false` and then updates weights gets the same exposure with no
  RolloutCore in the picture at all.
* The absence of the lane has a cost already visible in the tracker: #48762
  ("invalidate encoder cache on finish_weight_update") was closed unmerged, and
  nothing in the tree can demonstrate the delta either way — because nothing
  generates across an update.

---

## 3. Defect findings and verdicts

Verdicts are one of **report**, **patch**, or **already covered upstream**. All
line numbers are from the pinned checkout; the measured columns cite this repo's
artifacts.

### 3.1 `finish_weight_update` resets LoRA state and no cache

**Anchor.** `vllm/v1/worker/gpu_worker.py:1488-1505`. The method delegates to the
transfer engine (`:1499`), resets the session, and ends with
`self.model_runner.reset_lora_state()` (`:1505`) — which is
`vllm/v1/worker/lora_model_runner_mixin.py:29`, documented as *"Invalidate LoRA
state after base weights are replaced"* (`:30`) and a no-op without
`lora_config`. No prefix, encoder, or multimodal cache is touched.

**Correction to the inherited phrasing.** "`finish_weight_update` invalidates
nothing" is not literally true: it invalidates LoRA adapter state. The defensible
statement is that it invalidates *only* LoRA state, and no KV/encoder/MM cache.

**Known upstream.** RFC #48312 exit criterion 7 requires
*"#48762 or an equivalent non-reverted fix"*; #48762 was closed unmerged (recorded
in `source-map-vllm-main.md`, which cites GitHub state `closed`,
`merged_at: null`). The *reason* for closing is not in this checkout.

**Verdict: report, as a question first.** Re-proposing the same patch without
knowing why it was rejected is not a contribution. The useful artifact is §2's
test plus a question: which commit point should invalidate — the worker's
`finish_weight_update`, or the engine's commit before resume? RFC #48312's own
proposed design prefers the engine-side completion gate, which may be exactly
what a previous worker-side patch failed to satisfy.

### 3.2 Encoder / multimodal caches are not invalidated by an update, and the code says so

**Anchors.** `vllm/v1/engine/core.py:841` — `reset_encoder_cache`'s docstring
states the requirement outright: *"This should be called when model weights are
updated to ensure stale vision embeddings computed with old weights are not
reused"* (`:844`). The method resets both the scheduler-side and GPU-side caches.
The pieces exist and are wired: `_reset_caches` (`:861`) calls
`reset_mm_cache()` (`:874`) and `reset_encoder_cache()` (`:875`), and the pause
path reaches them through `_finish_pause` (`:877`, `:881`). The *update* path
does not — see §3.1. The endpoints exist too: `POST /reset_encoder_cache`
(`vllm/entrypoints/serve/dev/cache/api_router.py:57`) and `POST /reset_mm_cache`
(`:47`).

**`cache_salt`, precisely.** The inherited phrasing — that `cache_salt` does not
cover the encoder/MM/LoRA caches — is right in effect but needs the mechanism.
`cache_salt` *does* key the **prefix-KV** block hash: it is one of four extra-key
sources in `generate_block_hash_extra_keys`
(`vllm/v1/core/kv_cache_utils.py:610`), specifically
`cache_salt_keys` (`:632`, first block only) combined as
`lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys`
(`:639-640`). So the prefix cache *is* versionable by a caller — RolloutCore
relies on exactly this, stamping the version label as the salt
(`results/phase4a.json`, `binding.cache_salt: "rc-0"`). What is *not* versionable
by it are the encoder, multimodal, and adapter caches, which are separate stores
with separate reset endpoints and no version in their identity.

**Already covered upstream.** RFC #48312 category 7 tracks the same class as open
bugs, including a same-name tower/connector LoRA reload reusing stale multimodal
encoder-cache embeddings and a same-name LoRA reload reusing prefix-cache blocks
from the previous adapter contents.

**Verdict: already covered upstream.** Nothing to report that the RFC does not
already carry. Its value here is as a *reason* the lane in §2 must assert cache
identity rather than merely "outputs changed".

### 3.3 The `/pause` `clear_cache` documentation contradicts the code in every mode

**Anchor.** `vllm/entrypoints/serve/dev/rlhf/api_router.py:45` documents the
parameter as *"DEPRECATED. Whether to clear KV/prefix caches after draining.
**Ignored when mode=\"keep\"**."*

The code honours it in every mode, at both engine paths:

* `AsyncLLM.pause_generation` forwards it unchanged —
  `vllm/v1/engine/async_llm.py:950` — and also clears the multimodal cache when
  it is true (`:948`), which the docstring above does not mention;
* in-process: `pause_state = PauseState.PAUSED_ALL if mode == "keep" ...`
  (`vllm/v1/engine/core.py:908`) is followed unconditionally by
  `self._finish_pause(clear_cache)` (`:910`) → `_reset_caches()` (`:881`);
* out-of-process: the same pattern, with `_finish_pause(clear_cache)` on the idle
  callback (`vllm/v1/engine/core.py:2021`).

So a caller who sets `mode="keep"` and relies on the docstring will get their
caches cleared. RolloutCore never used `keep` — it is unreachable from the
controller by design, because a frozen request can span two weight versions — but
the defect is real for anyone reading that docstring.

**Verdict: patch (documentation).** Small, self-contained, no behaviour change:
either drop the "Ignored when mode=keep" clause or mark the parameter as
honoured in all modes, and mention the multimodal side effect. The code is
consistent across three call paths, so the docstring is the outlier.

### 3.4 The RL update endpoints pass no timeout, though the machinery supports one

This is the finding with the crispest fix, and it is the one RolloutCore measured
as an unbounded hang.

**Anchors.**

* The timeout exists end to end: `AsyncLLM.collective_rpc` accepts
  `timeout: float | None = None` (`vllm/v1/engine/async_llm.py:1142`, `:1145`)
  and forwards it (`:1150`); `EngineCore.collective_rpc` accepts and forwards it
  (`vllm/v1/engine/core.py:1033`); `EngineCoreClient.collective_rpc_async` accepts
  it (`vllm/v1/engine/core_client.py:1329`) and passes it as an argument
  (`:1336`).
* A sibling dev route already exposes it: `POST /collective_rpc` reads
  `timeout` from the request body (`vllm/entrypoints/serve/dev/rpc/api_router.py:23`,
  `:42`).
* The RL update calls omit it: `start_weight_update` (`vllm/v1/engine/async_llm.py:1267`),
  `update_weights` (`:1280`), `finish_weight_update` (`:1286`) call
  `collective_rpc` with no timeout, and the routes themselves take none
  (`vllm/entrypoints/serve/dev/rlhf/api_router.py:174`, `:186`, `:204`).
* With `timeout=None` the wait is a bare future:
  `return await future` (`vllm/v1/engine/core_client.py:1248`), and the sync path
  is `future.result()` with no deadline (`:996`, `:1002`).

**The observable symptom, measured.** RolloutCore's Phase 6 killed the engine
2.17 s into a 4.13 s broadcast. The controller's `complete_weight_update` never
returned: the harness's 120 s budget expired with
`detail.deep_kill.hung: true`, `seconds_to_outcome: 120.2254`, and the controller
left in `UPDATING` with `tainted: false` — fail-closed (nothing published, no
rollout admissible) but silent, and with no automatic exit
(`results/phase6-8b-with-deepkill.json`). The same scenario at 125M, where the
collective is 0.10 s long, was detected in 0.177 s
(`results/phase4d.json`), so the exposure scales with transfer duration.

On the trainer side the timeout is also absent, and for a different reason: the
transfer communicator is built by vLLM's own wrapper and exposes no timeout —
`PyNcclCommunicator.__init__(self, group, device, library_path=None)`
(`vllm/distributed/device_communicators/pynccl.py:95-100`) — while the file's
only timeout is on teardown, and its own comment records that a failed join
*"leaves the peer blocked in `ncclCommInitRank` until timeout"* (`:237`).

**Verdict: patch.** Pass a timeout at the three `AsyncLLM` call sites (a
configuration value with a sane default), and surface expiry as an error the
caller can act on rather than an indefinite await. The engine already accepts and
forwards the value, so the diff is small and needs no new architecture. Note the
honest limit: this bounds *one* of the two waits. The trainer-side collective has
no timeout to pass, so a complete fix also needs a watchdog around the broadcast
— that part is a design discussion, and is why the verdict is "patch" for the
server half and "report" for the trainer half.

### 3.5 No lease or recovery protocol for a managed engine

**Anchor.** The engine's version is a bare string with no owner record:
`set_weight_version` / `get_weight_version`
(`vllm/v1/engine/core.py:1042-1047`), surfaced as
`{"weight_version": ...}` by `GET /weight_info`
(`vllm/entrypoints/serve/dev/rlhf/api_router.py:222`). There is no heartbeat, no
lease, and no endpoint that hands a managed engine back.

**Measured.** In Phase 4B, after an update failed mid-flight, a fresh controller
was refused — the adapter read the label, concluded the engine was managed, and
raised before writing anything:

> `AlreadyManagedEngineError: engine already reports weight_version 'rc-1', which
> RolloutCore wrote: another controller may own it. Bootstrap refused before
> writing anything. A lease/recovery protocol is required to take over a managed
> engine and V1 does not implement one.`

(`results/phase4b.json`, `detail.identity_mismatch.fresh_controller_bootstrap`.)
The same run measured the other side of the gap: a drain that cannot complete
leaves the controller in `DRAINING`, untainted, with `DrainFailedError` and no
automated exit (`detail.dead_engine_drain_path` — `tainted: false`, state
`DRAINING`; `docs/phase4b-results.md`).

**Verdict: report — a design discussion, not a patch.** In V1 the version label
is the only ownership signal, and it is a string the caller writes. Adding a
lease means deciding who grants it, how it expires, and what an engine does when
the holder disappears — an RFC-scale question, appropriate for the #48312 /
#48306 track rather than a drive-by PR. The contribution is the measured failure:
refusal is correct, and refusal with no recovery path is the gap.

### 3.6 KV cache sized with no headroom for the transfer buffers

**Anchors.** At startup the KV budget is everything left over:
`self.available_kv_cache_memory_bytes = (self.requested_memory -
profile_result.non_kv_cache_memory - cudagraph_memory_estimate_applied)`
(`vllm/v1/worker/gpu_worker.py:661-665`). Nothing in that computation reserves
room for the weight-transfer receive buffers, which are allocated lazily per
chunk at receive time — `packed_tensors[buffer_idx] = torch.empty(
packing_tensor_sizes[buffer_idx], ...)` immediately before the broadcast
(`vllm/distributed/weight_transfer/packed_tensor.py:253`, `:256`).

**Measured.** Phase 6's first 8B attempt failed server-side with
`CUDA out of memory. Tried to allocate 1.16 GiB ... 531.00 MiB free`, i.e. the
receive buffer for the 1.24 GB embedding tensor against a card whose KV cache had
already claimed the budget. The failure mode is worth recording: the OOM was
raised and logged *server-side* while the trainer, already inside the broadcast,
blocked — and the controller was never told. Running the same harness with
`--gpu-memory-utilization 0.75 --packed-num-buffers 1` completed
(`results/phase6-8b.json`: 6/6, hot update 4.2544 s, `UPDATING` 4.1287 s).

**Verdict: report.** Reserving headroom is a policy choice about who owns the
last gigabyte of a card — vLLM cannot know a deployment's transfer backend, chunk
size, or buffer count. A documented reservation (or a configurable one) is the
actionable ask; the artifact supplies the number.

---

## 4. Distribution plan

Nothing below has been filed. Order is by expected value per unit of maintainer
attention, and each item carries the evidence this repo already holds.

| Order | Form | Title / scope | Evidence carried |
|---|---|---|---|
| 1 | **Issue** | "No test generates across a weight update; the state_transitions weight-transfer helpers have no callers" — include the §2.2 step list and the §2.3 sketch, and offer to implement Lane A | inventory greps from §2.1, the RFC category-7 text, the `cached_tokens` oracle already used at `tests/entrypoints/serve/dev/rlhf/state_transitions/test_pause_resume.py:155` |
| 2 | **PR** | "Pass a timeout to the RL weight-update collective calls" — the §3.4 diff at `vllm/v1/engine/async_llm.py:1267`, `vllm/v1/engine/async_llm.py:1280`, `vllm/v1/engine/async_llm.py:1286` plus the route plumbing | `results/phase6-8b-with-deepkill.json` (`hung: true`, 120 s), `results/phase4d.json` (0.177 s at 125M), the `/collective_rpc` precedent at `vllm/entrypoints/serve/dev/rpc/api_router.py:42` |
| 3 | **Issue** | "`/pause` `clear_cache` is documented as ignored for `mode=keep` but honoured in all modes" — the §3.3 docstring fix, offered as a PR | the three call paths at `vllm/v1/engine/core.py:908-910` and `vllm/v1/engine/core.py:2021`, plus the undocumented MM-cache side effect at `vllm/v1/engine/async_llm.py:948` |
| 4 | **Issue** | "Which commit point should invalidate caches after an update?" — the §3.1 question, referencing #48762 and RFC exit criterion 7, with §2's lane as the way to decide | `vllm/v1/worker/gpu_worker.py:1488-1505`, `vllm/v1/engine/core.py:841-846` (the code's own statement of intent), `results/phase4c.json` (`kv_reused_across_weights: true`) |
| 5 | **Issue** | "Reserve headroom for weight-transfer receive buffers in the KV budget" — §3.6 | `vllm/v1/worker/gpu_worker.py:661-665`, `vllm/distributed/weight_transfer/packed_tensor.py:253`, the measured OOM text and the flags that avoided it (`results/phase6-8b.json`) |
| 6 | **Comment on RFC #48306 §2.2** | the §1.1 admission-vs-completion table, as evidence for the open contract question | `results/phase5.json` (`R-long` bound `rc-0`, engine `rc-1` at completion), `results/phase4a.json` (identical prefix to the pre-update baseline) |
| 7 | **Comment on RFC #48312** | offer the §2 Lane A implementation and the §3.5 lease finding for the tracker | `results/phase4b.json` (`AlreadyManagedEngineError` text), §3.5 |

Not proposed: a patch for the encoder/MM invalidation (§3.2) — already tracked
upstream, and a prior attempt (#48762) was closed, so the question in row 4 comes
first. Also not proposed: any change to per-request version binding; §1.1 is
evidence for a discussion, and the removal by #49040 was deliberate.

## 5. What this document is not

* **No patch has been submitted.** No branch, no PR, no diff against vLLM.
* **No maintainer has responded to any of it.** Everything labelled "new" is new
  to *this* project, not to upstream.
* **Anchors are pinned.** They resolve against
  `00b7847c8036b667742b4efb21aab1de51fd4721` and will rot as `main` moves. Check
  them with
  `python3 scripts/verify_anchors.py --vllm <checkout>`, which re-derives every
  backticked `file.py:LINE` and fails on any that does not resolve.
* **Two claims are second-hand, not verified here.** The GitHub state of #48762
  (closed, unmerged) and of #49040 (merged) cannot be checked from a checkout;
  both are taken from this repo's own records (`source-map-vllm-main.md`,
  `research/comments_48306.md`). The *code* consequences are verified: the caches
  are not invalidated by an update, and no per-request version field exists.
* **Two inherited phrasings were corrected, not dropped.** `reset_lora_state()`
  means "invalidates nothing" should read "invalidates only LoRA state" (§3.1);
  and `cache_salt` does key the prefix-KV hash, so the finding is about the other
  caches, not about `cache_salt` being ineffective (§3.2).
* **The measurements are one host, one pair of GPUs.** `opt-125m` and
  `Qwen3-8B`, TP=1, two RTX 3090s, and most phases exercised a single controller
  per engine. They establish that a failure mode is reachable and how it
  presents; they are not a survey of configurations.
