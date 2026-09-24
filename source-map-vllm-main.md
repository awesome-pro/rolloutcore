# RolloutCore × vLLM — source-level integration map

**Target:** `vllm-project/vllm` @ `main`
**Commit audited:** `00b7847c8036b667742b4efb21aab1de51fd4721` — *"[Perf] Use Conv3dLayer for MiniMax M3 patch embedding (#58512)"*, committer date `2026-09-24T07:41:36Z`
**Verification:** git worktree at `/Users/abhinandan/Desktop/vllm-learning/vendor/vllm-main`, fetched directly from `https://github.com/vllm-project/vllm.git`. `git ls-remote origin refs/heads/main` and `gh api repos/vllm-project/vllm/commits/main` both return this SHA; working tree clean; audited read-only.

> The sibling checkout `/Users/abhinandan/Desktop/vllm-learning/vendor/vllm` is a **stale fork** (`d90f0eade5`) and was not used. The worktree is a **shallow** clone (depth 1), so "not present" means *absent at this commit*, not *never existed*.

Every claim carries a `path:LINE` anchor plus the owning class/function. Anything unsubstantiated is marked **NOT PRESENT on main@00b7847c**.

---

## 0. Fourteen facts that decide the design

| # | Fact | Anchor |
|---|---|---|
| 1 | There is **no generic per-request/per-output metadata bag**. `RequestOutput.__init__`'s `**kwargs` is *logged and dropped*. | `vllm/outputs.py:155`, `:157-160` |
| 2 | `weight_version` is an **opaque caller-supplied string**, initialised to the literal `"default"`. | `vllm/v1/engine/core.py:137` |
| 3 | `weight_version` is **never attached** to `Request`, `EngineCoreRequest`, `EngineCoreOutput`, `CompletionOutput` or `RequestOutput`. | §2.4 |
| 4 | It is **never auto-incremented**; the only mutation is a verbatim assignment. | `core.py:1043` |
| 5 | `finish_weight_update` invalidates **no cache**. Its only worker-side cleanup is `reset_lora_state()`, which clears LoRA adapters when one is configured. | `async_llm.py:1284-1288`; `gpu_worker.py:1488-1505` |
| 6 | Prefix-cache keys contain **no weight generation**. Extra keys are only LoRA *name*, MM hashes, `cache_salt`, prompt-embed hashes. | `kv_cache_utils.py:610-646` |
| 7 | `cache_salt` is the **only caller-controllable identity input** to the cache key — and it **is** exposed on the OpenAI API. | `kv_cache_utils.py:632-634`; `chat_completion/protocol.py:456` |
| 8 | `pause(mode="wait")` **is** a true drain and **does** clear prefix/MM/encoder caches when `clear_cache=True`. | `core.py:877-882`, `:1984-2026` |
| 9 | `call_utility` / `call_utility_async` have **no timeout** — a drain that never completes blocks its HTTP caller forever. | `core_client.py:996-1002`, `:1236-1248` |
| 10 | `mode="keep"` **honours** `clear_cache`, contradicting the `/pause` docstring. | `core.py:2017-2021` vs `rlhf/api_router.py:46-47` |
| 11 | `release_kv_cache_memory()` is a **KV-only** eviction requiring a completed pause; it does not touch weights. | `core.py:984-1002` |
| 12 | vLLM has a **first-class plugin API** for adding HTTP routes and worker RPCs without forking. | `vllm/plugins/endpoint_plugins/interface.py:44`; `vllm/plugins/__init__.py:93` |
| 13 | `trace_decode_token_ids` (replay fixed tokens, compute **real** logprobs) is landed; **requires Model Runner V2**. | `sampling_params.py:374`; `config/vllm.py:1266` |
| 14 | R3 routed-experts is landed end-to-end and reaches the OpenAI response as base64. | `outputs.py:64`; `chat_completion/serving.py:1082-1103` |

---

## 1. Request admission and lifecycle

### 1.1 HTTP entry

| Step | Anchor |
|---|---|
| App assembly `build_app()` → `register_api_routers()` | `vllm/entrypoints/launchers/app.py:19`, `:45` |
| Router registration; **dev routers gated on `envs.VLLM_SERVER_DEV_MODE`** | `vllm/entrypoints/launchers/api_server/routers.py:12`, `:34-38` |
| Dev router set (cache, rlhf, rpc, server_info, sleep) | `vllm/entrypoints/serve/__init__.py:43-69` |
| Generate routers `if "generate" in supported_tasks` | `routers.py:39-44` → `vllm/entrypoints/generate/api_router.py:21` |
| `POST /v1/chat/completions` route | `vllm/entrypoints/openai/chat_completion/api_router.py:41-54` |
| `POST /v1/completions` route | `vllm/entrypoints/openai/completion/api_router.py:35-47` |
| Handler class | `vllm/entrypoints/openai/chat_completion/serving.py:118` `OpenAIServingChat`; `create_chat_completion:244`, `_create_chat_completion:259` |
| Rendering (chat) | `serving.py:276` → `:221 render_chat_request` → `vllm/renderers/online_renderer.py:172 render_chat` |
| **Engine call** | `chat_completion/serving.py:369-386` `self.engine_client.generate(engine_input, sampling_params, sub_request_id, ...)`; completion equivalent `completion/serving.py:203-212` |
| **Pre-flight admission hook** | `vllm/entrypoints/generate/base/serving.py:174 _preflight` → `:186 self.engine_client.check_admission(n)`; `EngineClient.check_admission` `vllm/engine/protocol.py:71-87` (concrete no-op default); `AsyncLLM.check_admission` `async_llm.py:307-363` raises `QueueOverflowError` / `MaxQueuedTokensError` → HTTP 503 |
| Internal id allocation | `input_processor.py:305-322 assign_request_id` → `request.external_req_id = request.request_id` (`:314`), `request.request_id = f"{external}-{random_uuid():.8}"` (`:322`) |
| API-level id | `chat_completion/serving.py:282-284` `f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"`; N>1 sub-ids `:307-309` |
| Streaming | `chat_completion/api_router.py:77-80` `StreamingResponse(with_sse_keep_alive(generator, ...))`; `vllm/entrypoints/serve/utils/sse_keep_alive.py:21-33` |

> `vllm/entrypoints/openai/api_server.py` is a **deprecation re-export shim** (`:6-27`); it no longer defines `run_server` / `build_async_engine_client`. There is no `serving_chat.py` / `serving_completion.py`.

### 1.2 Input processing — the unification point

**`vllm/v1/engine/processor.py` does NOT exist on main@00b7847c.** The class is `InputProcessor` in `vllm/v1/engine/input_processor.py:41`.

| Item | Anchor |
|---|---|
| `InputProcessor.process_inputs(...)` | `input_processor.py:324-339` |
| Async variant is a wrapper, not a method | `input_processor.py:82-84` `self.process_inputs_async = make_async(self.process_inputs, executor=...)` |
| Parameter validation | `:340-341` `_validate_params` / `_validate_lora` |
| Prompt-length gate (the only one) | `_validate_prompt_len():492-537`, raises `VLLMValidationError` at `:523-537` |
| `EngineCoreRequest` construction | `:473-490` |
| `cache_salt` passthrough | `:483 cache_salt=decoder_input.get("cache_salt")` |
| Trace-replay normalisation | `:205-227`, invoked `:430-431`; hard error without `enable_trace_replay` `:168-176` |

### 1.3 `EngineCoreRequest` — `vllm/v1/engine/__init__.py:109`

`msgspec.Struct`, `array_like=True`, `omit_defaults=True`, `gc=False`:

`request_id:115`, `prompt_token_ids:116`, `mm_features:117`, `sampling_params:118`, `pooling_params:119`, `arrival_time:120`, `lora_request:121`, `cache_salt:122`, `data_parallel_rank:123`, `prompt_embeds:124`, `prompt_is_token_ids:130`, `client_index:134`, `current_wave:139`, `priority:140`, `trace_headers:142`, `resumable:143`, `external_req_id:149`, `reasoning_ended:151`, `reasoning_parser_kwargs:152`, `abort_immediately:158`, `session_id:160`, `kv_hints:161`; `params` property `:163-169`.

> **No metadata bag, no weight version.** `trace_headers: Mapping[str, str]` (`:142`) is the only opaque caller bag and is consumed only by tracing.
> `client_index`, `current_wave`, `external_req_id`, `reasoning_ended`, `reasoning_parser_kwargs`, `abort_immediately` are **not** set at `:473-490`; they are filled later.

### 1.4 API-server engine — `AsyncLLM` (`vllm/v1/engine/async_llm.py:80`)

| Item | Anchor |
|---|---|
| `input_processor` | `:155` |
| `output_processor` | `:165-171` |
| `engine_core` client | `:178-186` `EngineCoreClient.make_async_mp_client(...)` |
| `add_request(...)` | `:372-391` |
| Pre-built `EngineCoreRequest` accepted (deprecated) | `:429-447` |
| `process_inputs_async` path | `:464` |
| `assign_request_id` | `:485` |
| Output handler task (lazy start) | `:490`, `:811-872` (`:815 await engine_core.get_output_async()`) |
| `generate(...)` | `:664-683`; pull loop `:730-740`; `q.close()` `:788-789` |
| **There is no `AsyncLLM.output_queue`** | per-request `RequestOutputCollector` instead — `output_processor.py:51-111` |
| Abort on disconnect | `:745-750` (`asyncio.CancelledError` / `GeneratorExit`) → `abort(q.request_id, internal=True)` |
| `abort(...)` | `:874-885` |

`RequestOutputCollector` — `vllm/v1/engine/output_processor.py:51`: `__init__:59-63` (`self.aggregate = output_kind == RequestOutputKind.DELTA`), `put():67-81` (merges via `self.output.add(...)` when the producer outruns the consumer), `get():83-91`, `get_nowait():93-101`, `close():103-106`.

### 1.5 Core loop

| Item | Anchor |
|---|---|
| `EngineCore.add_request` | `core.py:489-533`; `:529 self.scheduler.add_request(request)`; `:530-533` honours `abort_immediately` |
| `EngineCore.abort_requests` | `core.py:535-540` → `scheduler.finish_requests(ids, FINISHED_ABORTED)` |
| `EngineCore.step` | `core.py:633-662` (`:643 schedule`, `:644 execute_model`, `:657 update_from_output`) |
| `EngineCore.step_with_batch_queue` | `core.py:673-786` |
| `_process_aborts_queue` | `core.py:788-796` |
| `EngineCore.preprocess_add_request` | `core.py:1049-1071` — `:1063 Request.from_engine_core_request(...)`, `:1064-1070` grammar init |
| `EngineCoreProc` | `core.py:1088`; `input_queue:1107`, `output_queue:1108` |
| `run_busy_loop` | `core.py:1469-1481` |
| `_process_input_queue` | `core.py:1496-1524` — idle branch `:1499-1501` fires idle callbacks |
| `_process_engine_step` | `core.py:1526-1542` |
| `_handle_client_request` | `core.py:1597-1629`; UTILITY `:1610-1623` |
| `process_input_sockets` | `core.py:1754-1854`; **ABORT pushed to both `aborts_queue` and `input_queue`** `:1846-1851` |
| `process_output_sockets` | `core.py:1856-1916` |
| `RequestStatus` | `vllm/v1/request.py:370-397` |

### 1.6 Scheduler — admission and state machine

| Item | Anchor |
|---|---|
| `Scheduler` | `vllm/v1/core/sched/scheduler.py:79` |
| queues | `:212 self.waiting`, `:214 self.skipped_waiting`, `:215 self.running`, `:221 self.finished_req_ids` |
| **`Scheduler.add_request` (engine-side admission)** | `:2485-2511`; `:2502 _enqueue_waiting_request`, `:2503 self.requests[...]`, `:2508-2509 connector.on_new_request(request)`, `:2510-2511 QUEUED event` |
| blocked-status routing | `_is_blocked_waiting_status:2318-2324`, `_enqueue_waiting_request:2326-2330`, `_select_waiting_queue_for_scheduling:2332-2342` |
| `schedule()` | `:556-576`; RUNNING phase `:623-625`; WAITING phase `:867-897` |
| KV allocation failure → stays WAITING | `:1213-1234` (`if new_blocks is None: break`) |
| async KV load → `WAITING_FOR_REMOTE_KVS` | `:1262-1293` |
| → RUNNING | `:1295-1318` |
| `_preempt_request` | `:1538-1581` |
| `finish_requests` (the only external finish sink) | `:2513-2575` |
| `_free_request` (completion path) | `:2577-2606`; `:2582-2583 aux_output_connector.request_finished`, `:2585 _connector_finished`, `:2592-2594 ec_connector.request_finished` |
| `_connector_finished` | `:2878-2931` |

`RequestStatus` (`vllm/v1/request.py:370-386`): `WAITING:373`, `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:374`, `WAITING_FOR_REMOTE_KVS:375`, `WAITING_FOR_STREAMING_REQ:376`, `RUNNING:377`, `PREEMPTED:378`, `FINISHED_STOPPED:381`, `FINISHED_LENGTH_CAPPED:382`, `FINISHED_ABORTED:383`, `FINISHED_IGNORED:384`, `FINISHED_ERROR:385`, `FINISHED_REPETITION:386`; `is_finished():391-393`.

> `FINISHED_IGNORED` is declared (`:384`) and mapped (`:408`) but **never assigned** anywhere in `vllm/**/*.py`. Over-long prompts are rejected earlier at `input_processor.py:523-537`.

### 1.7 Admission / completion hook points — what actually exists

**There is no generic admission or completion callback on `Request`, and no request-level hook registry.** (`grep callback` over `v1/request.py`, `v1/core/sched/scheduler.py`, `v1/engine/input_processor.py` → 0 hits.)

What exists:

| Hook | Anchor | Kind |
|---|---|---|
| KV-connector admission | `scheduler.py:2508-2509` `connector.on_new_request(request)` | connector |
| Completion fan-out | `scheduler.py:2582-2594` (aux-output / KV / EC connectors) | connector |
| Preemption also fires aux-output finish | `scheduler.py:1554-1555` | connector |
| HTTP-level pre-flight reject | `generate/base/serving.py:186` → `AsyncLLM.check_admission` `async_llm.py:307-363` | API layer |
| Engine-idle callbacks | `core.py:2025`, fired `:1544-1547` | engine |
| Worker/engine extension | `vllm.general_plugins` entry point, loaded in all processes | plugin |
| HTTP extension | `vllm.endpoint_plugins` entry point, API server only | plugin |

---

## 2. Current weight-version state and hooks

### 2.1 The state

| Location | Anchor | Behaviour |
|---|---|---|
| EngineCore field | `vllm/v1/engine/core.py:136-137` | `# Opaque weight version supplied by the caller.` / `self._weight_version = "default"` |
| Setter / getter | `core.py:1042-1047` | verbatim assignment; `get_weight_version` docstring *"Return the latest committed weight version."* |
| AsyncLLM | `async_llm.py:1290-1296` | `update_weight_version`, `get_weight_version` |
| LLMEngine | `llm_engine.py:447-452` | sync wrappers |
| Core clients (base decls) | `core_client.py:213`, `:216`, `:222`, `:225` | `raise NotImplementedError` |
| `InprocClient` | `core_client.py:423-427` | direct call |
| `MPClient` | `core_client.py:1059-1063` | `call_utility` |
| `AsyncMPClient` | `core_client.py:1306-1310` | `call_utility_async` |
| Offline `LLM` | `entrypoints/llm.py:915-921` | `update_weight_version` / `get_weight_version` |
| Trainer-side HTTP client | `weight_transfer/clients.py:89-93` | POSTs `{"weight_version": ...}` only when not `None` |
| Trainer-side Ray client | `weight_transfer/clients.py:123-130` | calls `finish_weight_update()` with **no** arg, then a separate `update_weight_version(weight_version)` per handle |
| Protocol declaration | `vllm/engine/protocol.py:307-313` | `update_weight_version`, `get_weight_version` |

`_weight_version` appears **only** at `core.py:137`, `:1043`, `:1047`. It is never read by the scheduler, `Request`, output processing, prefix-cache or KV logic.

### 2.2 The write hook

`AsyncLLM.finish_weight_update` — `async_llm.py:1284-1288`:

```python
await self.collective_rpc("finish_weight_update")
if weight_version is not None:
    await self.update_weight_version(weight_version)
```

- Version is written **after** the worker finalize RPC returns — matching RFC #48306's convention *"Version increments when `finish_weight_update` returns."*
- Worker side — `gpu_worker.py:1488-1505`: `weight_transfer_engine.finish_weight_update()` (`:1499`), `reset_weight_update_target()` (`:1500`), clear `_weight_update_active` (`:1501`), then `if not self._weight_update_is_draft: self.model_runner.reset_lora_state()` (`:1504-1505`). Comment at `:1503`: *"Weight transfer bypasses GPUModelRunner.reload_weights()."* **Nothing else — no cache invalidation.**
- `reset_lora_state` — `vllm/v1/worker/lora_model_runner_mixin.py:29-39`: removes all adapters and resets `LogitsProcessorWithLoRA` sharded→full mappings.

### 2.3 HTTP surface (all `VLLM_SERVER_DEV_MODE=1`)

`vllm/entrypoints/serve/dev/rlhf/api_router.py`:

| Route | Line | Effect |
|---|---|---|
| `POST /pause?mode=&clear_cache=&wait_for_inflight_requests=` | `:29-73` | `pause_generation(...)`; 400 on `ValueError` (`:63-67`) |
| `POST /resume` | `:76-92` | `resume_generation()` |
| `POST /abort_requests` | `:95-136` | body `{request_ids: [...]}`; empty/missing ⇒ abort all tracked ids (`:118-126`) |
| `GET /is_paused` | `:139-153` | `{"is_paused": bool}` |
| `POST /init_weight_transfer_engine` | `:156-171` | body `{init_info: {...}}` |
| `POST /start_weight_update` | `:174-177` | |
| `POST /start_draft_weight_update` | `:180-183` | |
| `POST /update_weights` | `:186-201` | body `{update_info: {...}}` |
| `POST /finish_weight_update` | `:204-210` | body `{weight_version: str \| None}` (embedded) |
| `POST /update_weight_version` | `:213-219` | body `{new_version: str}` (embedded); sets version **without** touching weights |
| `GET /weight_info` | `:222-225` | `{"weight_version": str}` |
| `GET /get_world_size?include_dp=` | `:228-248` | TP·PP·DP or TP·PP |

### 2.4 Definitive negative answers

- **`weight_version` attached to any request/output object?** **NO.** `grep -rl weight_version vllm/ --include=*.py` → 55 matches in exactly 9 files: `v1/engine/async_llm.py`, `v1/engine/core_client.py`, `v1/engine/llm_engine.py`, `v1/engine/core.py`, `distributed/weight_transfer/clients.py`, `distributed/weight_transfer/base.py`, `entrypoints/llm.py`, `entrypoints/serve/dev/rlhf/api_router.py`, `engine/protocol.py`. Zero occurrences in `vllm/v1/request.py`, `vllm/outputs.py`, `vllm/v1/engine/__init__.py`, `vllm/v1/engine/output_processor.py`.
- **Auto-incremented?** **NO.** Sole mutation is verbatim assignment (`core.py:1043`); initial value is the literal `"default"` (`core.py:137`); `grep "_weight_version *[+-]="` → 0 matches. Every entry originates from a caller argument.
- **Per-DP-rank?** `AsyncMPClient.set_weight_version_async` (`core_client.py:1306-1310`) routes through `call_utility_async`, which `DPLBAsyncMPClient` **overrides to broadcast** across `self.core_engines` (`core_client.py:1636-1645`) but which for `DPAsyncMPClient` (external LB) targets only its single colocated engine. So: *one engine per client* under external LB — the version must be set per instance.
- **Commit certificate / read-your-writes?** **NO.** `GET /weight_info` returns the last value *set*; there is no per-rank reconcile proof.

### 2.5 Why it is caller-supplied — the PR that made it so

`vllm-project/vllm#49040` — *"[Core][Frontend] Add weight version tagging for RL rollouts"* — **merged 2026-07-28**. Per the maintainer thread on RFC #48306 (comment by `hongzhi-gao`, 2026-09-16), #49040 *"implemented the query/update APIs, but intentionally removed binding a version to `Request`/`RequestOutput` after review noted that one request may span multiple weight versions."*

**Consequence:** per-request version stamping is an *open, deliberately deferred* contract, not an oversight. An MVP must not assume it.

---

## 3. Native weight-update APIs and backends

### 3.1 Worker-side interface — `vllm/distributed/weight_transfer/base.py`

| Symbol | Line | Notes |
|---|---|---|
| `layerwise_groups(names)` | `:51` | partitions flat names into per-decoder-layer groups |
| `materialize_full_tensor(tensor)` | `:107` | `full_tensor()` for FSDP DTensor, else identity |
| `ParamMeta` | `:120` | `name`, `dtype`, `shape` |
| `WeightSource(ABC)` | `:128` | `metadata():152` (abstract), `__iter__:165` (abstract), `held_names():169`, `groups():200`, `iter_groups():214` |
| `ModuleSource(WeightSource)` | `:244` | `__init__:253`, `metadata():256`, `__iter__:262` |
| `WeightTransferInitInfo(ABC)` | `:269` | empty base |
| `TrainerInitInfo` | `:276` | `backend: ClassVar[str]:293`, `rank:295`, `is_sender:305`, `__init_subclass__:297` |
| `WeightTransferUpdateInfo(ABC)` | `:311` | empty base |
| `WeightTransferInitRequest` | `:319` | `init_info: dict[str, Any]:322` |
| `WeightTransferUpdateRequest` | `:326` | `update_info: WeightTransferUpdatePayload:329`; payload type `:30` = `dict \| list[dict]` |
| **`WeightTransferEngine(ABC, Generic[...])`** | `:332` | `init_info_cls:352`, `update_info_cls:353`, `supports_draft_weight_update:355`, `defers_processing:357` |
| `__init__(config, vllm_config, device, model)` | `:382` | |
| `set_weight_update_target` / `reset_weight_update_target` | `:407` / `:416` | |
| `parse_init_info` / `parse_update_info` | `:421` / `:441` | base-only impls |
| **`init_transfer_engine(init_info)`** | `:462` | **abstract** — note the name: *not* `init_weight_transfer_engine` |
| `start_weight_update` | `:473` | abstract |
| `finish_weight_update` | `:483` | abstract |
| `update_weights(update_info: dict)` | `:491` | **concrete**: `parse_update_info` → `receive_weights` → `torch.accelerator.synchronize()` |
| `receive_weights(update_info)` | `:505` | abstract |
| `shutdown` | `:516` | abstract |
| `drain_pending()` | `:372` | no-op default; companion to `defers_processing` |
| `VLLMWeightSyncClient(Protocol)` | `:524` | `init_weight_transfer_engine:542`, `start_weight_update:544`, `update_weights:546`, `finish_weight_update(weight_version):548` |
| `TrainerWeightTransferEngine(ABC, Generic[...])` | `:551` | `init_info_cls:583`, `__init__:585`, `trainer_init:600`, `send_weights:617`, `shutdown:625` |

> **Name trap:** `init_weight_transfer_engine` exists only at the *outer* layers — `Worker` (`gpu_worker.py:1394`), `AsyncLLM` (`async_llm.py:1252`), `LLM` (`llm.py:869`), `EngineClient` (`protocol.py:285`), sync clients (`clients.py:78`, `:106`), HTTP route (`api_router.py:156`). The engine method is `init_transfer_engine` (`base.py:462`).

### 3.2 Backends — registry in `vllm/distributed/weight_transfer/factory.py`

| Key | Worker engine (module) | Trainer engine | Registered |
|---|---|---|---|
| `nccl` | `NCCLWeightTransferEngine` (`nccl_engine.py:113`) | `NCCLTrainerWeightTransferEngine` (`:237`) | `factory.py:222-226`, `:248-252` |
| `ipc` | `IPCWeightTransferEngine` (`ipc_engine.py:121`) | `IPCTrainerWeightTransferEngine` (`:252`) | `factory.py:228-232`, `:254-258` |
| `sparse_nccl` | `SparseNCCLWeightTransferEngine` (`sparse_nccl_engine.py:113`) | `SparseNCCLTrainerWeightTransferEngine` (`:193`) | `factory.py:234-238`, `:260-264` |
| `sharded_rdt` | `ShardedRDTWeightTransferEngine` (`sharded_rdt_engine.py:267`) | `ShardedRDTTrainerWeightTransferEngine` (`sharded_rdt_trainer.py:800`) | `factory.py:240-244`, `:266-270` |

**A plain `"rdt"` backend does NOT exist on main@00b7847c** — only `sharded_rdt`. `RdtRouter` (`sharded_rdt_common.py:102`) is a routing helper, not an engine.

Factories: `WeightTransferEngineFactory.create_engine()` `factory.py:85-121` (unknown backend → `ValueError` `:110-113`); `WeightTransferTrainerFactory.trainer_init()` `:167-215` (dispatches on `init_info.backend` `:197-204`). Both extensible at runtime via `register_engine(name, cls)` `:41-82` / `:137-164`. Re-exports `vllm/distributed/weight_transfer/__init__.py:19-22`.

Per-backend notes:

- **NCCL** — init info `NCCLWeightTransferInitInfo` (`nccl_common.py:71`, shared with sparse): `master_address:100`, `master_port:101`, `rank_offset:102`, `world_size:103`, `nccl_unique_id_b64:104`, `packed:105`, `packed_buffer_size_bytes:106`, `packed_num_buffers:107`. Update info `NCCLWeightTransferUpdateInfo` (`nccl_engine.py:85`): `names:94`, `dtype_names:95`, `shapes:96`. `init_transfer_engine:143`, `start_weight_update:160` (`initialize_layerwise_reload`), `finish_weight_update:168` (`finalize_layerwise_reload`), `receive_weights:176` (packed `:209-217` via `packed_nccl_broadcast_consumer(..., post_unpack_func=self.model.load_weights)`, unpacked `:225-228` `self.model.load_weights([(name, weight)])`).
- **IPC** — `IPCWeightTransferInitInfo:37` (`packed:45`), `IPCTrainerInitInfo:49`, `IPCWeightTransferUpdateInfo:64` (`names:67`, `dtype_names:68`, `shapes:69`, `ipc_handles:70`, `ipc_handles_pickled:74`, `tensor_sizes:76`; `__post_init__:80` requires handles XOR pickled handles; unpickling gated on `envs.VLLM_ALLOW_INSECURE_SERIALIZATION:87`). `receive_weights:183` → `self.model.load_weights(weights)` `:246`.
- **sparse_nccl** — `SparseWeightPatch:55`, `SparseNCCLTrainerInitInfo:66`, `SparseNCCLWeightTransferUpdateInfo:81`. `supports_draft_weight_update = False:125`. `start_weight_update:143` and `finish_weight_update:147` are **explicit no-ops** (docstring: *"sparse patches are applied in place, no layerwise reload"*). Reuses `NCCLWeightTransferInitInfo` (`:123`). Applies via `load_checkpoint_weight_patches` `:186` — **not** `model.load_weights`.
- **sharded_rdt** — `ShardedRDTWeightTransferInitInfo:160`, `ShardedRDTWeightTransferUpdateInfo:256` (**deliberately empty**), `defers_processing = True:293`, `supports_draft_weight_update = False:296`; requires `distributed_executor_backend == "ray"` (`:315-322`); `_bake():811` drives a meta dry-run `model.load_weights(...)` `:854`; `update_weights` **is overridden** `:665` (no device sync); `drain_pending():1051`.

### 3.3 Config — `vllm/config/weight_transfer.py:9`

```python
@config
class WeightTransferConfig:
    backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt"] | str = "nccl"   # :12
```

**Exactly one field.** No per-backend sub-config, no `enabled` flag, no version/counter field. `@config` is `vllm/config/utils.py:52` (pydantic dataclass wrapper, `extra="forbid"`).

Wiring: `VllmConfig.weight_transfer_config: WeightTransferConfig | None = None` (`vllm/config/vllm.py:452`); `vllm/config/__init__.py:65`, `:153`; `arg_utils.py:809` (field), `:843-846` (dict → object), `:1787` (CLI `--weight-transfer-config`), `:2734` (VllmConfig construction). Backend is validated lazily at `create_engine` time, not by the `Literal` (the annotation is `Literal[...] | str`). Ray + `sharded_rdt` needs `enable_tensor_transport` (`ray_executor_v2.py:355-363`).

### 3.4 Engine → worker dispatch chain

```
POST /update_weights  {"update_info": {...}}
  → EngineClient.update_weights(WeightTransferUpdateRequest)      protocol.py:299
  → AsyncLLM.update_weights()                                     async_llm.py:1273-1282
  → collective_rpc("update_weights", kwargs={"update_info": ...}) async_llm.py:1280
  → EngineCoreClient.collective_rpc_async                         core_client.py:1329
  → call_utility_async("collective_rpc", method, timeout, args, kwargs)  core_client.py:1336
  → UTILITY message over ZMQ                                      core_client.py:1236-1248
  → EngineCoreProc._handle_client_request (UTILITY)               core.py:1610-1623
  → EngineCore.collective_rpc                                     core.py:1033-1040
  → Executor.collective_rpc                                       vllm/v1/executor/abstract.py:185-218
  → Worker.update_weights(update_info)                            gpu_worker.py:1451-1486
  → weight_transfer_engine.update_weights(local_update_info)      gpu_worker.py:1482
```

`Worker` session state machine: construction in `load_model` (`gpu_worker.py:546-552`), `_check_weight_transfer_engine():1387`, `init_weight_transfer_engine:1394`, `start_weight_update:1408`, `start_draft_weight_update:1418`, `_start_weight_update:1425-1449` (rejects re-entry `:1435-1439`, rejects draft when unsupported `:1429-1433`), `update_weights:1451-1486` (requires active session `:1467-1470`; **list payload indexed by `data_parallel_index * world_size + rank`** `:1474-1479`), `finish_weight_update:1488-1505`, `supports_draft_weight_updates:1031-1042`, `_set_draft_weight_update_target:1044-1062`.

### 3.5 Offline `LLM` surface

`vllm/entrypoints/llm.py`: `init_weight_transfer_engine:869` (accepts dict or dataclass), `start_weight_update:886`, `start_draft_weight_update:890`, `update_weights:894`, `finish_weight_update:909` (`:911` RPC then `:913 set_weight_version`), `update_weight_version:915`, `get_weight_version:919`. Also `sleep:808`, `release_kv_cache_memory:833`, `wake_up:841`.

> `start_draft_weight_update` is **NOT PRESENT** on `LLMEngine`, `EngineCoreClient`, `InprocClient`, `MPClient` or `AsyncMPClient` — only on `AsyncLLM:1269`, `LLM:890`, `EngineClient:295`, `Worker:1418`, route `:180`.

---

## 4. Drain / pause / sleep / wake path

### 4.1 Types

- `PauseMode = Literal["abort", "wait", "keep"]` — `vllm/v1/engine/__init__.py:32`
- `PauseState(IntEnum)` = `UNPAUSED=0`, `PAUSED_NEW=1`, `PAUSED_ALL=2` — `vllm/v1/core/sched/interface.py:24-35`
- `FinishReason` = `STOP/LENGTH/ABORT/ERROR/REPETITION` — `vllm/v1/engine/__init__.py:48-69`

### 4.2 `EngineCore.pause_scheduler` (in-proc) — `core.py:884-912`

- validates mode (`:900-901`); **`mode="wait"` raises `ValueError("'wait' mode can't be used in inproc-engine mode")`** (`:902-903`)
- `abort` → `scheduler.finish_requests(None, FINISHED_ABORTED)` (`:906`)
- `pause_state = PAUSED_ALL if mode == "keep" else PAUSED_NEW` (`:908`)
- `_finish_pause(clear_cache)` unconditionally (`:910`)

`InprocClient.sleep` likewise rejects `mode="wait"` (`core_client.py:405-407`).

### 4.3 `EngineCoreProc.pause_scheduler` (multiprocess) — `core.py:1984-2026`

- `abort` → finish all + `_send_abort_outputs` (`:2011-2015`)
- `keep` → `PAUSED_ALL`; `wait`/`abort` → `PAUSED_NEW` (`:2017`)
- idle now (`_pause_complete()` = `not has_work()`, `:2028-2033`) → finish immediately, `return None` (`:2020-2022`)
- otherwise register idle callback and return a `Future` (`:2024-2026`); callbacks fire from `_process_input_queue` when `not has_work()` (`:1499-1501`, `_notify_idle_state_callbacks():1544-1547`)
- `has_work()` = `engines_running or scheduler.has_requests() or batch_queue` (`:1457-1463`). **`has_work` is defined only on `EngineCoreProc`, not on base `EngineCore`.**

**The actual drain predicate** is `Scheduler.get_num_unfinished_requests` — `scheduler.py:2665-2675`:

```python
if self._pause_state == PauseState.PAUSED_ALL:  return 0        # :2666-2667
if self._pause_state == PauseState.PAUSED_NEW:  return len(self.running)   # :2668-2669
num_waiting = len(self.waiting) + len(self.skipped_waiting) - self.num_waiting_for_streaming_input
```

With `PAUSED_NEW` (i.e. `mode="wait"`), `has_work()` stays `True` while RUNNING requests remain → the engine keeps stepping → they drain naturally → then `has_work()` goes `False` → idle callbacks fire → the pause `Future` resolves. `PAUSED_ALL` (i.e. `mode="keep"`) is idle immediately by construction.

The other three pause gates: `token_budget = 0` under `PAUSED_ALL` (`scheduler.py:580-582`); the RUNNING loop `while req_index < len(self.running) and token_budget > 0` (`:625`); the WAITING gate `if not preempted_reqs and self._pause_state == PauseState.UNPAUSED` (`:868`) — so **both** `PAUSED_NEW` and `PAUSED_ALL` stop admitting WAITING requests.

**This `Future` is the drain primitive.** It crosses the process boundary because `EngineCoreProc._invoke_utility_method` defers the utility output until a returned `Future` completes (`core.py:1659-1672`, esp. `:1664-1668`).

> **DP caveat:** `DPEngineCoreProc._pause_complete` **always returns `False`** (`core.py:2139-2155`), forcing every rank into a two-phase DP consensus checkpoint (it sets `pending_pause = True` / `engines_running = True` at `:2152-2153`). `DPEngineCoreProc.resume_scheduler` raises `RuntimeError("resume_scheduler called while pause is still in flight. ...")` (`:2173-2179`). So under DP, pause/resume is a *collective* two-phase protocol, not an independent per-rank call.

Chain: `AsyncLLM.pause_generation` (`async_llm.py:914-957`) → `engine_core.pause_scheduler_async(mode, clear_cache)` (`core_client.py:1262-1265`) → `call_utility_async("pause_scheduler", mode, clear_cache)`.

`AsyncLLM.pause_generation` also calls `self.renderer.clear_mm_cache_async()` when `clear_cache=True` (`:948-949`) and sleeps 20 ms to flush trailing outputs (`:957`).

> **Hazard (§0 fact 9):** `call_utility_async` has **no timeout** (`core_client.py:1236-1248`), and `SyncMPClient.call_utility` blocks on `future.result()` with no deadline (`:996-1002`). A drain that never reaches `not has_work()` blocks the HTTP handler indefinitely. `mode="wait"` is the worst case: `PAUSED_NEW` stops admitting but keeps stepping, so a request that cannot finish (e.g. a stop condition that never fires) hangs the pause forever.

### 4.4 `_finish_pause` and `_reset_caches` — `core.py:861-882`

```python
def _reset_caches(self, reset_running_requests=True, reset_connector=True):
    if not self.reset_prefix_cache(reset_running_requests=..., reset_connector=...):
        raise RuntimeError("Failed to reset the KV connector cache.")
    self.reset_mm_cache()
    self.reset_encoder_cache()

def _finish_pause(self, clear_cache: bool):
    self.model_executor.collective_rpc("synchronize_device")
    if clear_cache:
        self._reset_caches()
```

### 4.5 `sleep` / `wake_up` / `resume` — `core.py:914-1006`

- `resume_scheduler()` `:914-916` → `set_pause_state(UNPAUSED)` **only**.
- `is_scheduler_paused()` `:918-920`.
- `sleep(level, mode)` `:922-958`: `clear_prefix_cache = level >= 1` (`:936`); `pause_scheduler(mode, clear_cache=clear_prefix_cache)` (`:937`); `level < 1` returns the pause future (`:938-939`); `level >= 1` chains `model_executor.sleep(level)` onto it via `pause_complete` (`:947-958`). Docstring `:925-933`: **level 0 = pause scheduling only, no GPU memory change; level 1 = offload weights to CPU + discard KV; level 2 = discard all GPU memory.**
- `wake_up(tags)` `:960-982`: `"scheduling"` stripped from tags (`:970-972`); `model_executor.wake_up(tags)` (`:975`); `fully_awake = not model_executor.is_sleeping` → `resume_scheduler()` (`:979-981`).
- `is_sleeping()` `:1004-1006` = `is_scheduler_paused() or model_executor.is_sleeping`.

### 4.6 Worker-side sleep mechanism

| Item | Anchor |
|---|---|
| `Executor.sleep(level)` | `vllm/v1/executor/abstract.py:347-359` — `collective_rpc("sleep", kwargs=dict(level=level))`, then `sleeping_tags |= SLEEP_TAGS` |
| `Executor.wake_up(tags)` | `:359-382` — validates every tag against `sleeping_tags` before RPC (`:363-369`) |
| `Executor.discard(tags)` | `:384-398` — `tags_to_discard = set(tags) - self.sleeping_tags` (`:385`) |
| `SLEEP_TAGS` | `:36` `frozenset(("weights", "kv_cache"))` |
| `Executor.is_sleeping` | `:343-345` = `bool(self.sleeping_tags)` |

> **Asymmetry worth knowing:** `Executor.sleep` marks **both** tags asleep regardless of level (`:354`), so after `sleep(level=1)` the executor reports `is_sleeping == True` and `wake_up(tags=["kv_cache"])` is accepted even though the KV cache was *discarded*, not offloaded — waking it merely re-maps empty memory.
| `SleepModeBackend(ABC)` | `vllm/device_allocator/sleep_mode_backend.py:37`; `suspend(level):53`, `resume(tags):63`, `state():71`, `discard(tags):76` |
| capability probes | `is_supported:86`, `preserves_communicators:91`, `preserves_compiled_artifacts:97`, `preserves_graphs_with_communicators:103`, `supports_durable_storage:110` |
| **`CuMemBackend.suspend` — the level switch is one line** | `:127-132`, esp. `:132 allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())` |
| `CuMemBackend.resume` / `discard` | `:134-140` / `:142-147` |
| `SleepModeBackendFactory` | `:155`; `register_backend:166`, `get_backend_class:178`, `create_backend:189` |
| `SleepModeState` | `:34` `Literal["RUNNING","SUSPENDED","RESUMING"]` |
| `GPUWorker.sleep(level)` | `vllm/v1/worker/gpu_worker.py:270-307` — level 2 pre-copies all params (`:276-283`) and buffers (`:284-287`) to CPU; then `sleep_mode_backend.suspend(level)` (`:289`). **No `mode` parameter** — mode is engine-level only |
| `GPUWorker.wake_up(tags)` | `:309-336` — resume (`:310`), restore level-2 params/buffers only when `tags is None or "weights" in tags` (`:314-331`) |
| `GPUWorker.discard(tags)` | `:335-336` |
| `CuMemAllocator.sleep(offload_tags)` | `vllm/device_allocator/cumem.py:227-...` |
| `CuMemAllocator.discard(tags)` | `:294-322` |
| `CuMemAllocator.wake_up(tags)` | `:325-...`; re-maps the **same virtual addresses** via `create_and_map(handle)` (`:344`); docstring `:327-328`: *"the rest of the data will have empty memory"* |

> **KV is never re-allocated on wake.** `EngineCore._initialize_kv_caches` runs only from `EngineCore.__init__` (`core.py:151`) → `model_executor.initialize_from_config` (`core.py:358`) → `GPUWorker.initialize_from_config` (`gpu_worker.py:774`) → `initialize_kv_cache` under the `"kv_cache"` mem-pool tag (`:794`). Wake restores the address range, not the contents.

### 4.7 HTTP surface — `vllm/entrypoints/serve/dev/sleep/api_router.py`

`POST /sleep?level=&mode=` `:21-27`; `POST /release_kv_cache_memory` `:30-33`; `POST /wake_up?tags=` `:36-44`; `GET /is_sleeping` `:47-50`. Sleep endpoints additionally require `--enable-sleep-mode` (`docs/features/sleep_mode.md:106`).

### 4.8 Documented semantics vs. code — two discrepancies

1. `/pause` docstring says `clear_cache` is *"Ignored when mode='keep'"* (`rlhf/api_router.py:46-47`). The code **honours** it for `keep`: `_finish_pause(clear_cache)` runs on the idle callback for every mode (`core.py:2003-2009`, `:2017-2021`). Under `keep` + `clear_cache=True`, frozen requests are additionally preempted because `_reset_caches()` defaults `reset_running_requests=True` (`core.py:863`).
2. `docs/training/async_rl.md:61` documents that with `clear_cache=False` *"some tokens in context may still reflect the old weights (stale KV cache)"* — an accepted hazard that RolloutCore must not rely on.

---

## 5. KV release and invalidation path

### 5.1 `release_kv_cache_memory()` — `core.py:984-1002`

```python
if not (self.is_scheduler_paused() and not self.scheduler.has_requests() and not self.batch_queue):
    raise RuntimeError("release_kv_cache_memory() requires a completed pause first")
if self.model_executor.is_sleeping:
    raise RuntimeError("release_kv_cache_memory() requires all executor memory to be resident")
self._reset_caches()
self.model_executor.discard(("kv_cache",))
```

- Preconditions are **enforced**, not advisory.
- Resets prefix/MM/encoder caches and discards only `"kv_cache"`-tagged allocations — **weights stay resident**. This is the "release KV between rollout steps without full sleep" primitive RFC #48311 refers to (`#46438` → `#44890`).
- **Does not exist on the worker or model runner.** Tree-wide it appears only at engine/dispatch level: `core.py:984`, `async_llm.py:1104-1109`, `llm_engine.py:381-383`, `core_client.py:201/296/411/1047/1294`, `entrypoints/llm.py:833`, `engine/protocol.py:196-199`, `dev/sleep/api_router.py:31`.
- `AsyncLLM.release_kv_cache_memory` clears the MM cache first (`async_llm.py:1104-1106`) and records sleep state 0 (`:1108-1109`).
- `Worker.sleep` also has no `mode`; `free_kv_cache` / `_free_kv_cache` / `_reshape_kv_cache` / `CumemAllocator` (lowercase) do **not** exist on this commit.

### 5.2 Prefix cache reset

| Layer | Anchor |
|---|---|
| `EngineCore.reset_prefix_cache(...) -> bool` | `core.py:834-839` |
| `Scheduler.reset_prefix_cache(reset_running_requests=False, reset_connector=False) -> bool` | `scheduler.py:2706-2759`; aux-output guard `:2716-2725` (**raises `RuntimeError(... "pause(mode='keep')")`** unless `PAUSED_ALL`); preempts running in reverse order `:2734-2736`; `reset_successful = self.kv_cache_manager.reset_prefix_cache()` `:2744`; `reset_connector_cache()` `:2753-2754` |
| `KVCacheManager.reset_prefix_cache() -> bool` | `vllm/v1/core/kv_cache_manager.py:664-679` → `coordinator.reset_prefix_cache()` (`kv_cache_coordinator.py:373-378`, `all(...)` across single-type managers) |
| `BlockPool.reset_prefix_cache() -> bool` | `vllm/v1/core/block_pool.py:821-860` — **returns `False` while any non-null block is still in use** (`:831-838`, warns *"Failed to reset prefix cache because some blocks (%d) are not freed yet"*); on success replaces `cached_block_hash_to_block` and clears `cached_block_hashes_by_block` (`:840-842`), `block.reset_hash()` for every block (`:848-849`) |

> **Naming trap:** the *scheduler* parameter is `reset_connector`. `reset_external` exists **only** as the HTTP query param (`dev/cache/api_router.py:24`) and is passed **positionally** into the `reset_connector` slot (`:41-43`).
>
> `reset_connector_cache()` is a **no-op success** when no connector is configured (`scheduler.py:2762-2772`).
>
> `BlockPool.reset_prefix_cache` never zeroes KV contents and never touches `ref_cnt > 0` blocks — it only drops hash metadata, so subsequent lookups cannot hit.

`BlockPool.reset_prefix_cache` docstring `:822-824`: *"This function may be used in RLHF flows to invalid prefix caching after the weights are updated."*

### 5.3 Other caches

- `EngineCore.reset_encoder_cache()` — `core.py:841-859`: warns if requests are unfinished (`:850-854`); resets the scheduler's **logical** encoder cache (`:857`) and the GPU model runner's **physical** cache (`:858`). Docstring `:842-846` explicitly: *"This should be called when model weights are updated to ensure stale vision embeddings computed with old weights are not reused."*
- `EngineCore.reset_mm_cache()` — `core.py:861-875` region; `AsyncLLM.reset_mm_cache` `async_llm.py:1078-1084` joins the MM warmup first (`:1080-1082`).
- HTTP: `dev/cache/api_router.py` — `POST /reset_prefix_cache?reset_running_requests=&reset_external=` `:20-44` (returns `{"success": bool}`), `POST /reset_mm_cache` `:47-54`, `POST /reset_encoder_cache` `:57-64`.
- `AsyncLLM.reset_prefix_cache` — `async_llm.py:1086-1091`.

### 5.4 Cache identity — the core hazard (RFC #48312 category 7)

Block hash: `hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)` — `vllm/v1/core/kv_cache_utils.py:649-679`; digest is `hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))` (`:677-679`).

`extra_keys` is built by `generate_block_hash_extra_keys()` — `:610-646`:

```python
extra_keys = lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys   # :639-641
```

- `lora_extra_keys` = `[request.lora_request.lora_name]` — **name only**, no id, no version (`_gen_lora_extra_hash_keys():567-580`)
- `cache_salt_keys` = `[request.cache_salt]` **at block 0 only** (`:632-634`)
- `mm_extra_keys` / `prompt_embeds_keys` = content hashes (`:627-630`, `:635-637`)

**There is no model, adapter, target-model, draft-model or weight generation component in the cache key.** Consequences:

- A weight update at constant token ids + cache salt produces **cache hits on stale KV**. Invalidation must be explicit.
- A same-*name* adapter reload reuses stale blocks — the open bug tracked as `#42125` / `#44950`, and the most damaging failure class reported by the production RL operator in the #48312 thread.
- `cache_salt` is the **only caller-controllable identity input** — and it *is* reachable from the OpenAI API: `chat_completion/protocol.py:456` (+ validator `:514-517`), `completion/protocol.py:200` (+ `:411-414`), `responses/protocol.py:255` (+ `:474-477`); it lands on `EngineCoreRequest.cache_salt:122` via `input_processor.py:483` and on `Request.cache_salt:185`.
  **This is a zero-diff defence:** salting every rollout with the weight version makes cross-version cache reuse impossible *by construction*, independent of whether `reset_prefix_cache` succeeded.
- **LoRA caches are keyed by adapter *id* only, with no version.** `LRUCacheLoRAModelManager.add_adapter` merely LRU-touches an already-registered adapter (`vllm/lora/model_manager.py:1350-1360`); `WorkerLoRAManager.add_adapter` reloads only when `lora_request.lora_int_id not in self.list_adapters() or lora_request.load_inplace` (`vllm/lora/worker_manager.py:298-301`). `LoRARequest.load_inplace` (`vllm/lora/request.py:29`) is the *only* reload escape hatch, and `LoRARequest.__eq__`/`__hash__` are by `lora_name` alone (`:67-83`). Nothing in `vllm/lora/` invalidates prefix / MM / encoder caches.
- **MM processor and encoder caches carry no weight generation either.** The MM processor cache key is `model_id` + item content + processor kwargs (`vllm/multimodal/processing/inputs.py:95-102`), where `model_id` is `model_config.model` — a name/path (`multimodal/processing/context.py:367-369`). The encoder cache is `dict[mm_hash, Tensor]` (`vllm/v1/worker/gpu/mm/encoder_cache.py:8-13`). `MultiModalFeatureSpec.identifier` (`multimodal/inputs.py:363-364`) adds a LoRA prefix **name only** via `input_processor.py:248-263`; the P1 receiver cache key deliberately *excludes* LoRA, using `feature.mm_hash` (`multimodal/cache/base.py:447-456`).

**Per-operation invalidation summary (definitive):**

| Operation | Invalidates prefix / MM / encoder? | Anchor |
|---|---|---|
| `pause_scheduler(..., clear_cache=True)` | **Yes** | `core.py:877-882`, `:2017-2021` |
| `pause_scheduler(..., clear_cache=False)` | No | `core.py:881-882` |
| `sleep(level>=1)` | **Yes** (via pause; `clear_prefix_cache = level >= 1`) | `core.py:936-937` |
| `sleep(level=0)` | No | `core.py:936-939` |
| `wake_up(tags)` | **No** — re-maps memory + resumes scheduler only | `core.py:960-982` |
| `release_kv_cache_memory()` | **Yes** — explicit `_reset_caches()` + `discard(("kv_cache",))` | `core.py:1001-1002` |
| `resume_scheduler()` | No | `core.py:914-916` |
| `finish_weight_update()` | **No** | `async_llm.py:1284-1288`; `gpu_worker.py:1488-1505` |

### 5.5 `#48762` — the encoder-cache fix that did **not** land

`vllm-project/vllm#48762` — *"[Bugfix][V1] Invalidate encoder cache on finish_weight_update"* — GitHub state `closed`, `merged_at: null`, closed 2026-07-17. Consistent with the code: `finish_weight_update` still invalidates no KV, encoder or multimodal cache (its only cleanup is `reset_lora_state()`). RFC #48312 lists this as an exit criterion (*"#48762 or an equivalent non-reverted fix lands"*).

---

## 6. How request metadata reaches final responses

### 6.1 Output dataclasses

**`EngineCoreOutput`** — `vllm/v1/engine/__init__.py:199` (`msgspec.Struct`, `array_like`):
`request_id:205`, `new_token_ids:206`, `new_logprobs:208` (`LogprobsLists|None`), `new_prompt_logprobs_tensors:209` (`LogprobsTensors|None`), `pooling_output:211`, `finish_reason:213`, `stop_reason:214`, `events:215`, `kv_transfer_params:216`, `ec_transfer_params:217`, `trace_headers:219`, `prefill_stats:221`, `routed_experts:223` (`np.ndarray|None`), `num_nans_in_logits:226`, `mm_cache_miss_hashes:231`, `new_sampling_mask:233`, `spec_decode_metrics:237`; `finished` property `:239-241`.

**`EngineCoreOutputs`** — `:256`: `engine_index:265`, `outputs:268`, `scheduler_stats:269`, `timestamp:270`, `utility_output:272`, `finished_requests:273`, `wave_complete:277`, `start_wave:280`.

**`CompletionOutput`** — `vllm/outputs.py:33` (`@dataclass`): `index:59`, `text:60`, `token_ids:61`, `cumulative_logprob:62`, `logprobs:63`, `routed_experts:64` (`np.ndarray|None  # [seq_len,layer_num,topk]`), `finish_reason:65`, `stop_reason:66`, `lora_request:67`, `sampling_mask:68`, `spec_decode_metrics:69`.

**`RequestOutput`** — `vllm/outputs.py:108`, explicit `__init__:136-174`: `request_id:161`, `prompt:162`, `prompt_token_ids:163`, `prompt_logprobs:164`, `outputs:165`, `finished:166`, `metrics:167`, `lora_request:168`, `encoder_prompt:169`, `encoder_prompt_token_ids:170`, `num_cached_tokens:171`, `num_cache_creation_tokens:172`, `kv_transfer_params:173`, `ec_transfer_params:174` (the latter two keyword-only, `:151-152`).

### 6.2 The definitive metadata answer

- **No generic metadata bag exists end-to-end.**
- `RequestOutput.__init__` has `**kwargs: Any` at `outputs.py:155` — forward-compat tolerance only:
  ```python
  if kwargs:
      logger.warning_once("RequestOutput: Ignoring extra arguments: %s", str(kwargs))   # :157-160
  ```
- The only free-form dicts are `kv_transfer_params` / `ec_transfer_params` — **connector-specific**, populated from exactly two hard-coded `SamplingParams.extra_args` keys:
  - `vllm/v1/request.py:119-128` extracts `kv_transfer_params`, `ec_transfer_params`, `kv_cache_report_mode`
  - `vllm/v1/engine/output_processor.py:238-245` reads only `extra_args["kv_transfer_params"]["do_remote_prefill"]` / `remote_prefill_cached_tokens`
- `trace_headers: Mapping[str, str]` is a verbatim caller pass-through (`EngineCoreRequest:142` → `Request:196` → `EngineCoreOutput:219` → `output_processor.py:805`) but is consumed **only** by OpenTelemetry and never reaches a response.

### 6.3 The caller-extensible request bag: `vllm_xargs`

| Field | Anchor |
|---|---|
| `vllm_xargs: dict[str, str \| int \| float \| list[...]] \| None` | chat `chat_completion/protocol.py:482`; completion `completion/protocol.py:226`; responses `responses/protocol.py:290`; anthropic `anthropic/protocol.py:164`; transcription `vllm/entrypoints/speech_to_text/transcription/protocol.py:116`; translation `vllm/entrypoints/speech_to_text/translation/protocol.py:163` |
| → `SamplingParams.extra_args` | chat `protocol.py:703` + `:743`; completion `protocol.py:365,368-372`; responses `protocol.py:433,435-438` |
| Declaration | `vllm/sampling_params.py:343` — *"Arbitrary additional args, that can be used by custom sampling implementations, plugins, etc. Not used by any in-tree sampling implementations."* |
| Validation | `SamplingParams._verify_extra_args()` `sampling_params.py:670-681`, invoked `:557-558` (recursive JSON-primitive + int64-range check) |

**Arbitrary `vllm_xargs` keys reach `SamplingParams.extra_args` and stop there** — they are never surfaced in `RequestOutput` / `CompletionOutput`. There is **no server-side `extra_body`** (`extra_body` appears only in client-side benchmark code under `vllm/benchmarks/`).

### 6.4 Where `RequestOutput` is assembled

`vllm/v1/engine/output_processor.py`:

| Step | Anchor |
|---|---|
| `RequestOutputCollector` | `:51`; `put():67`, `get():83`, `close():103` |
| `OutputProcessorOutput` | `:114-117` (`request_outputs`, `reqs_to_abort`) |
| `RequestState` | `:134`; fields `:159-195` (incl. `logprobs_processor:172`, `detokenizer:173`, `routed_experts_chunks:190`, `sampling_mask_chunks:191`, `sent_tokens_offset:195`) |
| `RequestState.from_new_request` | `:222-297`; `kv_transfer_params` read `:238-245`; `LogprobsProcessor.from_new_request:252`; `IncrementalDetokenizer.from_new_request:256`; `external_req_id` assert `:274` |
| `make_request_output` | `:299-363`; FINAL_ONLY gating `:309-313`; DELTA slicing `:330-336`; parent merge `:352` |
| **`_new_request_output` — `RequestOutput(...)` constructed** | `:365-409` (call `:396-409`) |
| **`_new_completion_output` — `CompletionOutput(...)`** | `:411-459` (incl. `routed_experts` `:443-445`, `:451`) |
| `OutputProcessor` | `:464`; `request_states:479`, `parent_requests:480`, `external_req_ids:481` |
| `abort_requests` | `:512-574` (final `FinishReason.ABORT` output `:551-565`) |
| `add_request` | `:576-606` |
| **`process_outputs` — the single batch loop** | `:641-770`; per-output body `:669-770`; `:685-686` params; `:687-690` routed experts; `:715-720` stop strings; `:724` logprobs; `:727-734` request output; `:738-740` queue put |
| `_finish_request` | `:772-785` |
| `do_tracing` (only `trace_headers` consumer) | `:794`, `:805` |

`RequestOutput.add()` — `outputs.py:176-208` — merges streaming deltas; `:197-200` explicitly preserves `routed_experts` with the comment *"R3 is returned on the terminal output and must survive aggregation with earlier chunks that have no R3."*

---

## 7. Where token IDs and logprobs are stored

### 7.1 `Request` (`vllm/v1/request.py:60`, `__init__:61-83`)

| Field | Line |
|---|---|
| `request_id` / `client_index` / `priority` / `sampling_params` | `:84` / `:85` / `:86` / `:87` |
| `lora_request` / `structured_output_request` | `:89` / `:90` |
| `arrival_time` / `status = WAITING` / `events` / `stop_reason` | `:98` / `:100` / `:101` / `:102` |
| **`kv_transfer_params` / `ec_transfer_params`** | `:105` / `:107` |
| `max_tokens` | `:111`/`:115` |
| `kv_cache_report_mode` | `:126`/`:130` |
| `prompt_token_ids` / `prompt_embeds` | `:134` / `:135` |
| `prompt_is_token_ids` | `:139` |
| `_prompt_embeds_per_block_hashes` / `num_prompt_tokens` | `:142` / `:143` |
| **`_output_token_ids: list[int] = []`** | `:146` |
| `_all_token_ids` | `:148`/`:150`/`:156` |
| `num_output_placeholders` | `:162` |
| `num_stale_output_tokens` / `drop_stale_output` | `:165` / `:168` |
| `num_in_flight_tokens` / `next_decode_eligible_step` / `last_sched_seq` | `:173` / `:177` / `:181` |
| `spec_token_ids` / `num_computed_tokens` / `cache_salt` | `:183` / `:184` / `:185` |
| `mm_features` | `:188` |
| **`output_token_ids = ConstantList(self._output_token_ids)`** | `:193` |
| `all_token_ids` | `:194` |
| **`trace_headers`** | `:196` |
| `session_id` / `kv_hints` / `is_prefill_chunk` | `:197` / `:198` / `:201` |
| `shared_prefix_boundary` / `replay_start` | `:206` / `:209` |
| `num_nans_in_logits` / `num_preemptions` | `:213` / `:216` |
| `prefill_stats` / `spec_decode_metrics` | `:218` / `:223` |
| `block_hashes` / `_block_hasher` | `:225` / `:229` |
| `skip_reading_prefix_cache` / `resumable` | `:232` / `:235` |
| `streaming_queue` / `abort_immediately` | `:237` / `:241` |

Also: `from_engine_core_request()` `:243-270` (`trace_headers=request.trace_headers:262`), `append_output_token_ids()` `:272-283` (appends `:277-281`, then `update_block_hashes():283`), `num_output_tokens` property `:302`, `is_finished():327`, `record_event():337`, `take_prefill_stats():350`, `RequestStatus:370`.

### 7.2 Per-step append path

`SamplerOutput` (`vllm/v1/outputs.py:198`; `sampled_token_ids:204`, `logprobs_tensors:205`) ← `Sampler.forward` (`vllm/v1/sample/sampler.py:142-148`) → worker bookkeeping `gpu_model_runner.py:3785 req_state.output_token_ids.extend(sampled_ids)` → `ModelRunnerOutput.sampled_token_ids` (`vllm/v1/outputs.py:271`, assembled `gpu_model_runner.py:4745-4759`) → `Scheduler._update_request_with_output` (`scheduler.py:2362-2377`) → `request.append_output_token_ids(output_token_id)` (`scheduler.py:2371`) → `EngineCoreOutput.new_token_ids` (`scheduler.py:2153`).

Async D2H variant: `AsyncGPUModelRunnerOutput` (`gpu_model_runner.py:281`, `:312-317`, `:342`, `:357-358`).

`ModelRunnerOutput` also: `req_ids:263`, `req_id_to_index:265`, `logprobs:276`, `prompt_logprobs_dict:282`, `kv_connector_output:289`, `ec_connector_output:291`, `num_nans_in_logits:294`, `aux_output_connector_output:299`, `sampling_masks:302`; `EMPTY_MODEL_RUNNER_OUTPUT:396`.

### 7.3 Logprobs

| Layer | Anchor |
|---|---|
| Request knobs | `SamplingParams.logprobs:281`, `prompt_logprobs:289`, `logprob_token_ids:292`, `flat_logprobs:298`; `num_logprobs` property `:809-816`; `_validate_logprobs:843` |
| API → params | `chat_completion/protocol.py:715-722` |
| V1 sampler | `vllm/v1/sample/sampler.py:21`, `forward:72`, `compute_logprobs:304-306` (`log_softmax(dim=-1, dtype=torch.float32)`), `gather_logprobs:308-357` (`topk:334`, `token_ranks:348`, `LogprobsTensors:357`) |
| V2 sampler | `vllm/v1/worker/gpu/sample/sampler.py:118 get_logprobs_dims`, `:171-188` |
| Sampling metadata | `vllm/v1/sample/metadata.py:15`; `max_num_logprobs:26`, `logprob_token_ids:49` |
| Worker transport | `gpu_model_runner.py:3724-3725`, `:4750-4751`; prompt side `_get_prompt_logprobs_dict():5582`, `:5637` |
| Scheduler slice | `scheduler.py:2128-2134` (`logprobs.slice_request(req_index, len(new_token_ids))`), `:2155`, `:2157` |
| `LogprobsProcessor` | `vllm/v1/engine/logprobs.py:29`; fields `:33-40`; `from_new_request:42-67`; `_update_sample_logprobs:69` (`cumulative_logprob += sampled_token_logprob:107-108`); `_update_prompt_logprobs:120`; `pop_prompt_logprobs:187`; `update_from_output:350-354` |
| Invocation | `output_processor.py:724` |
| Output assembly | `output_processor.py:428-434`, `:453-454` |
| Types | `vllm/logprobs.py:13 Logprob`, `:32 FlatLogprobs`, `:160 PromptLogprobs`, `:162 SampleLogprobs`, `:165 create_prompt_logprobs`, `:173 create_sample_logprobs` |
| OpenAI conversion | `chat_completion/serving.py:629-642`, `:1175`, `:1219` |

Containers: `LogprobsLists` (`vllm/v1/outputs.py:34` — `logprob_token_ids:36`, `logprobs:38`, `sampled_token_ranks:40`, `cu_num_generated_tokens:45`, `slice_request():47`); `LogprobsTensors` (`:83` — `:85/:87/:89/:91/:95`; `tolists():97`, `to_cpu_nonblocking():110`, `filter():124`, `cat():139`, `empty_cpu():168`).

Detokenizer owns **text only**: `IncrementalDetokenizer` (`vllm/v1/engine/detokenizer.py:31`; `token_ids:33`, `output_token_ids:36`, `update():42-44`/`:96`, `get_next_output_text():148`; `FastIncrementalDetokenizer:166`, `SlowIncrementalDetokenizer:249`). `LogprobsProcessor` decodes logprob token ids independently (`logprobs.py:96-98`, `:145-147`, `_correct_decoded_token:249`).

### 7.4 Two RL-relevant features that ARE landed

**(a) Trace replay — `trace_decode_token_ids`.** PR #46701, merged 2026-08-20.

- `SamplingParams.trace_decode_token_ids: list[int] | None` — `sampling_params.py:374`; docstring `:375-377`: *"forces the engine to emit this predetermined sequence of token IDs during decoding instead of sampling randomly. Real logprobs are still computed."*
- Validation `_validate_trace_decode_token_ids():956-1001` — `n=1` required (`:962`), rejects `prompt_logprobs` (`:970`), speculative decoding (`:974`), structured outputs (`:978`), repetition detection (`:982`), thinking budget (`:986`), bad words (`:989`).
- Gate: `ModelConfig.enable_trace_replay: bool = False` (`vllm/config/model.py:276`; doc `:277-281`), CLI `--enable-trace-replay` (`arg_utils.py:951-953`, field `:586`), **requires Model Runner V2** (`config/vllm.py:1262-1267`, raises `:1266`: *"trace replay requires Model Runner V2"*).
- Admission `input_processor.py:205-227` (truncate to `max_model_len - prompt_len`, `max_tokens = min(len(trace), max_tokens)` `:219-221`, `min_tokens=0` `:222`, `ignore_eos=True` `:223`, clear stops `:225-227`), invoked `:430-431`; hard error without the flag `:168-176`.
- Runtime: `TraceReplayState` (`vllm/v1/worker/gpu/sample/trace_replay.py:11`), `StagedWriteTensor:26-31`, `add_request:34`, `apply_staged_writes:42`, `apply_trace:46`, Triton kernel `_trace_replay_kernel:61-92`.
- Wired `vllm/v1/worker/gpu/sample/sampler.py:53`, `:82-83`, `:106-107`, `:115-116`, and critically `:166-169`:
  ```python
  if self.trace_replay_state is not None:
      # Overwrite sampled tokens with the replay trace up-front so that
      # computed logprobs reflect the real distribution of the forced token.
      self.trace_replay_state.apply_trace(sampled, idx_mapping)
  ```
  applied **before** logprob computation (`:171-188`) — this is what makes the logprobs "real".

**(b) Routed experts / R3.** Landed end-to-end and reaches the client.

- Config: `AuxOutputConfig.enable_return_routed_experts: bool = False` (`vllm/config/aux_output.py:14`), `max_bytes:17`, `enabled` property `:21-23`, `compute_hash:25-30`; `VllmConfig.aux_output_config` (`config/vllm.py:380`); CLI `--enable-return-routed-experts` (`arg_utils.py:453`, `:831-832`); `LLM(enable_return_routed_experts=...)` (`entrypoints/llm.py:206`, `:322`).
- Per-request: `SamplingParams.routed_experts_prompt_start:367` (validated `input_processor.py:404-410`); API fields chat `protocol.py:426` / completion `protocol.py:179`.
- Capture: `RoutedExpertsCapturer` (`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:43`; `capture():86`, `snapshot_routing_data():184`, `bind_routed_experts_capturer():189`); bound `vllm/distributed/aux_output_connector/worker.py:94-97`.
- Transport is **not** the token IPC path — a block-hash-keyed shared store: `vllm/distributed/aux_output_connector/` (`connector.py:34 AuxRequestOutput`, `:39 AuxOutputSchedulerConnector`, `:48 build_connector_meta`, `:98 take_output`, `:140 request_finished`; `routed_experts.py:162/:169/:182`; `store.py`).
- Scheduler: connector constructed `scheduler.py:395-397`; metadata `:1493-1496`; `take_output` `:2088-2095`; `EngineCoreOutput(routed_experts=...)` `:2170`.
- Frontend: `output_processor.py:190` chunks, `:687-690` append, `:443-445` concat on finish, `:451` into `CompletionOutput`.
- API (base64): `chat_completion/serving.py:1082-1084`, `:1103`; field `chat_completion/protocol.py:124`; completion `completion/serving.py:571-573`, `:590`, field `completion/protocol.py:655`.

### 7.5 DSA indexer top-k — **NOT PRESENT as a return-to-user feature**

- `indexer_topk` is only a **kernel backend selector**: `SparseIndexerTopkBackend = Literal[...]` (`vllm/config/kernel.py:159`), `sparse_indexer_topk_backend:278`, validator `:343-345`; CLI `--sparse-indexer-topk-backend` (`arg_utils.py:1722-1725`); `get_indexer_topk()` (`vllm/model_executor/layers/indexer_topk.py:139`).
- `enable_return_indexer_topk` / `return_indexer_topk` / the symbol `dsa_index`: **NOT PRESENT on main@00b7847c**.
- Indexer top-k never reaches `EngineCoreOutput`, `vllm/v1/outputs.py`, `vllm/outputs.py`, `output_processor.py` or `sampling_params.py`.

---

## 8. Existing RL examples, tests and CI

### 8.1 Examples — `examples/rl/` (11 files, all verified)

| File | Scenario | Backend | Lifecycle APIs used |
|---|---|---|---|
| `rlhf_http_nccl.py` (214 L) | 3 GPUs: server TP2 fp8 + bf16 trainer; gibberish → sync → sensible | `nccl` (`:78-80`) | `/health:98`, `POST /pause:125`, `POST /resume:130`, `GET /get_world_size:135`; `NCCLTrainerInitInfo(...):180-186`; `engine.send_weights():196`; `VLLM_SERVER_DEV_MODE=1:83` |
| `rlhf_http_ipc.py` (199 L) | 1 colocated GPU | `ipc` (`:84-86`) | `/pause:132`, `/resume:137`; `IPCTrainerInitInfo:172`; also needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (`:57`, `:90`) |
| `rlhf_nccl_fsdp_ep.py` (368 L) | 8 GPUs: 4×FSDP2 Ray trainers + TP1/DP4 EP | `nccl` (`:188-189`) | `/pause:337`, `/resume:346`; world size `TP*DP+1:314` |
| `rlhf_ipc_fsdp_ep.py` (315 L) | 4 GPUs colocated FSDP2 + DP4/EP, packed 1 GiB chunks | `ipc` (`:163-164`) | **`/sleep`+`/wake_up`**: `sleep(level=1):290` → `wake_up(tags=["weights"]):293` → transfer `:296` → `wake_up(tags=["kv_cache","scheduling"]):300`; `--enable-sleep-mode:155` |
| **`rlhf_async_new_apis.py` (362 L)** | **Closest existing pattern to the MVP**: Ray, `AsyncLLMEngine` subclass, mid-flight swap + validation | `nccl` (`:203`) + `RayVLLMWeightSyncClient:148` | `pause_generation(mode="keep"):115`, `resume_generation:266`; token-threshold trigger `:111-117`; `VLLM_BATCH_INVARIANT=1:177`; `assert pass_rate >= MIN_PASS_RATE:357` |
| `rlhf_sparse_nccl.py` (253 L) | checkpoint-coordinate sparse patches, Qwen3 MoE TP2/EP2 | `sparse_nccl` (`:175`) | `llm.sleep.remote(level=0):221`, `llm.wake_up.remote(tags=["scheduling"]):225` |
| `rlhf_sharded_rdt_small_ep.py` (270 L) | CI-sized sharded_rdt: 2 FSDP2 → 2 DP+EP | `sharded_rdt` | pause/resume `:236,240,248,252`; asserts generation **changed** on sync 0 and **stable** on replay sync 1 (`:256-263`) |
| `rdt_vllm_serve.py` (152 L) | driver helpers | `sharded_rdt` (`:58-59`) | `launch_vllm_serve:25`, `/pause:139`, `/resume:143`; `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1:79` |
| `rdt_weight_source.py` (92 L) | fused → per-expert checkpoint names | — | `CheckpointNameSource(WeightSource):24` |
| `routed_experts_e2e.py` (383 L) | routed-experts capture (not weight transfer) | — | `enable_return_routed_experts=True:139-146` |
| `skip_loading_weights_in_engine_init.py` (53 L) | dummy → auto reload offline | — | `collective_rpc("update_config",...)`:40-42; `collective_rpc("reload_weights")`:44 |

Related examples elsewhere: `examples/features/pause_resume/pause_resume_offline.py` (`:66`, `:74`), `examples/features/pause_resume/data_parallel_pause_resume.py` (`/pause:42`, `/resume:50`), `examples/features/reset_kv/reset_kv_offline.py:62` (`reset_prefix_cache(reset_running_requests=True)`), `examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py:194`.

**NOT PRESENT:** `examples/rl/rlhf.py`, `examples/rl/new_weight_syncing/`, `examples/offline_inference/` (directory absent), `examples/features/sleep_mode/`.

> **None of the examples invalidate the prefix/encoder cache between update and resume.** That missing step is exactly what RolloutCore adds.

### 8.2 Tests

**Weight transfer**
- `tests/distributed/test_weight_transfer.py` (1978 L): `TestEngineRegistry:267`, `test_nccl_receive_weights_without_init_raises:320`, `test_sparse_nccl_receive_weights_without_init_raises:341`, **`test_nccl_weight_transfer_between_processes:503`**, `test_sparse_nccl_checkpoint_chunks_to_ep_local_experts_cpu:535`, `TestIPCWeightTransferUpdateInfoValidation:669`, `TestIPCEngineParsing:817`, **`test_ipc_weight_transfer_between_processes:1077`**, `test_ipc_receive_weights_missing_gpu_uuid_raises:1114`, `TestTrainerClients:1187` (weight version `:1221`, `:1262`), `TestModuleSource:1265`, `TestWeightSourceGroupContract:1301`, `TestDeferredProcessingContract:1420`, `TestTrainerFactory:1457`, `TestTrainerEngineBase:1506`, per-backend trainer tests `:1531-1978`.
- `tests/distributed/test_weight_transfer_nccl_uid.py`: `TestNCCLRendezvousValidation:41`, `test_nccl_weight_transfer_between_processes_uid:451`, `test_nccl_weight_transfer_torch_free_trainer:461`.
- `tests/distributed/test_sharded_rdt_plan.py` (`TestBakeRecording:279`, `TestRdtRouter:467`, `TestBuildCallPlan:866`, `TestLayerwiseGroups:1230`, `TestBakeOnARealModel:1728`), `test_sharded_rdt_producer.py` (`:220`, `:266`, `:401`, `:568`, `:616`, `:1131`), `test_sharded_rdt_trainer.py` (`:230`, `:250`, `:275`, `:304`, `:344`, `:459`).
- `tests/v1/worker/test_gpu_worker_weight_transfer.py`: `:75`, `:85`, `:101`, `:125`, `:139`, `:171`, `:183`, `:190`, `:196`, `:202`, `:215`.
- `tests/entrypoints/weight_transfer/test_weight_transfer_llm.py`: `:105`, `:122`, `:175`, **`test_full_weight_transfer_flow:241`** (asserts version `"default"` `:237` → `"step-42"` `:292`; `update_weight_version("manual-version")` `:294`), `:320`.

**RLHF dev endpoints** — `tests/entrypoints/serve/dev/rlhf/` contains only `conftest.py` + `state_transitions/test_pause_resume.py`.
- `conftest.py`: DEV_MODE gate `:96`; base args include `--enable-sleep-mode` (`:42`/`:58`); helpers `poll_until:146`, `gen:176`, `gen_with_logprobs:194`, `stream_completion:243`, `start_stream:276`, `pause:304`, `resume:312`, `completion_with_cache_details:316`, `golden_output:332`, `cached_tokens:344`, `sleep:353`, `wake:359`, `is_sleeping:364`, `is_paused:368`, `health:372`, `start_weight_update:384`, `finish_weight_update:392`, `get_world_size:396`, `gpu_free_bytes:409`, `sleep_metrics:422`.
- **Only `/pause`, `/resume`, `/is_paused`, `/v1/completions`, `/health` are actually called** by test code. `sleep`/`wake`/`is_sleeping`/`start_weight_update`/`finish_weight_update`/`get_world_size`/`sleep_metrics`/`gpu_free_bytes`/`poll_until` have **no importers** -- checked by import, because a bare-name grep is false here (`sleep` and `get_world_size` name unrelated functions elsewhere in the tree).
- `state_transitions/test_pause_resume.py` (168 L): parametrized MRV1/MRV2 (`:27-35`); `TestPauseResume:57`; `test_state_and_idempotency_across_cycles:58` (abort/wait/keep `:66`); `test_invalid_mode_preserves_state:75` (400 + `error.param == "query.mode"` `:86-87`); **`test_mode_request_lifecycle:100`** — the reference oracle for drain semantics (abort/wait ⇒ in-flight finished `:120-121`; keep ⇒ no new chunks until resume `:122-127`; new request must not complete while paused `:129-132`); `test_clear_cache_preserves_output_and_controls_prefix_cache:145` (cached_tokens 0 → >0 → preserved → 0, `:155-168`).

**Sleep / wake**
- `tests/entrypoints/serve/dev/test_sleep.py`: `test_release_kv_cache_memory_route:20` (mocked engine), `test_sleep_mode:35` (DEV_MODE `:50`; `/sleep:52`, `/is_sleeping:54/:68/:90/:99`, `/metrics:59`, `/wake_up:66`, partial wakes `:85`, `:95`).
- `tests/basic_correctness/test_mem.py`: **`test_release_kv_cache_memory_preserves_generation:114`** (params `["kv-only","sleep-1","sleep-2"]:112`), `test_sleep_with_only_weights_asleep:173`, `test_discard_tags:207`, `test_level2_discards_ordinary_tensor_with_weights_tag:242`, `test_deep_sleep:367`, `test_deep_sleep_lora:398`, `:449`, `test_deep_sleep_async:491`, `:530`, `:571`.
- `tests/v1/worker/test_sleep_mode_backend.py`: `:19`, `:25`, `:36`, `:42`, `:91`, `:96`, `:105`, `:121`.
- `tests/model_executor/test_sleep_mode_tensor_ownership.py::test_static_model_tensors_survive_level2_restore:261` (marked `slow_test:258` — **no CI lane runs it**).
- `tests/models/language/generation/test_gdn_sleep_wake.py:35`.

**Pause / resume / drain**
- `tests/v1/engine/test_async_llm.py`: `test_pause_resume_basic:749`, `test_pause_abort:815`, `test_pause_then_abort_queued_request:886`, `test_pause_wait:933`, `test_pause_keep_single_request:975`, `test_pause_keep_multi_request:1037`.
- `tests/v1/distributed/test_async_llm_dp.py`: `:278`, `test_dp_pause_late_request_does_not_block_drain:318`, `:441`, `:494`, `:547`, `test_dp_pause_barrier_request_deadlock:608`, `test_dp_pause_wait_mode_drains_in_flight:717`, `test_dp_pause_while_asleep:748`, **`test_dp_pause_completion_implies_device_idle:781`**.
- `tests/v1/distributed/test_pp_dp_v2.py:153`.

**Prefix cache reset**
- `tests/v1/core/test_scheduler.py`: **`test_scheduler_reset_prefix_cache:1544`** (fails with running requests; after `PAUSED_ALL` raises `"model output is in flight"` `:1571`), `test_aux_output_reset_follows_kv_reset_result:1587`, **`test_kv_cache_release_after_keep_pause_preserves_requests:1596`** (uses `pause_scheduler(mode="keep", clear_cache=False):1619`).
- `tests/v1/core/test_reset_prefix_cache_e2e.py:14`.

**End-to-end RL loop test (rollout → update → rollout): NOT PRESENT as a pytest test.** The nearest artifacts are CI-executed examples: `rlhf_sharded_rdt_small_ep.py:256-263` and `rlhf_async_new_apis.py:278-361`.

### 8.3 CI — no dedicated RL lane

No `RL`/`RLHF`/`Weight Transfer` *group* exists (`.buildkite/test_areas/*.yaml`). `.buildkite/test-pipeline.yaml:1-7` is deprecated. `.github/workflows/` has no RL job.

| Lane | Anchor | Runs |
|---|---|---|
| `:nvidia: (L4) Sharded RDT Weight Transfer` | `test_areas/distributed.yaml:21-35` (deps `:27-31`) | `test_sharded_rdt_plan.py`, `_producer`, `_trainer`; AMD mirror `:36-58` |
| `:nvidia: (L4) Distributed Torchrun + Examples` | `test_areas/distributed.yaml:180` (cmd lines `:211-221`) | runs `examples/rl/rlhf_http_nccl.py` (yaml `:211`), `examples/rl/rlhf_http_ipc.py` (`:212`), `examples/rl/rlhf_sharded_rdt_small_ep.py` (`:221`); AMD mirror `:253-256` |
| `:nvidia: (H100) Distributed Features` (optional, `test_areas/distributed.yaml:367`) | `:363-375` | runs `examples/rl/rlhf_async_new_apis.py` (yaml `:372`); `pytest tests/distributed/test_weight_transfer.py` (yaml `:375`) |
| Entrypoints | `test_areas/entrypoints.yaml:34,38` and `:74,87` | `entrypoints/weight_transfer`; `entrypoints/serve` glob sweeps in `test_pause_resume.py` and `test_sleep.py` |
| Intel | `intel_jobs/entrypoints_intel.yaml:79-81` | deselects the `clear_cache` MRV1 case `:80` |
| basic correctness | `test_areas/basic_correctness.yaml:14,17` | `test_mem.py` |

**Referenced by no CI file:** `test_weight_transfer_nccl_uid.py`, `routed_experts_e2e.py`, `rlhf_nccl_fsdp_ep.py`, `rlhf_ipc_fsdp_ep.py`, `rlhf_sparse_nccl.py`, `skip_loading_weights_in_engine_init.py`, `rdt_vllm_serve.py`, `rdt_weight_source.py`, `test_reset_prefix_cache_e2e.py` (explicitly ignored by Intel).

Consistent with RFC #48311 / #48305 listing the "RL CI matrix" (`#45585`) as an unmet exit criterion.

### 8.4 Docs

| Doc | Content |
|---|---|
| `docs/training/async_rl.md` (65 L) | pause/resume API; mode table `:23-27`; `clear_cache` `:29`, `:61`; HTTP list `:41-46`; DP caveat `:48-49`; the typical flow `:51-61` |
| `docs/training/rlhf.md` (26 L) | index; points at `async_rl.md:21` |
| `docs/training/weight_transfer/README.md` (169 L) | architecture `:7-20`; four-phase protocol `:22-28`; backend table `:32-37`; **endpoint table `:147-157`**; **"require `VLLM_SERVER_DEV_MODE=1`" `:159-160`**; Rust gRPC `rl_capabilities:162` |
| `docs/training/weight_transfer/{base,nccl,ipc,sharded_rdt}.md` | extension contracts, per-backend guides |
| `docs/training/layerwise.md` (146 L) | layerwise reload (QeRL) |
| `docs/training/trl.md` (54 L) | TRL integration; sleep mode `:51` |
| `docs/features/sleep_mode.md` (158 L) | levels `:19-21`; offline API `:36-58`; RLHF updates `:60-75`; `release_kv_cache_memory()` `:92-102`; `VLLM_SERVER_DEV_MODE=1` + `--enable-sleep-mode` `:106`; curl examples `:113-135`; endpoint list `:145-151` |
| `docs/serving/online_serving/README.md` | dev mode `:167`; **Weight Transfer APIs (RL Training) `:178-193`**; Sleep Mode APIs `:200-205`; Collective RPC `:195-197` |
| `docs/usage/security.md` | dev-endpoint list `:226-238`; *"CRITICAL: Never set `VLLM_SERVER_DEV_MODE=1` in production"* `:262` |

**R3 / routed-experts docs: NOT PRESENT** — zero matches for `routed_experts` / `enable_return_routed_experts` / `\bR3\b` under `docs/**/*.md`.

### 8.5 Environment gating

| Var | Anchor | Effect |
|---|---|---|
| `VLLM_SERVER_DEV_MODE` | declared `vllm/envs.py:169`; accessor `:1435` `bool(int(os.getenv("VLLM_SERVER_DEV_MODE","0")))` | **the only gate** for all RL/sleep/cache/rpc dev endpoints — `routers.py:34-38` |
| `--enable-sleep-mode` | `docs/features/sleep_mode.md:106` | additionally required for `/sleep`, `/wake_up`, `/is_sleeping`, `/release_kv_cache_memory` |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | `ipc_engine.py:87`; example usage `rlhf_http_ipc.py:57`, `:90` | required for IPC packed/pickled handles |
| `VLLM_PLUGINS` | `vllm/envs.py:116`, `:1157-1160` | allowlist for plugin groups; **endpoint plugins are only loaded when explicitly named** (`plugins/__init__.py:122-131`) |
| `VLLM_USE_V2_MODEL_RUNNER` | `vllm/envs.py:302` | selects Model Runner V2 (required by trace replay) |
| `VLLM_ALLOW_INSECURE_SERIALIZATION` | — | also needed by `/collective_rpc`-style arbitrary callable invocation |

### 8.6 Domain vocabulary

- **`trajectory`: NOT PRESENT** in `vllm/`.
- **`rollout`: no abstraction** — comment-only hits (`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py:2522`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py:655`, `routed_experts_capturer.py:154`).
- **`policy`: no RL policy model** — all `*Policy` classes are cache/scheduling/EPLB.
- **`trainer`: real but only weight-transfer trainer-side engines** (`base.py:276`, `:551`; `factory.py:124`; per-backend trainer classes). No optimizer or training loop.

**vLLM has no rollout/trajectory/policy/trainer domain model on main@00b7847c.** RL exists only as (a) the weight-transfer subsystem, (b) pause/resume + sleep lifecycle APIs, (c) routed-experts capture, (d) docs describing external trainers.

---

## 9. The smallest places RolloutCore would need to integrate

### 9.1 Preferred: **zero-diff** — plugin package + existing dev endpoints

**(a) HTTP surface — `vllm.endpoint_plugins`** (use only if new routes are genuinely needed)

- Contract: `EndpointPlugin` Protocol — `vllm/plugins/endpoint_plugins/interface.py:44`; members `name:52`, `required_tasks:55`, `attach_router(app):63`, `init_state(engine_client, state, args):72`.
- Discovery/gating: `load_endpoint_plugins()` — `vllm/plugins/__init__.py:93-159`; **must be named in `VLLM_PLUGINS`** (`:122-131`); `required_tasks` must intersect the server's supported tasks (`:143-154`). `SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]` (`vllm/tasks.py:43`).
- Wiring: `attach_endpoint_plugins(app, supported_tasks)` (`launchers/app.py:51`, attached **last** so routes can shadow core ones) and `init_endpoint_plugins_state(...)` (`launchers/api_server/app_state.py:159`).
- Documented constraint (`interface.py:6-16`, `:75-79`): reach the engine via `engine_client` exactly as in-tree handlers do; do not open a new engine access path.

**(b) Engine / worker-side behaviour — `vllm.general_plugins`**

- `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"` (`plugins/__init__.py:18`), loaded in process0, EngineCore **and workers** via `load_general_plugins()` (`:77-90`), called from `EngineCore.__init__`.
- This is the sanctioned place to install a new worker-side RPC name so `EngineClient.collective_rpc("<name>")` resolves — **no ZMQ framing change needed**, because `_invoke_utility_method` resolves EngineCore methods by name (`core.py:1659-1672`) and `run_method` resolves worker methods by name (`vllm/v1/serial_utils.py:485`).

**Precedent:** `#47173` added `/abort_requests` as a dev route; `#49040` added `/weight_info` + `/update_weight_version`. Both were dev-router additions, not new architecture.

### 9.2 The five engine seams that matter

| Seam | Anchor | Why |
|---|---|---|
| **Drain barrier** | `EngineCoreProc.pause_scheduler` returning a `Future` (`core.py:1984-2026`), awaited via `core_client.pause_scheduler_async` (`:1262-1265`), deferral in `_invoke_utility_method` (`:1659-1672`) | This **is** the "in-flight complete" guarantee |
| **Weight-version write** | `AsyncLLM.finish_weight_update(weight_version)` (`async_llm.py:1284-1288`) | The only version-commit hook |
| **Cache invalidation** | `EngineCore._reset_caches` (`core.py:861-875`); `BlockPool.reset_prefix_cache` returns `False` while blocks are held (`block_pool.py:831-838`) | Must be explicit; the `bool` must be honoured |
| **KV-only eviction** | `EngineCore.release_kv_cache_memory` (`core.py:984-1002`) | Cheaper than `sleep(level=2)` between rollout steps |
| **Worker RPC fan-out** | `EngineCore.collective_rpc` (`core.py:1033-1040`) → `Executor.collective_rpc` (`vllm/v1/executor/abstract.py:185-218`) | The only control-plane → weights route |

### 9.3 What RolloutCore must **not** do

- Do **not** invent a weight-sync transport. Four backends are registered and pluggable (`factory.py:41-82`).
- Do **not** invent a drain primitive. `pause(mode="wait")` already blocks until `not has_work()`.
- Do **not** add a per-request version field to `Request`/`RequestOutput` — #49040 removed that deliberately; the contract is still open (RFC #48306 §2.2).
- Do **not** assume any cache is version-aware. It is not (§5.4).
- Do **not** rely on `finish_weight_update` to invalidate anything (§2.2, §5.5).
- Do **not** rely on `mode="keep"` for a strict one-version-per-response invariant — it lets a single request span two weight versions by design (`docs/training/async_rl.md:61`).

---

## 10. RFC cross-check — claim vs. main

| RFC | Claim | Status on main@00b7847c |
|---|---|---|
| **#48314** (roadmap) | Anchors 31848 / 48311 / 48306 / 48305 | All four open; roadmap updated 2026-09-17 |
| **#48311** | `sleep(level=2, mode="wait")` implements drain | **Partially.** Drain is `pause_scheduler`'s Future (`core.py:1984-2026`). `mode="wait"` **raises** for in-proc engines (`core.py:902-903`) — matches the RFC's "inproc-engine support (currently raises)" |
| #48311 | Requests arriving during drain hang instead of being rejected (#45326) | Consistent: `PAUSED_NEW`/`PAUSED_ALL` queue new adds and never reject (`core.py:889`; `scheduler.py:2666-2668`) |
| #48311 | `release_kv_cache` API (#46438 → #44890) | **Landed** as `release_kv_cache_memory()` (`core.py:984`) with `Executor.discard(("kv_cache",))` |
| #48311 | Pause-state Prometheus metric (#45524) | **Open** (GitHub state `open`, not merged) |
| #48311 | CUDA checkpoint/restore (#34303) | `checkpoint_prepare` / `checkpoint_restore` exist (`async_llm.py:1117-1121`); `SleepModeBackend.supports_durable_storage():110` is the capability probe |
| **#31848** | `init_weight_transfer_engine` / `update_weights` / `finish_weight_update`; NCCL + IPC; `WeightTransferConfig` | **Landed and exceeded.** Routes are `/init_weight_transfer_engine` (`rlhf/api_router.py:156`) and `/start_weight_update` (`:174`); **four** backends registered (`factory.py:222-270`) |
| #31848 | `/finalize_weight_update` redundant | **Confirmed** — no such route; finalization lives inside `finish_weight_update` |
| #31848 | "tracking weight versions" | **Partially** via #49040 — query/update only, no per-request tagging |
| #31848 | Alternative to `DEV_MODE` for endpoints | **Still open** — RL endpoints remain `VLLM_SERVER_DEV_MODE`-gated (`routers.py:34`) |
| #31848 | RDT weight transfer engine | **Only `sharded_rdt` exists**; no plain `rdt` backend or registry key |
| #31848 | NCCL M2N sharding-aware transfer (#46439) | `sharded_rdt` + `sparse_nccl` exist; `#46439` itself is a separate open issue |
| **#48306** | `/abort_requests` (#47173) | **Merged** (`rlhf/api_router.py:95`) |
| #48306 §2.2 | `finish_weight_update` counter hook (#39212 merged) | **Landed**, but as a caller-supplied **opaque string**, not a counter (`core.py:1042-1047`) |
| #48306 §2.2 | `/weight_info` + version tagging "not upstreamed" | **Now upstreamed** by #49040 (`rlhf/api_router.py:222`), with per-request binding deliberately removed |
| #48306 §2.2 | Pause-state metric (#45524) | **Open** |
| #48306 §2.2 | Version metadata in rollout responses | **NOT PRESENT**; contract explicitly open (maintainer comment, 2026-09-16) |
| **#48305** §3.1 | `trace_decode_token_ids` (#46701) | **Landed** (`sampling_params.py:374`); **requires MRV2** (`config/vllm.py:1266`) |
| #48305 §3.2 | R3 routing replay | **Partially** — capture/transport/response exist (`outputs.py:64`); FlashInfer + P/D + KV-offload combos still open per the RFC checklist |
| #48305 §3.3 | DSA index replay (#47280/#47279) | **NOT PRESENT** — no `enable_return_indexer_topk` |
| #48305 §3.4 | Artifact transfer connector (#47809) | **Not merged** (`closed`, `merged_at: null`) |
| #48305 §3.6 | Dtype replay (#48390) | **Merged** 2026-07-13 |
| **#48312** | Cat. 7: "#48762 or an equivalent non-reverted fix lands" | **Not landed** — `#48762` closed unmerged; `finish_weight_update` still invalidates no cache |
| #48312 | Cat. 1 fixes #48251 / #46009 / #41670 / #48438 / #48539 | Several still open per exit criteria; `#48478` production registry still **open** |
| #48312 | Composite lane `pause/drain → sleep → wake(weights) → update → post-load → wake(kv) → invalidate → resume` | **Not present in CI.** Closest in-tree test is `test_pause_resume.py` (pause/resume only, no weight update) |
| #48312 | Queryable generation identity / read-your-writes | **Not present** — `GET /weight_info` returns the last value *set*; no commit certificate |

---

## Appendix A — Endpoint reference (all require `VLLM_SERVER_DEV_MODE=1`)

| Method | Path | Engine method | Router |
|---|---|---|---|
| POST | `/pause?mode=&clear_cache=&wait_for_inflight_requests=` | `pause_generation` | `dev/rlhf/api_router.py:29` |
| POST | `/resume` | `resume_generation` | `:76` |
| POST | `/abort_requests` | `abort` | `:95` |
| GET | `/is_paused` | `is_paused` | `:139` |
| POST | `/init_weight_transfer_engine` | `init_weight_transfer_engine` | `:156` |
| POST | `/start_weight_update` | `start_weight_update` | `:174` |
| POST | `/start_draft_weight_update` | `start_draft_weight_update` | `:180` |
| POST | `/update_weights` | `update_weights` | `:186` |
| POST | `/finish_weight_update` (body `weight_version`) | `finish_weight_update` | `:204` |
| POST | `/update_weight_version` (body `new_version`) | `update_weight_version` | `:213` |
| GET | `/weight_info` | `get_weight_version` | `:222` |
| GET | `/get_world_size?include_dp=` | `vllm_config.parallel_config` | `:228` |
| POST | `/sleep?level=&mode=` | `sleep` | `dev/sleep/api_router.py:21` |
| POST | `/release_kv_cache_memory` | `release_kv_cache_memory` | `:30` |
| POST | `/wake_up?tags=` | `wake_up` | `:36` |
| GET | `/is_sleeping` | `is_sleeping` | `:47` |
| POST | `/reset_prefix_cache?reset_running_requests=&reset_external=` | `reset_prefix_cache` | `dev/cache/api_router.py:20` |
| POST | `/reset_mm_cache` | `reset_mm_cache` | `:47` |
| POST | `/reset_encoder_cache` | `reset_encoder_cache` | `:57` |
| POST | `/collective_rpc` (body `method`, `args`, `kwargs`, `timeout`) | `collective_rpc` | `dev/rpc/api_router.py:23` |
| GET | `/server_info?config_format=` | reads `app.state.vllm_config` | `dev/server_info/api_router.py:43` |

## Appendix B — Minimal vLLM launch for a rollout engine

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model> \
  --weight-transfer-config '{"backend": "nccl"}' \
  [--enforce-eager] [--enable-prefix-caching]
```
Precedent: `examples/rl/rlhf_http_nccl.py:63-83` (which also uses `--load-format dummy` to start before real weights exist).

## Appendix C — Raw research artifacts

Fetched RFC bodies and comment threads (untrusted upstream data, kept verbatim) in `research/`:
`rfc_48314.json`, `rfc_48311.json`, `rfc_31848.json`, `rfc_48305.json`, `rfc_48306.json`, `rfc_48312.json`, `comments_48314.md`, `comments_31848.md`, `comments_48306.md`, `comments_48312.md`.
