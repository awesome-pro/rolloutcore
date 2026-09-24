#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3A: live control-plane smoke test against a real ``vllm serve``.

This is an **adapter-level integration test**, not a state-machine test. It does
not fabricate a weight update to reach ``READY`` again, because Phase 3A has no
NCCL driver; the planner's instruction is explicit that a fake no-op update must
not be added merely to make ``LifecycleRunner.run_cycle()`` usable here. The
full nine-state cycle becomes real in Phase 3C.

What it proves, in order, against a real server:

 1. ``GET /weight_info`` reports the unmanaged literal ``"default"``.
 2. ``HttpVLLMAdapter.bootstrap()`` with the default ``LifecycleOnlyDriver``
    seeds ``rc-0`` without initializing a transfer engine, and the controller
    reaches ``READY``.
 3. A *second* bootstrap against the same engine raises
    ``AlreadyManagedEngineError`` and leaves ``rc-0`` untouched -- the refusal
    happens before any write.
 4. Greedy generation is deterministic (two identical requests, identical text
    and token ids), and the rollout binding carries the committed identity.
 5. ``pause(mode=wait)`` does not block the RolloutCore side, and the engine's
    drain genuinely *waits* for an in-flight streaming request to finish.
 6. ``GET /is_paused`` reports ``true``, and the label is still ``rc-0``.
 7. ``/reset_prefix_cache``, ``/reset_encoder_cache`` and ``/reset_mm_cache``
    all succeed. (The pause's ``clear_cache=true`` already ran; this is the
    explicit re-assertion the design requires.)
 8. ``validate_pre_resume`` sees ``rc-0`` **and** still-paused.
 9. ``/resume`` is acknowledged and ``GET /is_paused`` reports ``false``.
10. Greedy generation is deterministic again, and matches the pre-pause output
    (cache-coherence signal: same weights, cold caches, same tokens).

It writes a structured JSON artifact (``results/phase3a.json`` by default) with
the vLLM version/commit, the RolloutCore commit, the GPU, the model, every
check, and the lifecycle timings.

At the end the controller is **deliberately tainted**: the engine was resumed by
the adapter *outside* the controller (V1's table has no update-free revalidation
path -- ``docs/state-machine.md`` section 12 item 5), and a controller that
cannot vouch for a serving engine must not pretend otherwise. The report marks
this ``expected``.

Usage
-----

Launch and drive a server in one shot (the turnkey path on a GPU box)::

    python3 scripts/live_control_plane_smoke.py \\
        --launch --model facebook/opt-125m \\
        --vllm-sha 00b7847c8036b667742b4efb21aab1de51fd4721

Attach to a server you started yourself::

    VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 \\
      vllm serve facebook/opt-125m --enforce-eager --port 8000 &
    python3 scripts/live_control_plane_smoke.py --base-url http://127.0.0.1:8000

Dry-run the whole harness with no GPU and no vLLM, against the in-repo stub::

    PYTHONPATH=src:tests python3 tests/fake_dev_server.py --port 8123 &
    python3 scripts/live_control_plane_smoke.py --base-url http://127.0.0.1:8123

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:  # allow running straight from a checkout
    sys.path.insert(0, str(_REPO / "src"))

from rolloutcore import (  # noqa: E402 - path bootstrap must precede the import
    AlreadyManagedEngineError,
    DrainFailedError,
    EvidenceNotReady,
    LifecycleController,
    LifecycleState,
    ParamSpec,
    UpdateTarget,
    WeightIdentity,
    WeightProvenance,
)
from rolloutcore.adapters.http import (  # noqa: E402
    HTTPAdapterError,
    HttpVLLMAdapter,
    Response,
    UrllibTransport,
)

DEFAULT_VLLM_SHA = "00b7847c8036b667742b4efb21aab1de51fd4721"
DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_PROMPT = "The capital of France is"

#: The refusal reason written into the report. Kept in one place so the JSON and
#: the stdout summary cannot drift.
TAINT_REASON = (
    "Phase 3A control-plane smoke resumed the engine through the adapter, "
    "outside the controller, because V1's table has no update-free revalidation "
    "path (docs/state-machine.md section 12 item 5). No update is fabricated. "
    "The controller cannot vouch for a serving engine it did not resume, so it "
    "is tainted deliberately."
)


class SmokeFailure(RuntimeError):
    """A check failed. The run aborts; every check already recorded is kept."""


@dataclass
class Check:
    name: str
    status: str  # pass | fail | warn
    detail: str = ""
    seconds: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    base_url: str = "http://127.0.0.1:8000"
    model: str = DEFAULT_MODEL
    launch: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    vllm_sha: str = DEFAULT_VLLM_SHA
    json_out: Path = Path("results/phase3a.json")
    server_log: Path = Path("results/phase3a-server.log")
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.6
    baseline_max_tokens: int = 16
    long_max_tokens: int = 256
    generation_timeout: float = 180.0
    nonblocking_budget: float = 0.5
    drain_timeout: float = 120.0
    drain_polls: int = 600
    drain_interval: float = 0.25
    strict_generation: bool = True
    identity_source: str = "declared"  # declared | manifest
    reset_label_to_default: bool = False
    prompt: str = DEFAULT_PROMPT
    health_timeout: float = 900.0


# --------------------------------------------------------------------- HTTP


class DevClient:
    """Direct dev-endpoint client, deliberately independent of the adapter.

    The checks that observe engine state must not go through the code under
    test, or a bug in the adapter would hide itself.
    """

    def __init__(self, base_url: str, transport: UrllibTransport, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.timeout = timeout

    def request(
        self, method: str, path: str, *, body: dict[str, Any] | None = None, query: str = ""
    ) -> Response:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        return self.transport.request(
            method, f"{self.base_url}{path}{query}", payload, self.timeout
        )

    def get_json(self, path: str, query: str = "") -> Any:
        resp = self.request("GET", path, query=query)
        if resp.status >= 400:
            hint = (
                " (a dev endpoint is missing: is VLLM_SERVER_DEV_MODE=1 set on the server?)"
                if resp.status == 404
                else ""
            )
            raise SmokeFailure(f"GET {path} -> HTTP {resp.status}: {resp.body[:200]}{hint}")
        return resp.json()

    def post(self, path: str, body: dict[str, Any] | None = None, query: str = "") -> Response:
        return self.request("POST", path, body=body, query=query)

    def weight_version(self) -> str:
        body = self.get_json("/weight_info")
        label = body.get("weight_version") if isinstance(body, dict) else None
        if not isinstance(label, str):
            raise SmokeFailure(f"/weight_info returned no weight_version: {body!r}")
        return label

    def is_paused(self) -> bool:
        body = self.get_json("/is_paused")
        return bool(body.get("is_paused")) if isinstance(body, dict) else False

    def healthy(self) -> bool:
        try:
            return self.request("GET", "/health").status < 400
        except Exception:
            return False


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    chunks: int
    seconds: float
    finish_reason: str | None

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "text_sha256": self.text_sha256,
            "token_ids": self.token_ids,
            "num_tokens": len(self.token_ids),
            "chunks": self.chunks,
            "seconds": round(self.seconds, 4),
            "finish_reason": self.finish_reason,
        }


class CompletionsClient:
    """Greedy, seeded, OpenAI-compatible completion client over urllib."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/completions"
        self.model = model
        self.timeout = timeout

    def _payload(self, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "return_token_ids": True,
            "stream": stream,
        }

    def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        payload = json.dumps(self._payload(prompt, max_tokens, stream=False)).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, method="POST", headers={"Content-Type": "application/json"}
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SmokeFailure(
                f"POST /v1/completions -> HTTP {exc.code}: {exc.read().decode('utf-8')[:400]}"
            ) from exc
        choices = body.get("choices") or []
        if not choices:
            raise SmokeFailure(f"/v1/completions returned no choices: {body!r}")
        choice = choices[0]
        return Completion(
            text=str(choice.get("text") or ""),
            token_ids=[int(t) for t in (choice.get("token_ids") or [])],
            chunks=1,
            seconds=time.monotonic() - started,
            finish_reason=choice.get("finish_reason"),
        )

    def stream_in_thread(
        self, prompt: str, *, max_tokens: int, on_first_chunk: Callable[[], None]
    ) -> tuple[threading.Thread, dict[str, Any]]:
        """Run a streaming completion on a worker thread.

        Returns the thread and a mutable box. The box holds ``result`` (a
        :class:`Completion`), or ``error``, and always ``done``.
        """
        box: dict[str, Any] = {"done": False}
        payload = json.dumps(self._payload(prompt, max_tokens, stream=True)).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload, method="POST", headers={"Content-Type": "application/json"}
        )

        def worker() -> None:
            pieces: list[str] = []
            tokens: list[int] = []
            chunks = 0
            first = True
            started = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        if first:
                            first = False
                            on_first_chunk()
                        chunks += 1
                        piece = choices[0].get("text")
                        if isinstance(piece, str):
                            pieces.append(piece)
                        for token in choices[0].get("token_ids") or []:
                            tokens.append(int(token))
                box["result"] = Completion(
                    text="".join(pieces),
                    token_ids=tokens,
                    chunks=chunks,
                    seconds=time.monotonic() - started,
                    finish_reason="stop",
                )
            except Exception as exc:  # reported through the box, never raised on the thread
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                box["done"] = True

        thread = threading.Thread(target=worker, name="smoke-inflight", daemon=True)
        thread.start()
        return thread, box


# ---------------------------------------------------------------- launcher


@dataclass
class ServerHandle:
    process: subprocess.Popen[str]
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - slow shutdown
            self.process.kill()

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "<no log>"


def launch_server(cfg: Config) -> ServerHandle:
    """Start ``vllm serve`` with the env vars Phase 3A requires."""
    exe = shutil.which("vllm")
    if exe is None:
        raise SmokeFailure(
            "`vllm` is not on PATH; activate the environment where the pinned "
            "vLLM is installed (see docs/phase3a-runbook.md)"
        )
    version = subprocess.run(
        [exe, "--version"], capture_output=True, text=True, check=False, timeout=60
    )
    reported = (version.stdout or version.stderr).strip().splitlines()
    print(f"   vllm     : {exe}")
    print(f"   version  : {reported[0] if reported else 'unknown'}")
    cfg.server_log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    cmd = [
        exe,
        "serve",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--enforce-eager",
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    log = cfg.server_log.open("w", encoding="utf-8")
    log.write("# " + " ".join(cmd) + "\n")
    log.write("# VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1\n")
    log.flush()
    process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return ServerHandle(process=process, log_path=cfg.server_log)


# ------------------------------------------------------------------- smoke


class Smoke:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.checks: list[Check] = []
        self.timings: dict[str, float] = {}
        self.extra: dict[str, Any] = {}
        self.environment: dict[str, Any] = {}
        self.generation: dict[str, Any] = {}
        self.started_at = dt.datetime.now(dt.UTC)
        self.ok = False
        self.transport = UrllibTransport()
        self.dev = DevClient(cfg.base_url, self.transport, timeout=30.0)
        self.completions = CompletionsClient(cfg.base_url, cfg.model, cfg.generation_timeout)
        self.ctrl = LifecycleController()
        self.adapter = HttpVLLMAdapter(
            base_url=cfg.base_url,
            weight_identity=self.declared_identity(),
            transport=self.transport,
            drain_timeout=cfg.drain_timeout,
        )
        self.target: UpdateTarget | None = None

    # ---------------------------------------------------------- bookkeeping

    def record(
        self,
        name: str,
        status: str,
        detail: str = "",
        seconds: float | None = None,
        **data: Any,
    ) -> None:
        self.checks.append(Check(name, status, detail, seconds, data))
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
        timing = f" ({seconds:.3f}s)" if seconds is not None else ""
        print(f"  [{mark}] {name}{timing}")
        if detail:
            print(f"         {detail}")

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise SmokeFailure(message)

    def checked(self, name: str, fn: Callable[[], Any], *, detail: str = "") -> Any:
        """Run one check, recording pass/fail and its duration."""
        started = time.monotonic()
        try:
            result = fn()
        except SmokeFailure as exc:
            # A `require()` inside the body: still record it under this check's
            # name, or the report would only carry a bare `run_aborted`.
            self.record(name, "fail", str(exc), time.monotonic() - started)
            raise
        except Exception as exc:
            self.record(name, "fail", f"{type(exc).__name__}: {exc}", time.monotonic() - started)
            raise SmokeFailure(f"{name}: {exc}") from exc
        self.record(name, "pass", detail, time.monotonic() - started)
        return result

    # --------------------------------------------------------- environment

    def declared_identity(self) -> WeightIdentity:
        provenance = WeightProvenance(
            checkpoint=f"{self.cfg.model}@{self._hf_revision() or 'unknown-revision'}"
        )
        if self.cfg.identity_source == "manifest":
            specs = self._model_manifest()
            if specs:
                return WeightIdentity.from_param_specs(specs, source=provenance)
        return WeightIdentity.from_param_specs([], source=provenance)

    def _hf_revision(self) -> str | None:
        try:
            from huggingface_hub import model_info  # type: ignore[import-not-found]
        except Exception:
            return None
        try:
            return str(model_info(self.cfg.model).sha)
        except Exception:
            return None

    def _model_manifest(self) -> list[ParamSpec]:
        """Real parameter manifest, if transformers+torch are importable.

        Only used with ``--identity-source manifest``: it is the honest way to
        get a manifest digest for the bootstrap identity, and it is exactly what
        a Phase 3B driver will do from ``WeightSource.metadata()``.
        """
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
        except Exception:
            return []
        try:
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model, torch_dtype=torch.float16)
            return [
                ParamSpec(name, str(param.dtype).replace("torch.", ""), tuple(param.shape))
                for name, param in model.named_parameters()
            ]
        except Exception:
            return []

    def capture_environment(self) -> None:
        vllm_sha, vllm_version = git_rev(_REPO), None
        dirty = git_dirty(_REPO)
        server_version: Any = None
        try:
            server_version = self.dev.get_json("/version")
        except Exception as exc:
            server_version = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(server_version, dict):
            vllm_version = server_version.get("version")
        self.environment = {
            "vllm_sha_expected": self.cfg.vllm_sha,
            "vllm_version_reported": vllm_version,
            "vllm_sha_matches_report": _sha_in_version(self.cfg.vllm_sha, vllm_version),
            "rolloutcore_sha": vllm_sha,
            "rolloutcore_dirty": dirty,
            "model": self.cfg.model,
            "base_url": self.cfg.base_url,
            "gpu": gpu_info(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python3": sys.version.split()[0],
            "identity_source": self.cfg.identity_source,
            "bootstrap_weight_identity": self.adapter.weight_identity.describe()
            if self.adapter.weight_identity
            else None,
        }
        status = "pass" if self.environment["vllm_sha_matches_report"] else "warn"
        detail = (
            f"vLLM reports {vllm_version!r}; expected commit {self.cfg.vllm_sha}"
            if status == "warn"
            else f"vLLM reports {vllm_version!r}, matching {self.cfg.vllm_sha}"
        )
        self.record("environment", status, detail, **self.environment)

    # -------------------------------------------------------------- checks

    def wait_for_health(self) -> None:
        started = time.monotonic()
        deadline = started + self.cfg.health_timeout
        while time.monotonic() < deadline:
            if self.dev.healthy():
                self.record(
                    "server_healthy",
                    "pass",
                    f"{self.cfg.base_url}/health responded after {time.monotonic() - started:.1f}s",
                    time.monotonic() - started,
                )
                return
            time.sleep(1.0)
        raise SmokeFailure(
            f"server at {self.cfg.base_url} did not become healthy within "
            f"{self.cfg.health_timeout:.0f}s"
        )

    def chk_fresh_label(self) -> None:
        def body() -> str:
            label = self.dev.weight_version()
            if label.startswith("rc-") and self.cfg.reset_label_to_default:
                self.dev.post("/update_weight_version", {"new_version": "default"})
                self.record(
                    "label_reset_to_default",
                    "warn",
                    "server was already managed; reset the label to 'default' for a "
                    "re-run (dev mode only -- never do this in production)",
                )
                label = self.dev.weight_version()
            self.require(
                label == "default",
                f"/weight_info reported {label!r}; Phase 3A requires a *fresh* engine "
                "(restart the server, or pass --reset-label-to-default when re-running)",
            )
            return label

        label = self.checked("fresh_weight_info_is_default", body)
        self.extra["fresh_label"] = label

    def chk_bootstrap(self) -> None:
        started = time.monotonic()
        evidence = self.adapter.bootstrap()
        self.ctrl.initialize(evidence)
        self.timings["bootstrap_s"] = time.monotonic() - started
        self.require(self.ctrl.state is LifecycleState.READY, "bootstrap did not reach READY")
        self.require(self.ctrl.current_version is not None, "no committed target after bootstrap")
        assert self.ctrl.current_target is not None
        self.target = self.ctrl.current_target
        self.require(
            evidence.pre_seed_label == "default",
            f"pre-seed label was {evidence.pre_seed_label!r}, expected 'default'",
        )
        self.require(
            evidence.weight_transfer_driver is None and not evidence.weight_transfer_initialised,
            "control-plane-only bootstrap must not claim a transfer engine",
        )
        self.record(
            "control_plane_bootstrap_seeds_rc0",
            "pass",
            f"pre_seed={evidence.pre_seed_label!r} -> {evidence.observed_engine_label!r}, "
            f"driver={evidence.weight_transfer_driver!r}, "
            f"world_size={evidence.world_size}",
            self.timings["bootstrap_s"],
            observed_label=evidence.observed_engine_label,
            pre_seed_label=evidence.pre_seed_label,
            weight_transfer_driver=evidence.weight_transfer_driver,
            world_size=evidence.world_size,
            identity=evidence.weight_identity.describe(),
        )

    def chk_managed_refusal(self) -> None:
        """A second bootstrap must refuse *and* leave the label alone."""
        assert self.target is not None
        second = HttpVLLMAdapter(
            base_url=self.cfg.base_url,
            weight_identity=self.adapter.weight_identity,
            transport=self.transport,
        )
        started = time.monotonic()
        raised: AlreadyManagedEngineError | None = None
        try:
            second.bootstrap()
        except AlreadyManagedEngineError as exc:
            raised = exc
        seconds = time.monotonic() - started
        if raised is None:
            raise SmokeFailure("a second bootstrap on a managed engine did not refuse")
        label_after = self.dev.weight_version()
        self.require(
            label_after == "rc-0", f"refused adoption changed the label to {label_after!r}"
        )
        self.record(
            "second_bootstrap_refused_without_write",
            "pass",
            f"AlreadyManagedEngineError(observed={raised.observed_label!r}); "
            f"/weight_info still {label_after!r}",
            seconds,
            observed_label=raised.observed_label,
            label_after=label_after,
        )

    def chk_baseline_generation(self) -> None:
        assert self.target is not None
        binding = self.ctrl.admit_rollout("smoke-R1")
        first = self.completions.complete(self.cfg.prompt, max_tokens=self.cfg.baseline_max_tokens)
        second = self.completions.complete(self.cfg.prompt, max_tokens=self.cfg.baseline_max_tokens)
        self.ctrl.finish_rollout("smoke-R1")
        same = first.text == second.text and first.token_ids == second.token_ids
        self.require(
            same,
            f"greedy generation is not deterministic: {first.text!r} vs {second.text!r}",
        )
        self.generation["baseline"] = first.as_dict()
        self.generation["baseline_repeat"] = second.as_dict()
        self.generation["prompt"] = self.cfg.prompt
        self.record(
            "deterministic_generation_before_pause",
            "pass",
            f"{len(first.token_ids)} tokens, sha256={first.text_sha256[:16]}, "
            f"identical across two requests",
            first.seconds,
            text=first.text,
            token_ids=first.token_ids,
            finished_with=first.finish_reason,
        )
        self.record(
            "rollout_binding_carries_identity",
            "pass",
            f"R1 bound to {binding.version.label} / {binding.weight_identity.short} "
            f"cache_salt={binding.cache_salt}",
            version=binding.version.label,
            identity=binding.weight_identity.describe(),
            cache_salt=binding.cache_salt,
        )

    def chk_drain_with_inflight(self) -> None:
        """The heart of Phase 3A: the pause must wait, and must not block us."""
        first_chunk = threading.Event()
        thread, box = self.completions.stream_in_thread(
            self.cfg.prompt, max_tokens=self.cfg.long_max_tokens, on_first_chunk=first_chunk.set
        )
        self.require(
            first_chunk.wait(self.cfg.generation_timeout),
            "the in-flight request never produced a first token",
        )
        self.record(
            "inflight_request_started",
            "pass",
            f"streaming {self.cfg.long_max_tokens} tokens, first chunk received",
        )
        try:
            data = self._drain_phase(thread, box)
        except (DrainFailedError, HTTPAdapterError) as exc:
            # The likeliest real-world Phase 3A failure, so it gets its own
            # actionable check rather than a bare traceback.
            detail = (
                f"{type(exc).__name__}: {exc} -- the server refused or could not finish "
                "pause(mode='wait'). An in-process engine refuses it outright "
                "(vllm/v1/engine/core.py:902), so check VLLM_ENABLE_V1_MULTIPROCESSING=1 "
                "and that the engine core is out of process."
            )
            self.record("pause_mode_wait_supported", "fail", detail)
            self.record("pause_waits_for_inflight_request", "fail", detail)
            raise SmokeFailure("pause(mode='wait') was not usable on this server") from exc
        except SmokeFailure as exc:
            self.record("pause_waits_for_inflight_request", "fail", str(exc))
            raise
        self.record(
            "pause_waits_for_inflight_request",
            "pass",
            f"request produced {data['tokens']} tokens in {data['request_seconds']:.2f}s; "
            f"drain completed {data['polls']} poll(s) later",
            data["generation_s"],
            polls=data["polls"],
            tokens=data["tokens"],
            request_seconds=round(data["request_seconds"], 4),
            drain_seconds=round(data["drain_seconds"], 4),
        )

    def _drain_phase(self, thread: threading.Thread, box: dict[str, Any]) -> dict[str, Any]:
        # READY -> DRAINING (controller), then the real pause (adapter).
        self.ctrl.begin_drain()
        started = time.monotonic()
        self.adapter.begin_drain()
        begin_s = time.monotonic() - started
        self.timings["begin_drain_s"] = begin_s
        self.require(
            begin_s < self.cfg.nonblocking_budget,
            f"begin_drain blocked for {begin_s:.3f}s (budget {self.cfg.nonblocking_budget}s)",
        )

        started = time.monotonic()
        evidence = self.adapter.await_drain()
        poll_s = time.monotonic() - started
        self.timings["first_await_drain_s"] = poll_s
        self.require(
            poll_s < self.cfg.nonblocking_budget,
            f"await_drain blocked for {poll_s:.3f}s (budget {self.cfg.nonblocking_budget}s)",
        )
        self.require(
            not evidence.engine_drain_completed,
            "the engine reported a completed drain while a request was still streaming",
        )
        self.require(
            not box.get("done", False),
            "the in-flight request finished before the first poll; the drain was not tested",
        )
        self.record(
            "pause_is_nonblocking_on_the_rolloutcore_side",
            "pass",
            f"begin_drain returned in {begin_s:.3f}s, first await_drain in {poll_s:.3f}s, "
            f"engine_drain_completed=False while generation continued",
            begin_s,
            begin_drain_s=round(begin_s, 4),
            first_await_drain_s=round(poll_s, 4),
            evidence_note=evidence.note,
        )

        # The long request must be allowed to finish, then the drain completes.
        started = time.monotonic()
        thread.join(timeout=self.cfg.generation_timeout)
        generation_s = time.monotonic() - started
        self.require(not thread.is_alive(), "the in-flight request did not finish")
        self.require("error" not in box, f"the in-flight request failed: {box.get('error')}")
        result = box.get("result")
        self.require(isinstance(result, Completion), "the in-flight request produced no result")
        assert isinstance(result, Completion)
        self.require(
            len(result.token_ids) >= self.cfg.long_max_tokens,
            f"in-flight request produced {len(result.token_ids)} tokens, "
            f"expected >= {self.cfg.long_max_tokens}",
        )
        self.generation["inflight"] = result.as_dict()

        polls = 1
        deadline = time.monotonic() + self.cfg.drain_timeout
        while True:
            evidence = self.adapter.await_drain()
            try:
                self.ctrl.confirm_drained(evidence)
                break
            except EvidenceNotReady:
                polls += 1
                self.require(
                    time.monotonic() < deadline,
                    f"drain did not complete within {self.cfg.drain_timeout}s",
                )
                time.sleep(self.cfg.drain_interval)
        self.timings["drain_after_inflight_s"] = time.monotonic() - started
        self.require(self.ctrl.state is LifecycleState.QUIESCED, "controller is not QUIESCED")
        return {
            "polls": polls,
            "tokens": len(result.token_ids),
            "request_seconds": result.seconds,
            "drain_seconds": self.timings["drain_after_inflight_s"],
            "generation_s": generation_s,
        }

    def chk_engine_paused(self) -> None:
        assert self.target is not None
        paused = self.dev.is_paused()
        label = self.dev.weight_version()
        self.require(paused, "GET /is_paused reported false after a completed drain")
        self.require(label == self.target.label, f"label drifted to {label!r} during the drain")
        self.record(
            "engine_reports_paused",
            "pass",
            f"is_paused=true, weight_version={label!r}",
            is_paused=paused,
            weight_version=label,
        )

    def chk_cache_reset(self) -> None:
        assert self.target is not None
        started = time.monotonic()
        evidence = self.adapter.invalidate_caches(self.target)
        seconds = time.monotonic() - started
        self.timings["cache_reset_s"] = seconds
        raw = {
            "prefix": self.dev.post(
                "/reset_prefix_cache",
                query="?reset_running_requests=true&reset_external=true",
            ),
            "encoder": self.dev.post("/reset_encoder_cache"),
            "mm": self.dev.post("/reset_mm_cache"),
        }
        statuses = {name: resp.status for name, resp in raw.items()}
        self.require(
            evidence.failure_reason() is None,
            f"cache invalidation incomplete: {evidence.failure_reason()}",
        )
        self.require(
            all(status < 400 for status in statuses.values()),
            f"a reset endpoint failed: {statuses}",
        )
        self.record(
            "cache_resets_succeed",
            "pass",
            f"prefix={evidence.prefix_cache_reset} encoder={evidence.encoder_cache_reset} "
            f"mm={evidence.mm_cache_reset}; raw statuses {statuses}",
            seconds,
            prefix=evidence.prefix_cache_reset,
            encoder=evidence.encoder_cache_reset,
            mm=evidence.mm_cache_reset,
            raw_statuses=statuses,
        )

    def chk_validate(self) -> None:
        assert self.target is not None
        started = time.monotonic()
        evidence = self.adapter.validate_pre_resume(self.target)
        seconds = time.monotonic() - started
        self.timings["validate_s"] = seconds
        self.require(
            evidence.observed_engine_label == self.target.label,
            f"validation saw {evidence.observed_engine_label!r}, expected {self.target.label!r}",
        )
        self.require(evidence.is_paused, "validation saw an unpaused engine")
        self.require(evidence.failure_reason() is None, f"{evidence.failure_reason()}")
        self.record(
            "pre_resume_validation_sees_rc0_and_paused",
            "pass",
            f"weight_version={evidence.observed_engine_label!r} is_paused={evidence.is_paused}",
            seconds,
            observed_engine_label=evidence.observed_engine_label,
            is_paused=evidence.is_paused,
        )

    def chk_resume(self) -> None:
        assert self.target is not None
        started = time.monotonic()
        evidence = self.adapter.resume(self.target)
        seconds = time.monotonic() - started
        self.timings["resume_s"] = seconds
        self.require(evidence.resume_acknowledged, "the engine did not acknowledge /resume")
        self.require(not evidence.is_paused, "GET /is_paused still reports paused after resume")
        self.require(evidence.failure_reason() is None, f"{evidence.failure_reason()}")
        self.require(not self.dev.is_paused(), "independent /is_paused check still reports paused")
        self.record(
            "resume_succeeds",
            "pass",
            f"acknowledged, is_paused={evidence.is_paused}, "
            f"weight_version={evidence.observed_engine_label!r}",
            seconds,
            resume_acknowledged=evidence.resume_acknowledged,
            is_paused=evidence.is_paused,
            observed_engine_label=evidence.observed_engine_label,
        )

    def chk_post_resume_generation(self) -> None:
        after = self.completions.complete(self.cfg.prompt, max_tokens=self.cfg.baseline_max_tokens)
        baseline = self.generation.get("baseline")
        self.require(isinstance(baseline, dict), "no baseline generation to compare against")
        assert isinstance(baseline, dict)
        identical = after.text == baseline["text"] and after.token_ids == baseline["token_ids"]
        self.generation["post_resume"] = after.as_dict()
        self.generation["identical_to_baseline"] = identical
        if identical or not self.cfg.strict_generation:
            self.record(
                "deterministic_generation_after_resume",
                "pass" if identical else "warn",
                f"{len(after.token_ids)} tokens, sha256={after.text_sha256[:16]}, "
                + (
                    "matches the pre-pause output"
                    if identical
                    else "DIFFERS from the pre-pause output"
                ),
                after.seconds,
                text=after.text,
                token_ids=after.token_ids,
                identical=identical,
            )
            return
        self.record(
            "deterministic_generation_after_resume",
            "fail",
            "greedy output changed across pause/resume with identical weights: "
            f"{baseline['text']!r} -> {after.text!r} "
            "(pass --no-strict-generation to record this as a warning)",
            after.seconds,
            baseline_text=baseline["text"],
            post_resume_text=after.text,
            identical=False,
        )
        raise SmokeFailure("post-resume generation differs from the baseline")

    def end_controller(self) -> None:
        assert self.target is not None
        self.ctrl.taint(TAINT_REASON)
        self.record(
            "controller_tainted_by_design",
            "pass",
            f"final state {self.ctrl.state.value}; see the report's controller section",
            final_state=self.ctrl.state.value,
            expected=True,
        )

    # -------------------------------------------------------------- report

    def summary(self) -> dict[str, Any]:
        failed = [c.name for c in self.checks if c.status == "fail"]
        warned = [c.name for c in self.checks if c.status == "warn"]
        return {
            "phase": "3A",
            "ok": self.ok,
            "started_at": self.started_at.isoformat(),
            "finished_at": dt.datetime.now(dt.UTC).isoformat(),
            "vllm_sha": self.cfg.vllm_sha,
            "rolloutcore_sha": self.environment.get("rolloutcore_sha"),
            "model": self.cfg.model,
            "gpu": self.environment.get("gpu"),
            "environment": self.environment,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "seconds": None if c.seconds is None else round(c.seconds, 4),
                    "data": c.data,
                }
                for c in self.checks
            ],
            "failed_checks": failed,
            "warned_checks": warned,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "generation": self.generation,
            "cache_reset": {
                c.name: c.data for c in self.checks if c.name == "cache_resets_succeed"
            },
            "controller": {
                "final_state": self.ctrl.state.value,
                "expected_final_state": "TAINTED",
                "reason": TAINT_REASON,
                "committed_version": self.ctrl.current_version.label
                if self.ctrl.current_version
                else None,
                "committed_identity": self.ctrl.current_identity.describe()
                if self.ctrl.current_identity
                else None,
                "orphaned_rollouts": [o.request_id for o in self.ctrl.orphaned_rollouts],
                "journal_events": [r.event for r in self.ctrl.journal],
            },
            "notes": [
                "Phase 3A is an adapter-level control-plane smoke: no weight transfer is "
                "performed and no fake update is fabricated.",
                "The controller ends TAINTED on purpose: the adapter resumed the engine "
                "outside the controller, which V1 cannot model as a revalidation cycle.",
            ],
        }

    def write_report(self) -> Path:
        cfg = self.cfg
        cfg.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        tmp = cfg.json_out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        tmp.replace(cfg.json_out)
        return cfg.json_out


# ------------------------------------------------------------------ helpers


def git_rev(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def git_dirty(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(out.stdout.strip())
    except OSError:
        return False


def gpu_info() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except OSError:
        return []


def _sha_in_version(sha: str, version: Any) -> bool:
    if not isinstance(version, str):
        return False
    return sha[:8] in version or sha in version


# --------------------------------------------------------------------- main


def run(cfg: Config) -> int:
    smoke = Smoke(cfg)
    server: ServerHandle | None = None
    print("== RolloutCore Phase 3A control-plane smoke ==")
    print(f"   target   : {cfg.base_url}  model={cfg.model}")
    print(f"   vLLM sha : {cfg.vllm_sha}")
    print()
    try:
        if cfg.launch:
            print(f"   launching `vllm serve {cfg.model}` (log: {cfg.server_log})")
            server = launch_server(cfg)
        smoke.capture_environment()
        smoke.wait_for_health()
        smoke.chk_fresh_label()
        smoke.chk_bootstrap()
        smoke.chk_managed_refusal()
        smoke.chk_baseline_generation()
        smoke.chk_drain_with_inflight()
        smoke.chk_engine_paused()
        smoke.chk_cache_reset()
        smoke.chk_validate()
        smoke.chk_resume()
        smoke.chk_post_resume_generation()
        smoke.end_controller()
        smoke.ok = not any(c.status == "fail" for c in smoke.checks)
    except Exception as exc:
        # A harness must always produce its artifact: anything that escapes a
        # check becomes a recorded failure with its exception type, never a bare
        # traceback that loses the checks already gathered.
        smoke.ok = False
        smoke.record("run_aborted", "fail", f"{type(exc).__name__}: {exc}")
    finally:
        path = smoke.write_report()
        if server is not None:
            tail = server.log_tail()
            server.stop()
            print(f"\n   server log tail:\n{tail}")
        print(f"\n   report: {path}")
    passed = sum(1 for c in smoke.checks if c.status == "pass")
    warned = sum(1 for c in smoke.checks if c.status == "warn")
    failed = [c for c in smoke.checks if c.status == "fail"]
    print(f"   checks: {passed} passed, {warned} warned, {len(failed)} failed")
    for check in failed:
        print(f"     FAIL {check.name}: {check.detail}")
    print("   RESULT: " + ("PASS" if smoke.ok else "FAIL"))
    return 0 if smoke.ok else 1


def parse_args(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--launch",
        action="store_true",
        help="launch `vllm serve` ourselves (default: attach to --base-url)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--vllm-sha", default=DEFAULT_VLLM_SHA)
    ap.add_argument("--json-out", type=Path, default=Path("results/phase3a.json"))
    ap.add_argument("--server-log", type=Path, default=Path("results/phase3a-server.log"))
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--baseline-max-tokens", type=int, default=16)
    ap.add_argument("--long-max-tokens", type=int, default=256)
    ap.add_argument("--generation-timeout", type=float, default=180.0)
    ap.add_argument("--nonblocking-budget", type=float, default=0.5)
    ap.add_argument("--drain-timeout", type=float, default=120.0)
    ap.add_argument("--health-timeout", type=float, default=900.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--identity-source",
        choices=("declared", "manifest"),
        default="declared",
        help="declared: model@revision only; manifest: also hash the parameter manifest "
        "(loads the model client-side, needs torch+transformers)",
    )
    ap.add_argument(
        "--no-strict-generation",
        dest="strict_generation",
        action="store_false",
        help="record a post-resume output mismatch as a warning instead of a failure",
    )
    ap.add_argument(
        "--reset-label-to-default",
        action="store_true",
        help="dev-mode escape hatch to re-run against an already-managed server",
    )
    args = ap.parse_args(argv)
    base_url = args.base_url
    if args.launch and args.base_url == "http://127.0.0.1:8000":
        base_url = f"http://127.0.0.1:{args.port}"
    return Config(
        base_url=base_url,
        model=args.model,
        launch=args.launch,
        port=args.port,
        vllm_sha=args.vllm_sha,
        json_out=args.json_out,
        server_log=args.server_log,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        baseline_max_tokens=args.baseline_max_tokens,
        long_max_tokens=args.long_max_tokens,
        generation_timeout=args.generation_timeout,
        nonblocking_budget=args.nonblocking_budget,
        drain_timeout=args.drain_timeout,
        health_timeout=args.health_timeout,
        strict_generation=args.strict_generation,
        identity_source=args.identity_source,
        reset_label_to_default=args.reset_label_to_default,
        prompt=args.prompt,
    )


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
