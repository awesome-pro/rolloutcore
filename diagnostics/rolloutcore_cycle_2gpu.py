#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3C: the first real RolloutCore cycle, with a real NCCL weight update.

Everything up to now tested a piece. This runs the whole thing end to end on one
two-GPU node:

    vllm serve (--load-format dummy)   GPU 0, gibberish weights
    trainer: facebook/opt-125m bf16    GPU 1, the real weights

    bootstrap  -> READY(rc-0)
    generate   -> gibberish
    DRAINING -> QUIESCED -> UPDATING (NCCL) -> INVALIDATING -> VALIDATING
             -> RESUMING -> READY(rc-1)
    generate   -> sensible, same vLLM PID

The cycle goes through ``LifecycleRunner.run_cycle`` -- the function Phase 3A
deliberately never called, because until the NCCL driver existed there was no
honest way to reach it.

What this proves that nothing else could:

* ``WeightTransferTrainerFactory`` + ``send_weights`` work through RolloutCore's
  driver, including the identity check computed from the trainer's own manifest;
* the engine reports ``rc-1`` **while still paused**, which only happens because
  :class:`~rolloutcore.adapters.nccl.RolloutCoreWeightSyncClient` supplies the
  version upstream omits (``nccl_engine.py:361``) -- without it VALIDATING taints;
* the commit point is the resume, not the finalize: ``current_version`` flips to
  ``rc-1`` only after ``/resume`` is acknowledged;
* the output changes from gibberish to coherent with no server restart.

Run on the two-GPU node, with vLLM installed and HF_HOME set:

    python3 diagnostics/rolloutcore_cycle_2gpu.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from vllm.distributed.weight_transfer import ModuleSource  # noqa: E402
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo  # noqa: E402
from vllm.utils.network_utils import get_ip, get_open_port  # noqa: E402

from rolloutcore import (  # noqa: E402
    LifecycleController,
    LifecycleRunner,
    LifecycleState,
)
from rolloutcore.adapters import HttpVLLMAdapter  # noqa: E402
from rolloutcore.adapters.nccl import NCCLWeightTransferDriver  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase3c-server.log"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
MAX_TOKENS = 16


# ------------------------------------------------------------------- server


def start_vllm_server() -> subprocess.Popen[str]:
    """Launch `vllm serve` with dummy weights and an NCCL transfer engine."""
    args = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
        "--device-ids",
        SERVER_DEVICE_IDS,
        "--enforce-eager",
        "--load-format",
        "dummy",  # gibberish baseline: the point of the before/after comparison
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        '{"backend": "nccl"}',
    ]
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    print(f"[server] {' '.join(args)}")
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Server output goes to a file, not this terminal: `vllm serve` logs every
    # /health poll, which would otherwise bury this script's own progress.
    with SERVER_LOG.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(args) + "\n")
        log.write("# VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1\n")
        log.flush()
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[server] log: {SERVER_LOG}")
    try:
        _await_ready(proc)
    except BaseException:
        # A readiness failure must not orphan a server still holding the port:
        # this function is what raised, so main()'s `finally` cannot run.
        _terminate(proc)
        raise
    return proc


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Stop the server: SIGTERM, then SIGKILL if it will not go."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str]) -> None:
    """Block until ``/health`` answers, or explain why it never did."""
    started = time.monotonic()
    deadline = started + 900
    last_beat = started
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming ready (see {SERVER_LOG})")
        if probe_ok(f"{BASE_URL}/health"):
            print(f"[server] ready after {time.monotonic() - started:.1f}s (pid {proc.pid})")
            return
        now = time.monotonic()
        if now - last_beat >= 15:
            print(f"[server] still loading ({now - started:.0f}s)")
            last_beat = now
        if now > deadline:
            raise RuntimeError(f"vLLM did not become ready in time (see {SERVER_LOG})")
        time.sleep(2)


# ------------------------------------------------------------------ http


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe_ok(url: str) -> bool:
    """Readiness probe: status only.

    ``/health`` is ``Response(status_code=200)`` with an *empty body*
    (``serve/instrumentator/health.py:28``), so it must never be fed to
    ``json.loads`` -- doing that raises, and a probe loop that swallows the
    exception never sees the server that is already up.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def post(url: str, body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else None


def weight_info() -> str:
    return str(get(f"{BASE_URL}/weight_info")["weight_version"])


def is_paused() -> bool:
    return bool(get(f"{BASE_URL}/is_paused")["is_paused"])


def world_size() -> int:
    return int(get(f"{BASE_URL}/get_world_size")["world_size"])


def git_revision() -> tuple[str, bool]:
    """``(HEAD, dirty)`` for this repo, tracked files only. Never fatal.

    ``--untracked-files=no`` matters: an untracked note in the working tree is
    not a code change, and letting it mark every run dirty would make the flag
    meaningless.
    """

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def vllm_version() -> str:
    """The server's own version string, so the artifact names its engine."""
    try:
        return str(get(f"{BASE_URL}/version")["version"])
    except Exception:
        return "unknown"


def server_log_tail(lines: int = 25) -> str:
    """Last few lines of the server log, for a self-diagnosing failure."""
    try:
        text = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


def generate() -> dict[str, Any]:
    body = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 0,
        "return_token_ids": True,
    }
    started = time.monotonic()
    payload = post(f"{BASE_URL}/v1/completions", body)
    choice = payload["choices"][0]
    return {
        "text": choice.get("text", ""),
        "token_ids": choice.get("token_ids", []),
        "seconds": round(time.monotonic() - started, 4),
        "weight_version": weight_info(),
    }


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "3C", "model": MODEL_NAME, "ok": False}
    sha, dirty = git_revision()
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")
    server = start_vllm_server()
    server_pid = server.pid
    try:
        report["vllm_version"] = vllm_version()
        print(f"[server] vllm {report['vllm_version']}")
        print(f"[trainer] loading {MODEL_NAME} on cuda:{TRAINER_DEVICE_INDEX}")
        torch.cuda.set_device(TRAINER_DEVICE_INDEX)
        train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        train_model.to(f"cuda:{TRAINER_DEVICE_INDEX}")

        source = ModuleSource(train_model)
        group_size = world_size() + 1  # every inference worker, plus this trainer
        driver = NCCLWeightTransferDriver(
            base_url=BASE_URL,
            trainer_init_info=NCCLTrainerInitInfo(
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=group_size,
                rank=0,  # single-GPU trainer is the sole sender
                packed=True,
            ),
            source=source,
            world_size=group_size,
        )
        # The identity comes from the trainer's own manifest, so the bootstrap
        # identity and what the driver will stage are computed the same way.
        identity = driver.identity()
        print(f"[identity] {identity.describe()}")

        adapter = HttpVLLMAdapter(
            base_url=BASE_URL,
            weight_identity=identity,
            driver=driver,
            drain_timeout=600,
        )
        ctrl = LifecycleController()
        runner = LifecycleRunner(ctrl, adapter)

        print("[cycle] bootstrap")
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        assert weight_info() == "rc-0", weight_info()

        before = generate()
        print(f"[before] {before['text']!r}  ({weight_info()})")

        binding = ctrl.admit_rollout("R1")
        ctrl.finish_rollout("R1")
        report["binding"] = {
            "request_id": binding.request_id,
            "version": binding.version.label,
            "identity": binding.weight_identity.describe(),
            "cache_salt": binding.cache_salt,
        }

        target = ctrl.next_target(identity)
        print(f"[cycle] run_cycle -> {target.describe()}")
        started = time.monotonic()
        result = runner.run_cycle(target)
        report["cycle_seconds"] = round(time.monotonic() - started, 4)
        report["states_visited"] = [s.value for s in result.states_visited]

        after = generate()
        print(f"[after]  {after['text']!r}  ({weight_info()})")

        report.update(
            {
                "world_size": group_size,
                "identity": identity.describe(),
                "before": before,
                "after": after,
                "committed_version": ctrl.current_version.label if ctrl.current_version else None,
                "engine_label_after": weight_info(),
                "engine_paused_after": is_paused(),
                "output_changed": before["text"] != after["text"],
                "server_pid": server_pid,
                "server_alive": server.poll() is None,
                "journal": [r.event for r in ctrl.journal],
            }
        )

        checks = {
            "committed_rc1": report["committed_version"] == "rc-1",
            "engine_label_rc1": report["engine_label_after"] == "rc-1",
            "engine_resumed": report["engine_paused_after"] is False,
            "output_changed": report["output_changed"],
            "server_never_restarted": report["server_alive"],
        }
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 3C ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  before: {before['text']!r}")
        print(f"  after : {after['text']!r}")
        print(f"  states: {' -> '.join(report['states_visited'])}")
        print(f"  RESULT: {'PASS' if report['ok'] else 'FAIL'}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        tail = server_log_tail()
        if tail:
            print(f"[server] last lines of {SERVER_LOG}:\n{tail}")
        return 1
    finally:
        out = _REPO / "results" / "phase3c.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")
        _terminate(server)


if __name__ == "__main__":
    sys.exit(main())
