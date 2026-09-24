#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4D: the engine dies during a weight update. Is it detected, and how fast?

Phase 4B killed the engine *before* a cycle and showed the two paths diverge: the
drain path does not taint (`_observe` never taints a failed read), the mutating
path does. This is the harder case -- the engine dies *inside* the update, when
the trainer may already be in a collective.

Why that could go badly: the trainer does not use `torch.distributed` with a
timeout. It forms the group through vLLM's own ``PyNcclCommunicator``
(``nccl_engine.py:156`` -> ``nccl_common.py:181``), and the only timeout in that
file is on *teardown* -- the code's own comment says a failed join "leaves the
peer blocked in ``ncclCommInitRank`` until timeout"
(``pynccl.py:237``). There is no ``NCCL_TIMEOUT``-style knob anywhere in the tree.
So a peer that disappears mid-collective may block far longer than an operator
would wait, and RolloutCore cannot bound what it does not control.

The measurement is therefore deliberately divided, because only one half is
reliably reached on a 125M model:

**Armed kill.** A healthy cycle first (which also times a real update), then a
second cycle to rc-2 is started on a daemon thread. The main thread watches
``ctrl.state`` and SIGKILLs the engine's process group the moment it reaches
UPDATING. With a ~250 MB broadcast the whole update is ~100 ms, so this lands in
the *early* phase -- the HTTP calls around the collective. That is the realistic
common case and it is where detection should be prompt.

**Not covered here: a kill inside the collective.** The window is too narrow to
hit reliably at 125M. It is covered by the benchmark run, where a multi-second
broadcast over a several-GB model makes the window wide.

The harness bounds its own wait (``DETECTION_BUDGET_S``) and exits via
``os._exit`` if the transfer thread is still blocked, so a hang costs the budget
rather than the NCCL timeout -- which is the whole point of measuring it.

Checks gate on fail-closed properties, not on the timings: nothing published, no
rollout admissible, engine dead. Whether that arrived as a taint or a hang is the
*result*, so it is recorded rather than asserted.

    python3 diagnostics/rolloutcore_updatekill_2gpu.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
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
    NotServingError,
)
from rolloutcore.adapters import HttpVLLMAdapter  # noqa: E402
from rolloutcore.adapters.nccl import NCCLWeightTransferDriver  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase4d-server.log"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
MAX_TOKENS = 16

#: How long to wait for the killed cycle to report *something*. If the transfer
#: thread is still blocked after this, that is a hang and it is the finding.
DETECTION_BUDGET_S = 120.0
#: How long to wait for the state to reach UPDATING before killing anyway.
ARM_TIMEOUT_S = 60.0
POLL_S = 0.05


# ------------------------------------------------------------------- server


def start_vllm_server() -> subprocess.Popen[str]:
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
        "dummy",
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
    with SERVER_LOG.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(args) + "\n")
        log.write("# VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1\n")
        log.flush()
        # `start_new_session=True` is required: the kill below signals a process
        # *group* (vLLM spawns an EngineCore child), and without its own session
        # the engine would share this harness's group.
        proc = subprocess.Popen(
            args,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    print(f"[server] log: {SERVER_LOG}")
    try:
        _await_ready(proc)
    except BaseException:
        _terminate(proc)
        raise
    return proc


def _own_group(proc: subprocess.Popen[str]) -> int | None:
    """The server's process group, or ``None`` if it is not its own leader."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:  # pragma: no cover - already gone
        return None
    return None if pgid == os.getpgid(0) else pgid


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Stop the engine's own group: SIGTERM, then SIGKILL. Idempotent."""
    if proc.poll() is not None:
        return
    pgid = _own_group(proc)
    if pgid is None:
        print("[warn] engine is not in its own process group; killing it alone")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=30)


def _kill_engine_hard(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the engine's whole group. Refuses if that group is ours."""
    pgid = _own_group(proc)
    if pgid is None:
        raise RuntimeError("refusing to killpg: the engine is not its own group")
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str]) -> None:
    """Block until ``/health`` answers (status only: 200 with an empty body)."""
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


def probe_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post(url: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else None


def generate() -> str:
    body = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": 0,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0].get("text", ""))


def weight_info() -> str:
    return str(get(f"{BASE_URL}/weight_info")["weight_version"])


def world_size() -> int:
    return int(get(f"{BASE_URL}/get_world_size")["world_size"])


def git_revision() -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    code = ("src", "tests", "scripts", "diagnostics")
    try:
        return run("rev-parse", "HEAD"), bool(
            run("status", "--porcelain", "--untracked-files=no", "--", *code)
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def vllm_version() -> str:
    try:
        return str(get(f"{BASE_URL}/version")["version"])
    except Exception:
        return "unknown"


def server_log_tail(lines: int = 25) -> str:
    try:
        text = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


def record(checks: dict[str, bool], name: str, ok: Any) -> None:
    checks[name] = bool(ok)
    print(f"  [{'PASS' if bool(ok) else 'FAIL'}] {name}")


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "4D", "model": MODEL_NAME, "ok": False}
    sha, dirty = git_revision()
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")

    server = start_vllm_server()
    hung = False
    try:
        report["vllm_version"] = vllm_version()
        print(f"[server] vllm {report['vllm_version']}")
        print(f"[trainer] loading {MODEL_NAME} on cuda:{TRAINER_DEVICE_INDEX}")
        torch.cuda.set_device(TRAINER_DEVICE_INDEX)
        train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        train_model.to(f"cuda:{TRAINER_DEVICE_INDEX}")

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
            source=ModuleSource(train_model),
            world_size=group_size,
        )
        identity = driver.identity()
        adapter = HttpVLLMAdapter(
            base_url=BASE_URL,
            weight_identity=identity,
            driver=driver,
            drain_timeout=600,
        )
        ctrl = LifecycleController()
        runner = LifecycleRunner(ctrl, adapter, drain_polls=600, drain_interval=2.0)

        print("[cycle] bootstrap")
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        assert weight_info() == "rc-0", weight_info()

        # ---- a healthy cycle first: the baseline, and a known-good rc-1 -------
        print("\n--- baseline: one healthy update, for its duration ---")
        started = time.monotonic()
        runner.run_cycle(ctrl.next_target(identity))
        healthy_seconds = round(time.monotonic() - started, 4)
        report["healthy_cycle_seconds"] = healthy_seconds
        record(checks, "baseline_cycle_committed_rc1", ctrl.current_version.label == "rc-1")
        print(f"[baseline] healthy cycle {healthy_seconds}s -> {weight_info()}")

        # ---- armed kill, during the second update ----------------------------
        print("\n--- armed: kill the engine the moment the update begins ---")
        before_kill_text = generate()
        record(checks, "engine_served_before_the_kill", bool(before_kill_text))
        outcome: dict[str, Any] = {}

        def cycle() -> None:
            began = time.monotonic()
            try:
                runner.run_cycle(ctrl.next_target(identity))
                outcome["returned"] = "committed"
            except BaseException as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            outcome["seconds"] = round(time.monotonic() - began, 4)

        worker = threading.Thread(target=cycle, name="rolloutcore-cycle", daemon=True)
        worker.start()

        # Watch for the update to begin. The controller is read from two threads
        # by design, and its transitions and views are locked.
        armed = time.monotonic()
        state_at_kill = LifecycleState.READY
        while time.monotonic() - armed < ARM_TIMEOUT_S:
            state_at_kill = ctrl.state
            if state_at_kill is LifecycleState.UPDATING:
                break
            time.sleep(POLL_S)
        killed_at = time.monotonic()
        phase_at_kill = state_at_kill.value
        print(f"[armed] state={phase_at_kill} after {killed_at - armed:.3f}s; SIGKILL")
        _kill_engine_hard(server)
        detail["kill"] = {
            "state_at_kill": phase_at_kill,
            "seconds_from_cycle_start_to_kill": round(killed_at - armed, 4),
            "healthy_cycle_seconds": healthy_seconds,
            "arm_timed_out": phase_at_kill != "UPDATING",
        }

        # ---- how long until the update reports anything? ---------------------
        worker.join(timeout=DETECTION_BUDGET_S)
        detection = round(time.monotonic() - killed_at, 4)
        hung = worker.is_alive()
        detail["detection"] = {
            "hung": hung,
            "seconds_from_kill": detection,
            "budget_seconds": DETECTION_BUDGET_S,
            "outcome": outcome.get("error") or outcome.get("returned"),
        }
        print(
            f"[armed] {'STILL BLOCKED' if hung else 'returned'} {detection}s after the "
            f"kill: {detail['detection']['outcome']}"
        )

        # ---- fail-closed, whichever way it went ------------------------------
        # `engine_is_dead` is a real request, not just a poll: the process being
        # gone and the engine no longer answering are different claims.
        try:
            generate()
            served_after_kill = True
        except Exception:
            served_after_kill = False
        record(
            checks,
            "engine_is_dead",
            server.poll() is not None and not served_after_kill,
        )
        record(checks, "no_version_was_published", ctrl.current_version.label == "rc-1")
        try:
            ctrl.admit_rollout("R-after-kill")
            admissible = True
        except NotServingError:
            admissible = False
        record(checks, "no_rollout_can_be_admitted", not admissible)
        record(
            checks,
            "update_did_not_report_success",
            outcome.get("returned") != "committed",
        )
        record(
            checks,
            "outcome_is_recorded",
            bool(outcome.get("error")) or hung,
        )
        # A hang is not a pass. The whole question is whether an operator learns
        # promptly, so "still blocked after the budget" fails explicitly rather
        # than hiding behind the fail-closed properties, which a hang satisfies.
        record(checks, "update_failure_detected_within_budget", not hung)
        detail["controller"] = {
            "state": ctrl.state.value,
            "tainted": ctrl.is_tainted,
            "taint_reason": ctrl.taint_reason,
            "serving": ctrl.is_serving,
            "committed": ctrl.current_version.label if ctrl.current_version else None,
        }
        print(
            f"[controller] state={detail['controller']['state']} "
            f"tainted={detail['controller']['tainted']} "
            f"serving={detail['controller']['serving']}"
        )
        if detail["controller"]["taint_reason"]:
            print(f"[controller] reason: {detail['controller']['taint_reason']}")

        report["detail"] = detail
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 4D ===")
        print(f"  baseline      : {healthy_seconds}s for one healthy update")
        print(f"  kill at       : {phase_at_kill} (+{killed_at - armed:.3f}s)")
        print(
            f"  detection     : {'HANG' if hung else f'{detection}s'} "
            f"-> {detail['detection']['outcome']}"
        )
        print(
            f"  controller    : state={detail['controller']['state']} "
            f"tainted={detail['controller']['tainted']}"
        )
        print(
            f"  RESULT: {'PASS' if report['ok'] else 'FAIL'} "
            f"({sum(checks.values())}/{len(checks)} checks)"
        )
        if hung:
            # A blocked transfer thread must not hold up interpreter shutdown --
            # bounding the hang is the entire point of the budget.
            _write(report, detail, checks)
            print("[exit] transfer thread still blocked; os._exit to bound the cost")
            sys.stdout.flush()
            os._exit(2)
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        print(f"[server] last lines:\n{server_log_tail()}")
        return 1
    finally:
        _write(report, detail, checks)
        _terminate(server)


def _write(report: dict[str, Any], detail: dict[str, Any], checks: dict[str, bool]) -> None:
    report["detail"] = detail
    report["checks"] = checks
    out = _REPO / "results" / "phase4d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    sys.exit(main())
