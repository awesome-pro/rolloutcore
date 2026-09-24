#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 6: what a hot update costs, next to restarting the engine.

The headline number, and the last thing that needs a GPU. Three engine lifetimes:

**L1 -- the hot path.** A server booted from `--load-format dummy`, a trainer
holding the real checkpoint, and one `run_cycle`. A watcher samples the
controller's state so the cycle decomposes into per-state dwell times instead of a
single opaque total.

**L2 -- the restart baseline.** Kill the engine and start it again on the same
checkpoint, timed from the kill to `/health` *and* to the first token. This is
**deliberately generous to the baseline**: it assumes the new checkpoint is already
on disk. In a real restart-based update the trainer must also *write* a checkpoint
first, which this does not include. So the gap reported here is a floor on the
hot path's advantage, not a ceiling.

Why there should be a gap at all, when both paths move the same bytes: the restart
re-reads them from disk and rebuilds the engine, while the trainer already holds
them in GPU memory and the broadcast goes over the interconnect. The measured
split is in `detail["hot"]["states"]`.

**L3 -- the deep-broadcast kill.** 4D could only kill the engine in the early
phase of an update, because a 125M broadcast is ~100ms. Here the broadcast is
gigabytes, so the window is seconds wide: the kill is aimed at half of L1's
*measured* update duration, landing inside the collective. That is the case where
the trainer has no timeout of its own -- it joins through vLLM's
``PyNcclCommunicator``, whose only timeout is on teardown, and whose own comment
says a failed join "leaves the peer blocked in ``ncclCommInitRank`` until timeout"
(``pynccl.py:237``). The harness bounds its own wait and exits via ``os._exit`` if
the transfer thread is still blocked.

    python3 diagnostics/rolloutcore_benchmark_2gpu.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
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
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
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

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase6-server"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
MAX_TOKENS = 16

#: Small on purpose, and safe here: no rollout is in flight across L1's cycle, so
#: there is no client-side release racing the drain poll the way there was in 4A.
#: At the 4A value of 2.0s the poll would dominate the measurement -- Phase 4D
#: showed a "2.12s cycle" that was ~2s of sleep and ~120ms of work.
DRAIN_INTERVAL_S = 0.1
ARM_TIMEOUT_S = 600.0
DETECTION_BUDGET_S = 120.0
SAMPLE_S = 0.02


# ------------------------------------------------------------------- server


def start_server(
    model: str,
    log_name: str,
    *,
    dummy: bool,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> subprocess.Popen[str]:
    args = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
        "--device-ids",
        SERVER_DEVICE_IDS,
        "--enforce-eager",
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        '{"backend": "nccl"}',
        # Both of these exist to leave room for the weight-transfer receive
        # buffers. vLLM sizes the KV cache to fill the budget and does not reserve
        # any for them, and the packed consumer allocates per chunk, lazily:
        # `torch.empty(packing_tensor_sizes[buffer_idx])`. Qwen3-8B OOM'd on a
        # 1.16GiB receive against 531MiB free, which the trainer -- already inside
        # the broadcast -- experienced as a hang rather than an error.
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if dummy:
        args += ["--load-format", "dummy"]
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    log_path = SERVER_LOG.with_name(f"{log_name}.log")
    print(f"[server] {' '.join(args)}  (log: {log_path.name})")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(args) + "\n")
        log.write("# VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1\n")
        log.flush()
        proc = subprocess.Popen(
            args,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    try:
        _await_ready(proc, log_path)
    except BaseException:
        _terminate(proc)
        raise
    return proc


def _own_group(proc: subprocess.Popen[str]) -> int | None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:  # pragma: no cover - already gone
        return None
    return None if pgid == os.getpgid(0) else pgid


def _terminate(proc: subprocess.Popen[str]) -> None:
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
    pgid = _own_group(proc)
    if pgid is None:
        raise RuntimeError("refusing to killpg: the engine is not its own group")
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str], log_path: Path) -> None:
    """Status-only readiness probe: ``/health`` is 200 with an empty body."""
    started = time.monotonic()
    deadline = started + 1800
    last_beat = started
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming ready (see {log_path})")
        if probe_ok(f"{BASE_URL}/health"):
            print(f"[server] ready after {time.monotonic() - started:.1f}s (pid {proc.pid})")
            return
        now = time.monotonic()
        if now - last_beat >= 20:
            print(f"[server] still loading ({now - started:.0f}s)")
            last_beat = now
        if now > deadline:
            raise RuntimeError(f"vLLM did not become ready in time (see {log_path})")
        time.sleep(1)


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


def generate(model: str) -> str:
    body = {
        "model": model,
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


def _safe_weight_info() -> str:
    """The engine label, or a note if it cannot answer -- it may be blocked."""
    try:
        return weight_info()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def record(checks: dict[str, bool], name: str, ok: Any) -> None:
    checks[name] = bool(ok)
    print(f"  [{'PASS' if bool(ok) else 'FAIL'}] {name}")


def only_bos(text: str) -> bool:
    return bool(text) and not text.replace("<s>", "").strip()


class StateTimeline:
    """Samples the controller's state so a cycle decomposes into per-state dwell.

    Polling rather than hooking: the controller is read from two threads by design
    and its views are locked, so this needs no new API.
    """

    def __init__(self, ctrl: LifecycleController, interval: float = SAMPLE_S) -> None:
        self._ctrl = ctrl
        self._interval = interval
        self._events: list[tuple[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="state-timeline", daemon=True)

    def start(self) -> None:
        self._events.append((self._ctrl.state.value, time.monotonic()))
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            state = self._ctrl.state.value
            if state != self._events[-1][0]:
                self._events.append((state, time.monotonic()))
            self._stop.wait(self._interval)

    def stop(self) -> tuple[tuple[str, float], ...]:
        self._stop.set()
        self._thread.join(timeout=5)
        final = self._ctrl.state.value
        if final != self._events[-1][0]:
            self._events.append((final, time.monotonic()))
        return tuple(self._events)

    def durations(self) -> dict[str, float]:
        """Seconds spent in each state. The last state runs to the final sample."""
        out: dict[str, float] = {}
        for (state, at), (_next, next_at) in zip(self._events, self._events[1:], strict=False):
            out[state] = round(out.get(state, 0.0) + (next_at - at), 4)
        return out

    def total(self) -> float:
        if len(self._events) < 2:
            return 0.0
        return round(self._events[-1][1] - self._events[0][1], 4)


def build_engine(
    train_model: Any,
    model: str,
    *,
    packed_buffer_bytes: int | None = None,
    packed_num_buffers: int | None = None,
) -> tuple[Any, HttpVLLMAdapter, LifecycleController, LifecycleRunner]:
    group_size = world_size() + 1  # every inference worker, plus this trainer
    init_info: dict[str, Any] = {
        "master_address": get_ip(),
        "master_port": get_open_port(),
        "world_size": group_size,
        "rank": 0,  # single-GPU trainer is the sole sender
        "packed": True,
    }
    if packed_buffer_bytes is not None:
        # An 8B checkpoint's embedding matrix is 1.24GB and vLLM's default buffer is
        # 1GiB (packed_tensor.py:15). Both sides chunk with the same rule, so they
        # *should* agree -- but a divergence deadlocks an NCCL collective instead of
        # erroring, so this is a hypothesis to test cheaply, not debug in place.
        init_info["packed_buffer_size_bytes"] = packed_buffer_bytes
    if packed_num_buffers is not None:
        # Receive buffers are allocated per chunk on the worker and nothing
        # reserves room for them, so halving their number is the cheapest way to
        # buy headroom on a card that is already full of weights.
        init_info["packed_num_buffers"] = packed_num_buffers
    driver = NCCLWeightTransferDriver(
        base_url=BASE_URL,
        trainer_init_info=NCCLTrainerInitInfo(**init_info),
        source=ModuleSource(train_model),
        world_size=group_size,
    )
    adapter = HttpVLLMAdapter(
        base_url=BASE_URL,
        weight_identity=driver.identity(),
        driver=driver,
        drain_timeout=600,
    )
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, drain_polls=6000, drain_interval=DRAIN_INTERVAL_S)
    return driver, adapter, ctrl, runner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="the checkpoint both sides use")
    ap.add_argument("--skip-deep-kill", action="store_true")
    ap.add_argument(
        "--packed-buffer-gib",
        type=float,
        default=None,
        help="override NCCLTrainerInitInfo.packed_buffer_size_bytes (default 1GiB)",
    )
    ap.add_argument(
        "--packed-num-buffers",
        type=int,
        default=None,
        help="override NCCLTrainerInitInfo.packed_num_buffers (vLLM default 2)",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="lower than the 0.9 default on purpose: reserve room for transfer buffers",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="keeps the KV cache small, so the transfer buffers have somewhere to go",
    )
    ap.add_argument(
        "--transfer-budget",
        type=float,
        default=180.0,
        help="seconds to wait for one hot update before declaring a hang",
    )
    args = ap.parse_args()
    model = args.model
    packed_bytes = None if args.packed_buffer_gib is None else int(args.packed_buffer_gib * 1024**3)
    packed_buffers = args.packed_num_buffers
    # A cap above the checkpoint's own context is a hard error in vLLM, not a
    # warning: "User-specified max_model_len (4096) is greater than the derived
    # max_model_len (max_position_embeddings=2048...)". opt-125m is 2048, so a
    # fixed default stopped it from starting at all.
    model_max = None
    try:
        model_max = getattr(AutoConfig.from_pretrained(model), "max_position_embeddings", None)
    except Exception:
        model_max = None
    max_model_len = args.max_model_len
    if model_max is not None and max_model_len > int(model_max):
        print(f"[server] clamping --max-model-len {max_model_len} -> {int(model_max)}")
        max_model_len = int(model_max)

    report: dict[str, Any] = {"phase": "6", "model": model, "ok": False}
    report["flags"] = {
        "model": model,
        "packed_buffer_bytes": packed_bytes,
        "packed_num_buffers": packed_buffers,
        "transfer_budget_seconds": args.transfer_budget,
        "drain_interval_seconds": DRAIN_INTERVAL_S,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len_requested": args.max_model_len,
        "max_model_len_effective": max_model_len,
        "model_max_position_embeddings": model_max,
    }
    sha, dirty = git_revision()
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")

    current: subprocess.Popen[str] | None = None
    hung = False

    def lifetime(log_name: str, *, dummy: bool) -> subprocess.Popen[str]:
        nonlocal current
        if current is not None:
            _terminate(current)
        current = start_server(
            model,
            log_name,
            dummy=dummy,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        return current

    try:
        server = lifetime("l1-dummy", dummy=True)
        report["vllm_version"] = vllm_version()
        print(f"[server] vllm {report['vllm_version']}")

        print(f"[trainer] loading {model} on cuda:{TRAINER_DEVICE_INDEX}")
        torch.cuda.set_device(TRAINER_DEVICE_INDEX)
        train_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
        train_model.to(f"cuda:{TRAINER_DEVICE_INDEX}")

        driver, _adapter, ctrl, runner = build_engine(
            train_model,
            model,
            packed_buffer_bytes=packed_bytes,
            packed_num_buffers=packed_buffers,
        )
        identity = driver.identity()
        print("[cycle] bootstrap")
        runner.bootstrap()
        before = generate(model)
        print(f"[rc-0] {before[:60]!r}")

        # ============================ L1: the hot path
        print("\n--- L1: hot update, decomposed ---")
        timeline = StateTimeline(ctrl)
        cycle_outcome: dict[str, Any] = {}

        def do_hot_cycle() -> None:
            try:
                cycle_outcome["result"] = runner.run_cycle(ctrl.next_target(identity))
            except BaseException as exc:
                cycle_outcome["error"] = f"{type(exc).__name__}: {exc}"

        timeline.start()
        hot_started = time.monotonic()
        # On a daemon thread with a budget. A divergent packed stream deadlocks the
        # NCCL collective instead of erroring, and the broadcast runs concurrently
        # with the HTTP calls -- so the client's 300s timeout (clients.py:66) is on
        # the requests, not on the collective, and nothing else bounds it.
        cycle_thread = threading.Thread(target=do_hot_cycle, name="hot-cycle", daemon=True)
        cycle_thread.start()
        cycle_thread.join(timeout=args.transfer_budget)
        if cycle_thread.is_alive():
            hung = True
            timeline.stop()
            detail["hot_hang"] = {
                "seconds_waited": round(time.monotonic() - hot_started, 4),
                "budget_seconds": args.transfer_budget,
                "states": timeline.durations(),
                "engine_label": _safe_weight_info(),
                "verdict": (
                    "the update never returned: a packed-stream divergence deadlocks "
                    "the collective rather than raising. Nothing published, engine "
                    "left paused."
                ),
            }
            print(f"[hot] HUNG for {args.transfer_budget}s; states: {detail['hot_hang']['states']}")
            record(checks, "hot_path_did_not_hang", False)
            # os._exit skips the finally block, so the engine has to be stopped
            # here or it keeps port 8000 and the next lifetime cannot bind.
            if current is not None:
                _terminate(current)
            _write(report, detail, checks)
            print("[exit] os._exit to bound the cost of a blocked collective")
            sys.stdout.flush()
            os._exit(3)
        hot_seconds = round(time.monotonic() - hot_started, 4)
        events = timeline.stop()
        if cycle_outcome.get("error"):
            raise RuntimeError(f"hot update failed: {cycle_outcome['error']}")
        result = cycle_outcome["result"]
        after = generate(model)
        states = timeline.durations()
        update_seconds = float(states.get(LifecycleState.UPDATING.value, 0.0))
        detail["hot"] = {
            "seconds": hot_seconds,
            "timeline_seconds": timeline.total(),
            "states": states,
            "events": [(state, round(at - hot_started, 4)) for state, at in events],
            "before": before,
            "after": after,
            "committed": ctrl.current_version.label if ctrl.current_version else None,
            "engine_label": weight_info(),
            "update_phase_seconds": update_seconds,
        }
        print(f"[hot] {hot_seconds}s total; per state: {states}")
        record(checks, "hot_path_committed_rc1", ctrl.current_version.label == "rc-1")
        record(
            checks,
            "hot_path_changed_the_output",
            after != before and not only_bos(after),
        )
        record(checks, "timeline_saw_the_update", update_seconds > 0)
        print(f"[hot] update phase measured at {update_seconds}s (aims the deep kill)")

        # ============================ L2: the restart baseline
        print("\n--- L2: restart baseline (checkpoint assumed already on disk) ---")
        _terminate(server)
        restart_started = time.monotonic()
        server = lifetime("l2-restart", dummy=False)
        ready_seconds = round(time.monotonic() - restart_started, 4)
        first_token = generate(model)
        restart_seconds = round(time.monotonic() - restart_started, 4)
        detail["restart"] = {
            "kill_to_ready_seconds": ready_seconds,
            "kill_to_first_token_seconds": restart_seconds,
            "text": first_token,
            "note": (
                "generous to the baseline: the checkpoint is already on disk, so the "
                "checkpoint write a real restart-based update needs is not counted"
            ),
        }
        print(f"[restart] ready in {ready_seconds}s, serving by {restart_seconds}s")
        record(checks, "restart_served", bool(first_token))
        record(checks, "restart_cost_is_measurable", restart_seconds >= ready_seconds > 0)
        record(
            checks,
            "hot_path_is_faster_than_a_restart",
            hot_seconds < ready_seconds,
        )

        # ============================ L3: kill inside the collective
        if not args.skip_deep_kill:
            print("\n--- L3: SIGKILL inside the collective ---")
            server = lifetime("l3-deep-kill", dummy=True)
            _d3, _a3, ctrl3, runner3 = build_engine(
                train_model,
                model,
                packed_buffer_bytes=packed_bytes,
                packed_num_buffers=packed_buffers,
            )
            runner3.bootstrap()
            identity3 = _d3.identity()
            outcome: dict[str, Any] = {}

            def cycle() -> None:
                began = time.monotonic()
                try:
                    runner3.run_cycle(ctrl3.next_target(identity3))
                    outcome["returned"] = "committed"
                except BaseException as exc:
                    outcome["error"] = f"{type(exc).__name__}: {exc}"
                outcome["seconds"] = round(time.monotonic() - began, 4)

            worker = threading.Thread(target=cycle, name="cycle", daemon=True)
            worker.start()
            armed = time.monotonic()
            state_at_kill = LifecycleState.READY
            while time.monotonic() - armed < ARM_TIMEOUT_S:
                state_at_kill = ctrl3.state
                if state_at_kill is LifecycleState.UPDATING:
                    break
                time.sleep(0.01)
            reached_updating = state_at_kill is LifecycleState.UPDATING
            # Aim at half the measured update, so the kill lands in the collective
            # rather than before it.
            time.sleep(max(update_seconds * 0.5, 0.0))
            # Re-read at the moment of the kill: if the update finished while we
            # slept, the kill did not land inside anything and the scenario is not
            # the one it claims to be.
            state_at_kill = ctrl3.state
            killed_at = time.monotonic()
            _kill_engine_hard(server)
            print(
                f"[deep-kill] state={state_at_kill.value}, killed "
                f"{killed_at - armed:.2f}s in (aim was {update_seconds * 0.5:.2f}s)"
            )

            worker.join(timeout=DETECTION_BUDGET_S)
            hung = worker.is_alive()
            detail["deep_kill"] = {
                "state_at_kill": state_at_kill.value,
                "reached_updating": reached_updating,
                "aimed_at_seconds": round(update_seconds * 0.5, 4),
                "killed_after_seconds": round(killed_at - armed, 4),
                "hung": hung,
                "seconds_to_outcome": round(time.monotonic() - killed_at, 4),
                "outcome": outcome.get("error") or outcome.get("returned"),
                "budget_seconds": DETECTION_BUDGET_S,
            }
            print(
                f"[deep-kill] {'STILL BLOCKED' if hung else 'returned'}: "
                f"{detail['deep_kill']['outcome']}"
            )
            # That the kill landed *inside* the update is the whole premise, and it
            # is not guaranteed: a slow host, or an aim that overshoots, would put
            # it after the transfer and quietly test nothing.
            record(
                checks,
                "deep_kill_landed_during_the_update",
                state_at_kill is LifecycleState.UPDATING,
            )
            record(
                checks,
                "deep_kill_no_version_published",
                ctrl3.current_version.label == "rc-0",
            )
            record(checks, "deep_kill_engine_is_dead", server.poll() is not None)
            record(
                checks,
                "deep_kill_outcome_is_recorded",
                bool(outcome.get("error")) or hung,
            )
            record(checks, "deep_kill_detected_within_budget", not hung)
            detail["deep_kill_controller"] = {
                "state": ctrl3.state.value,
                "tainted": ctrl3.is_tainted,
                "taint_reason": ctrl3.taint_reason,
            }

        report["states_visited"] = [s.value for s in result.states_visited]
        report["detail"] = detail
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 6 ===")
        print(f"  hot update    : {hot_seconds}s  (update phase {update_seconds}s)")
        print(f"  restart       : {ready_seconds}s to ready, {restart_seconds}s to serving")
        if detail.get("deep_kill"):
            how = "HANG" if hung else f"detected in {detail['deep_kill']['seconds_to_outcome']}s"
            print(f"  deep kill     : {how}")
        detail["comparison"] = {
            "hot_seconds": hot_seconds,
            "restart_kill_to_ready_seconds": ready_seconds,
            "restart_kill_to_serving_seconds": restart_seconds,
            "speedup_vs_ready": round(ready_seconds / hot_seconds, 2) if hot_seconds else None,
            "speedup_vs_serving": (
                round(restart_seconds / hot_seconds, 2) if hot_seconds else None
            ),
        }
        print(f"  comparison    : {detail['comparison']}")
        print(
            f"  RESULT: {'PASS' if report['ok'] else 'FAIL'} "
            f"({sum(checks.values())}/{len(checks)} checks)"
        )
        if hung:
            _write(report, detail, checks)
            print("[exit] transfer thread still blocked; os._exit to bound the cost")
            sys.stdout.flush()
            os._exit(2)
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        return 1
    finally:
        _write(report, detail, checks)
        if current is not None:
            _terminate(current)


def _write(report: dict[str, Any], detail: dict[str, Any], checks: dict[str, bool]) -> None:
    report["detail"] = detail
    report["checks"] = checks
    out = _REPO / "results" / "phase6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[report] {out}")


if __name__ == "__main__":
    sys.exit(main())
