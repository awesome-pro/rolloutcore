#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4B: the failure paths, against a real engine.

Every fail-closed claim in this project has so far been tested against the fake
engine, which *models* the failures we thought of. This runs them against real
vLLM: a real socket that dies, a real ``/pause`` that times out, a real trainer
holding different weights than the target declares.

Four scenarios, three engine lifetimes:

**A. Drain failure is recoverable and does not taint.** ``/pause?mode=wait`` is
given a 0.1 s socket timeout while a 256-token generation is in flight, so the
attempt fails. ``DrainFailedError`` is not a taint: the engine is healthy, we
simply failed to drain it, so the controller stays in DRAINING and a fresh
attempt finishes the job *without restarting anything*. The script then drives
the rest of the cycle by hand and lands in READY(rc-1) -- the recovery is the
test.

``drain_reissues=0`` on purpose. With a reissue enabled the second ``/pause`` can
succeed, because the FAILED path aborts stragglers *before* reissuing and the
engine is then already idle -- good behaviour, but it would not exercise the
failure. With no reissues the first timeout surfaces deterministically.

**B. An identity mismatch fails before any mutation.** The target declares the
manifest of a *different* module than the trainer holds, so the driver's
precheck raises ``WeightIdentityMismatchError`` before ``send_weights()`` runs.
The strongest evidence is the server log: ``/start_weight_update`` appears *zero*
times for that attempt. The engine is left paused, still at rc-1, because nothing
was written.

This is also where the harness reports a **gap** rather than a pass:
``(UPDATING, CONFIRM_UPDATED)`` is the only transition out of UPDATING
(``lifecycle.py:132``), so a driver-side failure mid-update leaves the engine
paused with no controller-level recovery. A fresh controller cannot adopt it
either, because the label is still managed -- checked, not assumed.

**C. A dead engine taints, and the taint is terminal.** The server's whole
process group is SIGKILLed, then a cycle is attempted: the mutating ``/pause``
gets a real ``ConnectionRefusedError``, the runner cannot know what the engine
did, so it taints. The next cycle must be refused locally, with no network I/O.

**D. Recovery is a fresh process.** A third engine, a fourth controller:
bootstrap succeeds. Taint is controller-scoped, and the old engine's state died
with the process.

Run on the two-GPU node, with vLLM installed and HF_HOME set:

    python3 diagnostics/rolloutcore_failures_2gpu.py
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
)
from rolloutcore.adapters import HttpVLLMAdapter  # noqa: E402
from rolloutcore.adapters.nccl import NCCLWeightTransferDriver  # noqa: E402
from rolloutcore.errors import (  # noqa: E402
    AlreadyManagedEngineError,
    DrainFailedError,
    EvidenceNotReady,
    WeightIdentityMismatchError,
)

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
BASELINE_TOKENS = 16
LONG_TOKENS = 256
SETTLE_S = 0.3

#: One log per engine lifetime, so scenario B can assert on *this* engine's log.
LOG_1 = _REPO / "results" / "phase4b-server-1.log"
LOG_2 = _REPO / "results" / "phase4b-server-2.log"
LOG_3 = _REPO / "results" / "phase4b-server-3.log"


# ------------------------------------------------------------------- server


def start_vllm_server(log_path: Path) -> subprocess.Popen[str]:
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
        "dummy",
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        '{"backend": "nccl"}',
    ]
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[server] {' '.join(args)}  (log: {log_path.name})")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(args) + "\n")
        log.write("# VLLM_SERVER_DEV_MODE=1 VLLM_ENABLE_V1_MULTIPROCESSING=1\n")
        log.flush()
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        _await_ready(proc)
    except BaseException:
        _terminate(proc)
        raise
    return proc


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Stop the server's whole process group: SIGTERM, then SIGKILL. Idempotent."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover - already gone
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - already gone
            return
        proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str]) -> None:
    """Block until ``/health`` answers, or explain why it never did."""
    started = time.monotonic()
    deadline = started + 900
    last_beat = started
    while True:
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited before becoming ready")
        if probe_ok(f"{BASE_URL}/health"):
            print(f"[server] ready after {time.monotonic() - started:.1f}s (pid {proc.pid})")
            return
        now = time.monotonic()
        if now - last_beat >= 15:
            print(f"[server] still loading ({now - started:.0f}s)")
            last_beat = now
        if now > deadline:
            raise RuntimeError("vLLM did not become ready in time")
        time.sleep(2)


# ------------------------------------------------------------------ http


def probe_ok(url: str) -> bool:
    """Status-only readiness probe: ``/health`` has an empty body
    (``serve/instrumentator/health.py:28``). See ``tests/test_script_hygiene.py``."""
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


def completion_request(max_tokens: int, *, ignore_eos: bool) -> urllib.request.Request:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    return urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def generate(max_tokens: int = BASELINE_TOKENS) -> str:
    with urllib.request.urlopen(completion_request(max_tokens, ignore_eos=False), timeout=600) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return str(payload["choices"][0].get("text", ""))


def weight_info() -> str:
    return str(get(f"{BASE_URL}/weight_info")["weight_version"])


def is_paused() -> bool:
    return bool(get(f"{BASE_URL}/is_paused")["is_paused"])


def world_size() -> int:
    return int(get(f"{BASE_URL}/get_world_size")["world_size"])


def vllm_version() -> str:
    try:
        return str(get(f"{BASE_URL}/version")["version"])
    except Exception:
        return "unknown"


def log_count(path: Path, needle: str) -> int:
    """Occurrences of ``needle`` in a server log.

    Anchored on the request line, not the bare route name: `vllm serve` prints its
    whole route table at startup, so counting `/start_weight_update` would start
    at 1 for a request that was never made.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace").count(needle)
    except OSError:
        return 0


def git_revision() -> tuple[str, bool]:
    """``(HEAD, dirty)`` for the code under test. Scoped, so the pod's tree --
    which lacks the tracked data files under ``results/`` -- is not 'dirty'."""

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


# ------------------------------------------------------------------ engine


def make_driver(train_model: Any) -> NCCLWeightTransferDriver:
    group_size = world_size() + 1  # every inference worker, plus this trainer
    return NCCLWeightTransferDriver(
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


def make_adapter(
    driver: NCCLWeightTransferDriver, *, drain_timeout: float = 600.0, drain_reissues: int = 2
) -> HttpVLLMAdapter:
    return HttpVLLMAdapter(
        base_url=BASE_URL,
        weight_identity=driver.identity(),
        driver=driver,
        drain_timeout=drain_timeout,
        drain_reissues=drain_reissues,
    )


def drain_until_quiesced(ctrl: LifecycleController, adapter: HttpVLLMAdapter) -> int:
    """Start a pause attempt and poll until the controller accepts the drain."""
    adapter.begin_drain()
    for poll in range(1, 601):
        try:
            ctrl.confirm_drained(adapter.await_drain())
            return poll
        except EvidenceNotReady:
            time.sleep(0.2)
    raise RuntimeError("the drain never completed")


def drive_to_ready(ctrl: LifecycleController, adapter: HttpVLLMAdapter, target: Any) -> list[str]:
    """Finish a cycle from QUIESCED by hand, recording the states reached.

    ``run_cycle`` cannot be re-entered from QUIESCED (it starts at READY), and the
    point of scenario A is that a failed drain does not force a restart.
    """
    visited = [ctrl.state.value]
    ctrl.begin_update(target)
    adapter.start_weight_update(target)
    ctrl.confirm_updated(adapter.complete_weight_update(target))
    visited.append(ctrl.state.value)
    ctrl.confirm_invalidated(adapter.invalidate_caches(target))
    visited.append(ctrl.state.value)
    ctrl.confirm_validated(adapter.validate_pre_resume(target))
    visited.append(ctrl.state.value)
    ctrl.confirm_resumed(adapter.resume(target))
    visited.append(ctrl.state.value)
    return visited


def tail(path: Path, lines: int = 10) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "4B", "model": MODEL_NAME, "ok": False}
    sha, dirty = git_revision()
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    servers: list[subprocess.Popen[str]] = []
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")

    server = start_vllm_server(LOG_1)
    servers.append(server)
    try:
        report["vllm_version"] = vllm_version()
        print(f"[server] vllm {report['vllm_version']}")
        print(f"[trainer] loading {MODEL_NAME} on cuda:{TRAINER_DEVICE_INDEX}")
        torch.cuda.set_device(TRAINER_DEVICE_INDEX)
        train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        train_model.to(f"cuda:{TRAINER_DEVICE_INDEX}")

        # ================================================== A. drain failure
        print("\n--- A: a failed drain is recoverable ---")
        driver, adapter_short, ctrl, runner = _engine(
            train_model, drain_timeout=0.1, drain_reissues=0
        )
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        identity = driver.identity()
        ctrl.admit_rollout("R-drain")

        inflight: dict[str, Any] = {}

        def long_request() -> None:
            try:
                req = completion_request(LONG_TOKENS, ignore_eos=True)
                with urllib.request.urlopen(req, timeout=600) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                inflight["finish_reason"] = payload["choices"][0].get("finish_reason")
            except BaseException as exc:
                inflight["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=long_request)
        worker.start()
        time.sleep(SETTLE_S)

        aborts_before = log_count(LOG_1, "/abort_requests HTTP/1.1")
        ctrl.begin_drain()
        adapter_short.begin_drain()
        drain_error: str | None = None
        for _ in range(60):
            try:
                adapter_short.await_drain()
            except EvidenceNotReady:
                time.sleep(0.1)
                continue
            except DrainFailedError as exc:
                drain_error = str(exc)
                break
            time.sleep(0.1)

        worker.join(timeout=60)
        if worker.is_alive():
            raise RuntimeError("the in-flight request never returned")
        ctrl.finish_rollout("R-drain")

        detail["drain_failure"] = {
            "error": drain_error,
            "state": ctrl.state.value,
            "tainted": ctrl.is_tainted,
            "abort_requests_posted": log_count(LOG_1, "/abort_requests HTTP/1.1") - aborts_before,
            "inflight_outcome": (
                inflight.get("error") or inflight.get("finish_reason") or "unknown"
            ),
        }
        checks["A_drain_failed_with_DrainFailedError"] = drain_error is not None
        checks["A_drain_failure_does_not_taint"] = not ctrl.is_tainted
        checks["A_state_stays_DRAINING"] = ctrl.state is LifecycleState.DRAINING
        checks["A_stragglers_were_aborted"] = detail["drain_failure"]["abort_requests_posted"] >= 1

        # Recovery, without restarting anything. Same driver (its NCCL session with
        # the engine is still live); a fresh adapter with a real timeout.
        adapter_ok = make_adapter(driver, drain_timeout=600.0)
        polls = drain_until_quiesced(ctrl, adapter_ok)
        target = ctrl.next_target(identity)
        states = drive_to_ready(ctrl, adapter_ok, target)
        after_text = generate()
        detail["recovery"] = {
            "drain_polls": polls,
            "states": states,
            "committed": ctrl.current_version.label if ctrl.current_version else None,
            "engine_label": weight_info(),
            "tainted": ctrl.is_tainted,
            "text": after_text,
        }
        checks["A_recovery_reaches_READY"] = ctrl.state is LifecycleState.READY
        checks["A_recovery_installs_the_update"] = ctrl.current_version.label == "rc-1"
        checks["A_recovery_serves"] = after_text.startswith(" the capital")

        # ========================================== B. identity mismatch
        print("\n--- B: an identity mismatch fails before mutating ---")
        # The trainer holds opt-125m. This driver holds a different manifest, which
        # is what a target would declare if the trainer were serving other weights.
        other_driver = NCCLWeightTransferDriver(
            base_url=BASE_URL,
            trainer_init_info=NCCLTrainerInitInfo(
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=world_size() + 1,
                rank=0,
                packed=True,
            ),
            source=ModuleSource(torch.nn.Linear(4, 8)),
            world_size=world_size() + 1,
        )
        other_identity = other_driver.identity()  # offline: no rendezvous, no request
        print(f"[B] trainer holds {identity.describe()}")
        print(f"[B] target claims {other_identity.describe()}")

        drain_until_quiesced(ctrl, adapter_ok)  # READY -> QUIESCED
        mismatched = ctrl.next_target(other_identity)
        ctrl.begin_update(mismatched)
        adapter_ok.start_weight_update(mismatched)
        starts_before = log_count(LOG_1, "/start_weight_update HTTP/1.1")
        mismatch_error: str | None = None
        try:
            adapter_ok.complete_weight_update(mismatched)
        except WeightIdentityMismatchError as exc:
            mismatch_error = str(exc)
        starts_after = log_count(LOG_1, "/start_weight_update HTTP/1.1")

        # A fresh controller cannot adopt the paused engine either: the label is
        # still managed, so this is checked rather than assumed.
        handback: str | None = None
        try:
            _d, adapter_hb, _c, _r = _engine(train_model)
            adapter_hb.bootstrap()
            handback = "adopted (unexpected)"
        except AlreadyManagedEngineError as exc:
            handback = f"AlreadyManagedEngineError: {exc}"

        detail["identity_mismatch"] = {
            "error": mismatch_error,
            "state": ctrl.state.value,
            "tainted": ctrl.is_tainted,
            "engine_paused": is_paused(),
            "engine_label": weight_info(),
            "start_weight_update_posts_this_attempt": starts_after - starts_before,
            "fresh_controller_bootstrap": handback,
        }
        checks["B_mismatch_raises_WeightIdentityMismatchError"] = mismatch_error is not None
        checks["B_nothing_was_written"] = starts_after - starts_before == 0
        checks["B_engine_still_paused"] = detail["identity_mismatch"]["engine_paused"] is True
        checks["B_engine_label_unchanged"] = weight_info() == "rc-1"
        checks["B_no_taint"] = not ctrl.is_tainted

        # The engine itself is fine: raw /resume puts it back in service.
        post(f"{BASE_URL}/resume", timeout=60)
        detail["identity_mismatch"]["resumed_by_hand"] = not is_paused()

        # ============================================ C. a dead engine
        print("\n--- C: a dead engine ---")
        _terminate(server)
        server_c = start_vllm_server(LOG_2)
        servers.append(server_c)
        driver4, _adapter4, ctrl4, runner4 = _engine(train_model)
        runner4.bootstrap()
        assert ctrl4.state is LifecycleState.READY, ctrl4.state
        dead_target = ctrl4.next_target(driver4.identity())
        print(f"[C] SIGKILL the engine's process group (pid {server_c.pid})")
        os.killpg(os.getpgid(server_c.pid), signal.SIGKILL)
        server_c.wait(timeout=30)

        # C1 -- the drain path. The failure surfaces through `await_drain`, which
        # the runner calls with `_observe`, and `_observe` deliberately does not
        # taint a failed read: the engine is paused during DRAINING, so it cannot
        # serve anything new, and a read that failed left nothing ambiguous. That
        # reasoning still holds for a *dead* engine. What it costs is recorded
        # rather than asserted, because it is the interesting part.
        drain_path_error: str | None = None
        try:
            runner4.run_cycle(dead_target)
        except Exception as exc:
            drain_path_error = f"{type(exc).__name__}: {exc}"

        # C2 -- the mutating path, and the designed taint. A fresh controller
        # cannot even bootstrap: its first read fails with an unknown engine
        # outcome, which `bootstrap` turns into a taint.
        taint_error: str | None = None
        _d6, _a6, ctrl6, runner6 = _engine(train_model)
        try:
            runner6.bootstrap()
        except Exception as exc:
            taint_error = f"{type(exc).__name__}: {exc}"

        # C3 -- a tainted controller must refuse locally: legality is checked
        # before any precondition, so this does no network I/O at all.
        refused: str | None = None
        began = time.monotonic()
        try:
            runner6.run_cycle(dead_target)
        except Exception as exc:
            refused = f"{type(exc).__name__}: {exc}"
        refusal_seconds = round(time.monotonic() - began, 4)

        detail["engine_death"] = {
            "drain_path_error": drain_path_error,
            "drain_path_state": ctrl4.state.value,
            "drain_path_tainted": ctrl4.is_tainted,
            "drain_path_committed": (
                ctrl4.current_version.label if ctrl4.current_version else None
            ),
            "bootstrap_error": taint_error,
            "tainted": ctrl6.is_tainted,
            "taint_reason": ctrl6.taint_reason,
            "second_cycle": refused,
            "second_cycle_seconds": refusal_seconds,
        }
        checks["C_drain_path_surfaces_a_failure"] = drain_path_error is not None
        checks["C_nothing_was_committed"] = detail["engine_death"]["drain_path_committed"] == "rc-0"
        checks["C_bootstrap_on_a_dead_engine_taints"] = ctrl6.is_tainted
        checks["C_taint_has_a_reason"] = bool(ctrl6.taint_reason)
        checks["C_taint_is_terminal"] = bool(refused and "IllegalTransition" in refused)
        checks["C_refusal_is_local"] = refusal_seconds < 0.05

        # ===================================== D. recovery is a fresh process
        print("\n--- D: recovery is a fresh engine and a fresh controller ---")
        server = start_vllm_server(LOG_3)
        servers.append(server)
        _d5, _adapter5, ctrl5, runner5 = _engine(train_model)
        runner5.bootstrap()
        recovered_text = generate()
        detail["fresh_recovery"] = {
            "state": ctrl5.state.value,
            "engine_label": weight_info(),
            "text": recovered_text,
            "prev_controller_still_tainted": ctrl6.is_tainted,
        }
        checks["D_fresh_controller_bootstraps"] = ctrl5.state is LifecycleState.READY
        checks["D_fresh_engine_is_rc0"] = weight_info() == "rc-0"
        checks["D_it_serves"] = bool(recovered_text)

        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 4B ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  A drain error  : {detail['drain_failure']['error']}")
        print(f"  A in-flight    : {detail['drain_failure']['inflight_outcome']}")
        print(f"  B mismatch     : {detail['identity_mismatch']['error']}")
        print(f"  B handback     : {detail['identity_mismatch']['fresh_controller_bootstrap']}")
        print(f"  C drain path   : {detail['engine_death']['drain_path_error']}")
        print(
            f"  C drain result : state={detail['engine_death']['drain_path_state']} "
            f"tainted={detail['engine_death']['drain_path_tainted']} "
            f"committed={detail['engine_death']['drain_path_committed']}"
        )
        print(f"  C bootstrap    : {detail['engine_death']['bootstrap_error']}")
        print(f"  C taint reason : {detail['engine_death']['taint_reason']}")
        print(f"  C retry        : {detail['engine_death']['second_cycle']}")
        print(f"  D fresh label  : {detail['fresh_recovery']['engine_label']}")
        print(f"  RESULT: {'PASS' if report['ok'] else 'FAIL'}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        print(f"[server] last lines:\n{tail(LOG_1)}")
        return 1
    finally:
        report["detail"] = detail
        report["checks"] = checks
        out = _REPO / "results" / "phase4b.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")
        for proc in servers:
            _terminate(proc)


def _engine(
    train_model: Any, *, drain_timeout: float = 600.0, drain_reissues: int = 2
) -> tuple[NCCLWeightTransferDriver, HttpVLLMAdapter, LifecycleController, LifecycleRunner]:
    """A fresh driver/adapter/controller/runner against the running server."""
    driver = make_driver(train_model)
    adapter = make_adapter(driver, drain_timeout=drain_timeout, drain_reissues=drain_reissues)
    ctrl = LifecycleController()
    runner = LifecycleRunner(ctrl, adapter, drain_polls=600, drain_interval=2.0)
    return driver, adapter, ctrl, runner


if __name__ == "__main__":
    sys.exit(main())
