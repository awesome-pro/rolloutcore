#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4A: one version per rollout, while an update is requested underneath it.

Phase 3C proved the cycle works. This proves the guarantee the cycle exists for,
and it is the invariant with **no engine-side support** in vLLM `main`: PR #49040
removed per-request version binding on purpose, so nothing stops a response from
spanning two weight versions except the component that refuses to let it happen.

    READY(rc-0) -> admit R-long -> start a 256-token generation
                -> run_cycle(rc-1)          <- DRAINING waits for R-long
                -> R-long completes, released from its own thread
                -> QUIESCED -> UPDATING (the mutation) -> ... -> READY(rc-1)
                -> admit R-2 -> must be rc-1

The claim under test is about **delay, not survival**: the update is requested
while a generation is live, and RolloutCore holds the mutation off until that
generation is gone, so no rollout ever runs across a weight change. The evidence
is the text. The server runs `--load-format dummy`, whose output is degenerate,
and the update installs `facebook/opt-125m`'s real weights, whose output for this
prompt is coherent and was measured in Phase 3A. The two are unmistakable, so a
256-token generation whose 16-token prefix matches the pre-update baseline is a
rollout that ran on rc-0 throughout.

(An earlier version of this docstring said the generation "finishes after the
update". It does not: the journal shows ``finish_rollout`` before ``begin_update``.
The rollout spans the *drain*, which is the point -- the corrected reading is in
`docs/phase4a-results.md`.)

Two subtleties this harness exists to expose, both found while writing it:

* ``/pause?mode=wait`` blocks until the engine is idle, and ``confirm_drained``
  **taints** if the engine claims quiescence while RolloutCore still counts live
  work (``lifecycle.py:465-472``). So the rollout must be released from the thread
  that owns its request, the moment that request returns -- not after the cycle.
  The engine goes idle a few milliseconds before the client sees the response, so
  ``drain_interval`` is deliberately far larger than that gap: the release has to
  win the race against the next poll.
* The controller is therefore used from two threads at once, which is why it now
  holds a lock (``tests/test_controller_threading.py``).

Run on the two-GPU node, with vLLM installed and HF_HOME set:

    python3 diagnostics/rolloutcore_concurrency_2gpu.py
"""

from __future__ import annotations

import json
import os
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

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase4a-server.log"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
BASELINE_TOKENS = 16
#: `ignore_eos` makes this exact, so the request cannot finish early and quietly
#: stop overlapping the cycle.
LONG_TOKENS = 256
#: Time for the long request to actually be in flight before the cycle starts.
SETTLE_S = 0.4


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
        "dummy",  # the pre-update weights are deliberately degenerate
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
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[server] log: {SERVER_LOG}")
    try:
        _await_ready(proc)
    except BaseException:
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


def probe_ok(url: str) -> bool:
    """Readiness probe: status only.

    ``/health`` is ``Response(status_code=200)`` with an *empty body*
    (``serve/instrumentator/health.py:28``), so it must never be fed to
    ``json.loads``. See ``tests/test_script_hygiene.py``.
    """
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
    data = json.dumps(body).encode("utf-8")
    return urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def generate(max_tokens: int = BASELINE_TOKENS) -> dict[str, Any]:
    with urllib.request.urlopen(completion_request(max_tokens, ignore_eos=False), timeout=600) as r:
        payload = json.loads(r.read().decode("utf-8"))
    choice = payload["choices"][0]
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "weight_version": weight_info(),
    }


def weight_info() -> str:
    return str(get(f"{BASE_URL}/weight_info")["weight_version"])


def is_paused() -> bool:
    return bool(get(f"{BASE_URL}/is_paused")["is_paused"])


def world_size() -> int:
    return int(get(f"{BASE_URL}/get_world_size")["world_size"])


def git_revision() -> tuple[str, bool]:
    """``(HEAD, dirty)`` for the code under test. Never fatal.

    Scoped to the code directories on purpose. The pod's tree legitimately lacks
    the tracked data files under ``results/`` (the copy excludes them), and git
    reports a missing tracked file as a modification -- which would mark every
    artifact dirty and make the flag worthless.
    """

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


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "4A", "model": MODEL_NAME, "ok": False}
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    sha, dirty = git_revision()
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")
    ctrl: LifecycleController | None = None
    server = start_vllm_server()
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
        print(f"[identity] {identity.describe()}")

        adapter = HttpVLLMAdapter(
            base_url=BASE_URL,
            weight_identity=identity,
            driver=driver,
            drain_timeout=600,
        )
        ctrl = LifecycleController()
        # 600 x 2s: the release race below is comfortably inside one poll.
        runner = LifecycleRunner(ctrl, adapter, drain_polls=600, drain_interval=2.0)

        print("[cycle] bootstrap")
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        assert weight_info() == "rc-0", weight_info()

        baseline = generate(BASELINE_TOKENS)
        print(f"[rc-0] baseline {baseline['text']!r}")
        report["baseline"] = baseline

        # ---- the rollout that is in flight when the update is requested --
        binding = ctrl.admit_rollout("R-long")
        report["pre_update_binding"] = {
            "request_id": binding.request_id,
            "version": binding.version.label,
            "cache_salt": binding.cache_salt,
        }
        print(f"[admit] R-long bound to {binding.version.label}")

        outcome: dict[str, Any] = {}

        def long_rollout() -> None:
            """Own the request, and release the rollout the instant it returns.

            The release happens before the body is parsed on purpose: the engine
            reports itself drained within milliseconds of finishing this request,
            and `confirm_drained` taints if RolloutCore still counts it.
            """
            try:
                outcome["request_started_at"] = time.monotonic()
                req = completion_request(LONG_TOKENS, ignore_eos=True)
                with urllib.request.urlopen(req, timeout=600) as resp:
                    outcome["request_ended_at"] = time.monotonic()
                    released = ctrl.finish_rollout("R-long")
                    outcome["released_version"] = released.version.label
                    payload = json.loads(resp.read().decode("utf-8"))
                choice = payload["choices"][0]
                outcome["text"] = choice.get("text", "")
                outcome["finish_reason"] = choice.get("finish_reason")
            except BaseException as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
                outcome["request_ended_at"] = time.monotonic()

        worker = threading.Thread(target=long_rollout, name="R-long")
        worker.start()
        time.sleep(SETTLE_S)  # let it be genuinely in flight first

        target = ctrl.next_target(identity)
        print(f"[cycle] run_cycle -> {target.describe()} while R-long is generating")
        cycle_started_at = time.monotonic()
        result = runner.run_cycle(target)
        cycle_ended_at = time.monotonic()
        worker.join(timeout=120)
        if worker.is_alive():
            raise RuntimeError("the in-flight rollout never returned")

        after = generate(BASELINE_TOKENS)
        print(f"[rc-1] after    {after['text']!r}")

        # A second rollout, admitted after the update, must bind to rc-1.
        second = ctrl.admit_rollout("R-2")
        ctrl.finish_rollout("R-2")

        inflight = outcome.get("text", "")
        req_started = outcome.get("request_started_at")
        req_ended = outcome.get("request_ended_at")
        overlapped = (
            isinstance(req_started, float)
            and isinstance(req_ended, float)
            and cycle_started_at < req_ended
        )
        remaining = (req_ended - cycle_started_at) if overlapped else 0.0
        # Recorded so a prefix mismatch is diagnosable rather than just red.
        shared = 0
        for a, b in zip(inflight, baseline["text"], strict=False):
            if a != b:
                break
            shared += 1

        report.update(
            {
                "world_size": group_size,
                "identity": identity.describe(),
                "inflight": {
                    "max_tokens": LONG_TOKENS,
                    "text": inflight,
                    "finish_reason": outcome.get("finish_reason"),
                    "error": outcome.get("error"),
                    "released_version": outcome.get("released_version"),
                    "request_seconds": (
                        round(req_ended - req_started, 4)
                        if isinstance(req_started, float) and isinstance(req_ended, float)
                        else None
                    ),
                    "seconds_overlapping_cycle": round(remaining, 4),
                    "chars_matching_baseline_prefix": shared,
                    "baseline_prefix_chars": len(baseline["text"]),
                },
                "cycle_seconds": round(cycle_ended_at - cycle_started_at, 4),
                "drain_polls": result.drain_polls,
                "states_visited": [s.value for s in result.states_visited],
                "after": after,
                "post_update_binding": second.version.label,
                "engine_label_after": weight_info(),
                "engine_paused_after": is_paused(),
                "server_pid": server.pid,
                "server_alive": server.poll() is None,
                "journal": [r.event for r in ctrl.journal],
            }
        )

        checks = {
            "pre_update_binding_is_rc0": binding.version.label == "rc-0",
            "rollout_released_cleanly": "error" not in outcome,
            "inflight_overlapped_the_cycle": overlapped,
            "inflight_ran_to_length": report["inflight"]["finish_reason"] == "length",
            "inflight_used_rc0_weights": bool(inflight) and inflight.startswith(baseline["text"]),
            "inflight_is_not_rc1": bool(inflight) and not inflight.startswith(after["text"][:8]),
            "drain_outlasted_the_request": (
                report["cycle_seconds"] >= 0.5 * remaining and remaining > 0
            ),
            "committed_rc1": ctrl.current_version.label == "rc-1",
            "post_update_binding_is_rc1": second.version.label == "rc-1",
            "engine_label_rc1": report["engine_label_after"] == "rc-1",
            "engine_resumed": report["engine_paused_after"] is False,
            "server_never_restarted": report["server_alive"],
        }
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 4A ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  rc-0 baseline : {baseline['text']!r}")
        print(f"  in-flight     : {inflight[:70]!r}{'...' if len(inflight) > 70 else ''}")
        print(f"  rc-1 after    : {after['text']!r}")
        print(
            f"  timing        : request {report['inflight']['request_seconds']}s, "
            f"{report['inflight']['seconds_overlapping_cycle']}s of it under the cycle, "
            f"cycle {report['cycle_seconds']}s over {result.drain_polls} poll(s)"
        )
        print(f"  states: {' -> '.join(report['states_visited'])}")
        print(f"  RESULT: {'PASS' if report['ok'] else 'FAIL'}")
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        tail = server_log_tail()
        if tail:
            print(f"[server] last lines of {SERVER_LOG}:\n{tail}")
        if ctrl is not None:
            print(f"[controller] state={ctrl.state} taint={ctrl.taint_reason}")
        return 1
    finally:
        out = _REPO / "results" / "phase4a.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")
        _terminate(server)


if __name__ == "__main__":
    sys.exit(main())
