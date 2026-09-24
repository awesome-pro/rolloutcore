#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 5: trajectories that name the training step, not just the architecture.

Item 5, on real hardware. Two things had to exist first and now do: a driver that
can **re-declare** its provenance (`NCCLWeightTransferDriver.declare`, because a
cached identity cannot follow weights that changed), and a record type that binds
a rollout's output to that identity (`rolloutcore.trajectory`).

The experiment is Phase 4A's, with the provenance added, because 4A produced the
case that makes the whole design necessary: a rollout admitted at ``rc-0`` that
finishes *after* the update to ``rc-1``. The engine then reports ``rc-1`` for a
request whose tokens came entirely from ``rc-0``. A trajectory that recorded the
engine's last word would be wrong about which weights produced it; one that
records the binding is not.

So the run asserts the divergence on purpose:

* ``R-long`` is bound to step 0 and its tokens are the degenerate ``<s>`` output of
  dummy weights -- while the engine says ``rc-1`` by the time it finished;
* ``R2`` is admitted afterwards and is bound to step 1;
* both carry ``exactness == "declared-source"``, and both survive a JSONL round
  trip. **A manifest-only identity would identify neither**, which is what
  Phase 3C measured (the same digest for dummy and real weights) and Phase 4C
  confirmed (the same digest for a corrupted checkpoint).

The provenance is **declared**, which is the honest word for it: the harness
states "step 0" and "step 1" for two weight sets; nothing here verifies any byte.
What the identity buys is that two versions can no longer collide.

    python3 diagnostics/rolloutcore_trajectory_2gpu.py
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
    Trajectory,
    TrajectoryRecorder,
    WeightProvenance,
)
from rolloutcore.adapters import HttpVLLMAdapter  # noqa: E402
from rolloutcore.adapters.nccl import NCCLWeightTransferDriver  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase5-server.log"
TRAJECTORY_FILE = _REPO / "results" / "phase5-trajectories.jsonl"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

PROMPT = "The capital of France is"
BASELINE_TOKENS = 16
LONG_TOKENS = 256
SETTLE_S = 0.4

#: Declared, not verified. The server boots from `--load-format dummy`, so step 0
#: is named for what it actually is rather than dressed up as real weights.
RUN_ID = "phase5-declared"
STEP0 = WeightProvenance(checkpoint=f"{MODEL_NAME}@dummy-init", run_id=RUN_ID, step=0)
STEP1 = WeightProvenance(checkpoint=MODEL_NAME, run_id=RUN_ID, step=1)


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


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str]) -> None:
    """Status-only readiness probe: ``/health`` is 200 with an empty body."""
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


def completion_request(max_tokens: int, *, ignore_eos: bool) -> urllib.request.Request:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "return_token_ids": True,
    }
    if ignore_eos:
        body["ignore_eos"] = True
    return urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )


def generate(max_tokens: int = BASELINE_TOKENS) -> dict[str, Any]:
    started = time.monotonic()
    with urllib.request.urlopen(
        completion_request(max_tokens, ignore_eos=False), timeout=600
    ) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = payload["choices"][0]
    return {
        "text": choice.get("text", ""),
        "token_ids": tuple(choice.get("token_ids") or ()),
        "finish_reason": choice.get("finish_reason"),
        "seconds": round(time.monotonic() - started, 4),
        "weight_version": weight_info(),
    }


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


def only_bos(text: str) -> bool:
    return bool(text) and not text.replace("<s>", "").strip()


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "5", "model": MODEL_NAME, "ok": False}
    sha, dirty = git_revision()
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    recorder = TrajectoryRecorder()
    print(f"[repo] rolloutcore {sha[:12]}{' (dirty)' if dirty else ''}")

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
            provenance=STEP0,
            world_size=group_size,
        )
        identity_v0 = driver.identity()
        adapter = HttpVLLMAdapter(
            base_url=BASE_URL,
            weight_identity=identity_v0,
            driver=driver,
            drain_timeout=600,
        )
        ctrl = LifecycleController()
        # 2.0s between drain polls: the release below races the next poll, and this
        # is the margin that wins it (see Phase 4A).
        runner = LifecycleRunner(ctrl, adapter, drain_polls=600, drain_interval=2.0)

        detail["provenance"] = {
            "step0": STEP0.describe(),
            "step1": STEP1.describe(),
            "step0_identity": identity_v0.describe(),
            "note": (
                "declared by the harness: the checkpoint and step are the caller's "
                "claim, which is what 'declared' means. Nothing here verifies weight bytes."
            ),
        }
        record(checks, "v0_is_declared_source", identity_v0.exactness == "declared-source")
        print(f"[identity] step 0 -> {identity_v0.describe()}")

        print("[cycle] bootstrap")
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        assert weight_info() == "rc-0", weight_info()

        baseline = generate()
        print(f"[rc-0] baseline {baseline['text']!r}")

        # ---- the rollout that spans the update ---------------------------
        binding_long = ctrl.admit_rollout("R-long")
        admission_label = weight_info()
        detail["inflight_binding"] = {
            "version": binding_long.version.label,
            "exactness": binding_long.weight_identity.exactness,
            "engine_version_at_admission": admission_label,
        }
        print(
            f"[admit] R-long bound to {binding_long.version.label} (engine says {admission_label})"
        )

        outcome: dict[str, Any] = {}

        def long_rollout() -> None:
            """Own the request, and release the rollout the instant it returns.

            The release comes before the body is parsed on purpose: the engine
            reports itself drained within milliseconds of finishing, and
            `confirm_drained` taints if RolloutCore still counts it.
            """
            try:
                req = completion_request(LONG_TOKENS, ignore_eos=True)
                with urllib.request.urlopen(req, timeout=600) as resp:
                    released = ctrl.finish_rollout("R-long")
                    outcome["released_version"] = released.version.label
                    payload = json.loads(resp.read().decode("utf-8"))
                choice = payload["choices"][0]
                outcome["text"] = choice.get("text", "")
                outcome["token_ids"] = tuple(choice.get("token_ids") or ())
                outcome["finish_reason"] = choice.get("finish_reason")
            except BaseException as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=long_rollout, name="R-long")
        worker.start()
        time.sleep(SETTLE_S)

        # A training step happened: the trainer's weights are now step 1, and the
        # driver has to say so or `transfer` would refuse the target.
        identity_v1 = driver.declare(STEP1)
        record(checks, "v1_is_declared_source", identity_v1.exactness == "declared-source")
        record(checks, "steps_have_different_identities", identity_v1 != identity_v0)
        record(
            checks,
            "steps_share_the_manifest",
            identity_v1.manifest_digest == identity_v0.manifest_digest,
        )
        print(f"[identity] step 1 -> {identity_v1.describe()}")

        target = ctrl.next_target(identity_v1)
        print(f"[cycle] run_cycle -> {target.describe()} while R-long generates")
        result = runner.run_cycle(target)
        worker.join(timeout=120)
        if worker.is_alive():
            raise RuntimeError("the in-flight rollout never returned")

        after = generate()
        print(f"[rc-1] after {after['text']!r}")

        # ---- the records --------------------------------------------------
        # Assembled after the cycle so the engine label is deterministic rather
        # than whatever it happened to be mid-update.
        long_trajectory = recorder.record(
            Trajectory(
                binding=binding_long,
                prompt=PROMPT,
                text=str(outcome.get("text", "")),
                token_ids=tuple(outcome.get("token_ids") or ()),
                finish_reason=outcome.get("finish_reason"),
                engine_version_at_admission=admission_label,
                engine_version_at_completion=after["weight_version"],
            )
        )

        binding_r2 = ctrl.admit_rollout("R2")
        r2_admission = weight_info()
        r2 = generate()
        ctrl.finish_rollout("R2")
        r2_trajectory = recorder.record(
            Trajectory(
                binding=binding_r2,
                prompt=PROMPT,
                text=r2["text"],
                token_ids=r2["token_ids"],
                finish_reason=r2["finish_reason"],
                seconds=r2["seconds"],
                engine_version_at_admission=r2_admission,
                engine_version_at_completion=weight_info(),
            )
        )

        written = recorder.write_jsonl(TRAJECTORY_FILE)
        restored = TrajectoryRecorder.read_jsonl(written)
        detail["trajectories"] = [t.to_json() for t in recorder.trajectories]
        detail["summary"] = recorder.summary()
        print(f"[trajectories] {recorder.summary()}")
        for t in recorder.trajectories:
            print(f"  {t.describe()}")

        record(
            checks,
            "inflight_bound_to_step_0",
            long_trajectory.version == "rc-0" and long_trajectory.identity == identity_v0,
        )
        record(
            checks,
            "inflight_engine_reported_a_different_version_at_completion",
            long_trajectory.spans_an_update is True,
        )
        record(
            checks,
            "inflight_admission_agreed_with_the_engine",
            long_trajectory.admission_agrees is True,
        )
        record(checks, "inflight_is_replay_ready", long_trajectory.replay_ready)
        record(
            checks,
            "inflight_text_is_the_step_0_weights",
            only_bos(str(outcome.get("text", ""))),
        )
        record(
            checks,
            "post_update_bound_to_step_1",
            r2_trajectory.version == "rc-1" and r2_trajectory.identity == identity_v1,
        )
        record(checks, "post_update_is_replay_ready", r2_trajectory.replay_ready)
        record(checks, "no_manifest_only_trajectories", not recorder.manifest_only())
        record(
            checks,
            "trajectories_round_trip",
            restored == recorder.trajectories,
        )
        record(checks, "cycle_committed_rc1", ctrl.current_version.label == "rc-1")
        record(checks, "engine_label_rc1", weight_info() == "rc-1")
        record(checks, "server_never_restarted", server.poll() is None)

        report["states_visited"] = [s.value for s in result.states_visited]
        report["trajectory_file"] = str(written)
        report["detail"] = detail
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 5 ===")
        print(f"  step 0 : {identity_v0.describe()}")
        print(f"  step 1 : {identity_v1.describe()}")
        print(
            f"  R-long : bound {long_trajectory.version}, engine said "
            f"{long_trajectory.engine_version_at_completion} at completion"
        )
        print(f"  R2     : bound {r2_trajectory.version}")
        print(f"  file   : {written}")
        print(
            f"  RESULT: {'PASS' if report['ok'] else 'FAIL'} "
            f"({sum(checks.values())}/{len(checks)} checks)"
        )
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[error] {report['error']}")
        print(f"[server] last lines:\n{server_log_tail()}")
        return 1
    finally:
        report["detail"] = detail
        report["checks"] = checks
        out = _REPO / "results" / "phase5.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")
        _terminate(server)


if __name__ == "__main__":
    sys.exit(main())
