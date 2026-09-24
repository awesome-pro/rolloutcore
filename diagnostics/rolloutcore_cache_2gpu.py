#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4C: does a weight update leak KV cache computed under the old weights?

This is the second guarantee RolloutCore claims, and like the first it has **no
engine-side support**. vLLM's prefix cache maps token blocks to KV blocks and
knows nothing about which weights produced them; `reset_prefix_cache` exists
precisely because the engine will not invalidate on its own -- the only callers in
the tree are the explicit engine APIs (`vllm/v1/engine/llm_engine.py:361`,
`async_llm.py:1089`), never the weight-update path.

So the reset is RolloutCore's job, and this measures two things:

**Phase 1 -- the guarantee, and it gates.** With prefix caching on (the default,
`vllm/config/cache.py:142`), prompt ``P`` is sent twice at rc-0 to prove the cache
works and is measurable, then a full cycle installs rc-1, then ``P`` is sent
again. The first post-update request must score **zero** new cache hits -- the
tokens were all computed under the old weights, so reusing any of that KV would
mean a trajectory whose output came from two different weight sets. A second
request then has to score hits again, or "no reuse" would be indistinguishable
from "caching silently turned off".

The cycle's drain now pauses with ``clear_cache=false`` (amendment 7,
``docs/state-machine.md`` "The invalidation boundary"), so Phase 1 attributes the
zero to ``INVALIDATING`` alone. The artifact committed as
``results/phase4c.json`` was produced before that amendment, when the pause also
cleared; Phase 2 is unaffected either way, because it drives the update out of
band and clears nothing.

**Phase 2 -- the hazard, and it is recorded rather than asserted.** The trainer's
first decoder layer is negated in place, and the update is driven *out of band*:
`/pause?mode=wait&clear_cache=false`, broadcast, `/resume`. No controller, no
reset. If the cache is still handed back afterwards, then RolloutCore's reset step
is load-bearing -- and it is: 32 hits came back, computed under the previous
weights, with different tokens. The fact that the negated weights keep the *same*
manifest digest is the manifest-only blind spot that item 5 exists to fix,
measured rather than argued.

Server: `--load-format dummy` so rc-0 is visibly degenerate.

    python3 diagnostics/rolloutcore_cache_2gpu.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
    UpdateTarget,
    WeightVersion,
)
from rolloutcore.adapters import HttpVLLMAdapter  # noqa: E402
from rolloutcore.adapters.nccl import NCCLWeightTransferDriver  # noqa: E402

MODEL_NAME = "facebook/opt-125m"
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"
SERVER_LOG = _REPO / "results" / "phase4c-server.log"

INFERENCE_TP_SIZE = 1
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE_INDEX = 1

#: Long enough to fill several KV blocks: a prefix-cache hit needs whole cached
#: blocks, and the 5-token prompts the other harnesses use would cache nothing.
PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is Madrid. "
    "The capital of Portugal is"
)
MAX_TOKENS = 16


# ------------------------------------------------------------------- server


def start_vllm_server() -> subprocess.Popen[str]:
    """Launch `vllm serve`. Prefix caching is on by default; we do not ask."""
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
    """Stop the server: SIGTERM, then SIGKILL if it will not go. Idempotent."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _await_ready(proc: subprocess.Popen[str]) -> None:
    """Block until ``/health`` answers, or explain why it never did.

    Status only: ``/health`` is 200 with an empty body
    (``serve/instrumentator/health.py:28``).
    """
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


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def post(url: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else None


def completion(text: str = PROMPT) -> dict[str, Any]:
    body = {
        "model": MODEL_NAME,
        "prompt": text,
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
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = payload["choices"][0]
    return {
        "text": choice.get("text", ""),
        "seconds": round(time.monotonic() - started, 4),
        "weight_version": weight_info(),
    }


def weight_info() -> str:
    return str(get(f"{BASE_URL}/weight_info")["weight_version"])


def is_paused() -> bool:
    return bool(get(f"{BASE_URL}/is_paused")["is_paused"])


def world_size() -> int:
    return int(get(f"{BASE_URL}/get_world_size")["world_size"])


# --------------------------------------------------------------- metrics


def counter(name: str) -> tuple[str, float]:
    """``(metric name actually found, summed value)`` across every label set.

    `prometheus_client` renders counters with a `_total` suffix, so the bare name
    from the source (`vllm/v1/metrics/loggers.py:600`) will not match the scrape.
    Both spellings are tried, and the one that matched is reported so
    "no reuse" cannot be confused with "no metrics".
    """
    text = get_text(f"{BASE_URL}/metrics")
    for candidate in (f"{name}_total", name):
        total = 0.0
        found = False
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            metric, _, value = line.rpartition(" ")
            head = metric.split("{", 1)[0]
            if head != candidate:
                continue
            try:
                total += float(value)
            except ValueError:  # pragma: no cover - malformed scrape line
                continue
            found = True
        if found:
            return candidate, total
    return "", 0.0


def cache_hits() -> float:
    return counter("vllm:prefix_cache_hits")[1]


def cache_queries() -> float:
    return counter("vllm:prefix_cache_queries")[1]


# ------------------------------------------------------------------ misc


def git_revision() -> tuple[str, bool]:
    """``(HEAD, dirty)`` for the code under test. Scoped to the code directories,
    so the pod's tree -- which lacks the tracked files under ``results/`` -- is
    not reported as dirty."""

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
    """Store a check and print it now: a mid-run failure must still show the rest."""
    checks[name] = bool(ok)
    print(f"  [{'PASS' if bool(ok) else 'FAIL'}] {name}")


def only_bos(text: str) -> bool:
    """True for the degenerate all-`<s>` output `--load-format dummy` produces."""
    stripped = text.replace("<s>", "").strip()
    return bool(text) and not stripped


# ------------------------------------------------------------------- main


def main() -> int:
    report: dict[str, Any] = {"phase": "4C", "model": MODEL_NAME, "ok": False}
    sha, dirty = git_revision()
    report["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["rolloutcore_sha"] = sha
    report["rolloutcore_dirty"] = dirty
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
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
        init_info = NCCLTrainerInitInfo(
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=group_size,
            rank=0,
            packed=True,
        )
        driver = NCCLWeightTransferDriver(
            base_url=BASE_URL,
            trainer_init_info=init_info,
            source=ModuleSource(train_model),
            world_size=group_size,
        )
        identity = driver.identity()
        adapter = HttpVLLMAdapter(
            base_url=BASE_URL, weight_identity=identity, driver=driver, drain_timeout=600
        )
        ctrl = LifecycleController()
        runner = LifecycleRunner(ctrl, adapter, drain_polls=600, drain_interval=2.0)

        print("[cycle] bootstrap")
        runner.bootstrap()
        assert ctrl.state is LifecycleState.READY, ctrl.state
        assert weight_info() == "rc-0", weight_info()

        # ==================================== Phase 1: the guarantee
        print("\n--- Phase 1: no cross-version KV reuse after a full cycle ---")
        # One request first: a Prometheus counter may not appear in the scrape
        # until it has been touched, so checking before this would fail for a
        # reason that has nothing to do with the cache.
        cold = completion()
        metric_name, _ = counter("vllm:prefix_cache_hits")
        detail["metrics"] = {
            "hits_metric": metric_name,
            "queries_metric": counter("vllm:prefix_cache_queries")[0],
            "available": bool(metric_name),
        }
        record(checks, "metrics_are_exposed", bool(metric_name))
        if not metric_name:
            raise RuntimeError(
                "/metrics has no vllm:prefix_cache_hits; the guarantee cannot be measured"
            )

        hits_after_cold = cache_hits()
        warm = completion()
        hits_after_warm = cache_hits()
        detail["pre_update"] = {
            "cold_text": cold["text"],
            "warm_text": warm["text"],
            "hits_after_cold": hits_after_cold,
            "hits_after_warm": hits_after_warm,
            "hit_delta_on_repeat": hits_after_warm - hits_after_cold,
            "queries": cache_queries(),
        }
        print(f"[rc-0] cold hits={hits_after_cold:.0f} warm hits={hits_after_warm:.0f}")
        record(
            checks,
            "prefix_cache_is_actually_working",
            hits_after_warm - hits_after_cold > 0,
        )

        target = ctrl.next_target(identity)
        print(f"[cycle] run_cycle -> {target.describe()}")
        started = time.monotonic()
        result = runner.run_cycle(target)
        report["cycle_seconds"] = round(time.monotonic() - started, 4)
        record(checks, "cycle_committed_rc1", ctrl.current_version.label == "rc-1")

        # The first post-update request must reuse nothing.
        hits_before_after = cache_hits()
        after1 = completion()
        hits_after_after1 = cache_hits()
        after2 = completion()
        hits_after_after2 = cache_hits()

        detail["post_update"] = {
            "first_text": after1["text"],
            "first_hit_delta": hits_after_after1 - hits_before_after,
            "second_text": after2["text"],
            "second_hit_delta": hits_after_after2 - hits_after_after1,
            "weight_version": after1["weight_version"],
        }
        print(
            f"[rc-1] first hit delta={detail['post_update']['first_hit_delta']:.0f} "
            f"second hit delta={detail['post_update']['second_hit_delta']:.0f}"
        )
        record(
            checks,
            "no_cross_version_reuse",
            detail["post_update"]["first_hit_delta"] == 0,
        )
        record(
            checks,
            "post_update_output_is_the_new_version",
            after1["text"] != cold["text"] and not only_bos(after1["text"]),
        )
        record(
            checks,
            "cache_still_works_after_update",
            detail["post_update"]["second_hit_delta"] > 0,
        )

        # ============================ Phase 2: the hazard, out of band
        print("\n--- Phase 2: what happens without the reset (recorded, not gating) ---")
        # Populate the cache again at rc-1, then change the weights that produce
        # the KV for the *prefix* while leaving the cache in place.
        seeded = completion()
        hits_before_stale = cache_hits()

        with torch.no_grad():
            for name, param in train_model.named_parameters():
                if name.startswith("model.decoder.layers.0."):
                    param.mul_(-1.0)
        # The manifest is unchanged by construction, so the identity cannot see it.
        fresh_identity = NCCLWeightTransferDriver(
            base_url=BASE_URL,
            trainer_init_info=init_info,
            source=ModuleSource(train_model),
            world_size=group_size,
        ).identity()
        record(
            checks,
            "corrupted_weights_share_the_manifest_digest",
            fresh_identity.digest == identity.digest,
        )

        # Out of band on purpose: no controller, and crucially no cache reset.
        post(f"{BASE_URL}/pause?mode=wait&clear_cache=false", timeout=600)
        stale_target = UpdateTarget(version=WeightVersion(2), identity=identity)
        driver.transfer(stale_target)
        post(f"{BASE_URL}/resume", timeout=60)

        stale = completion()
        hits_after_stale = cache_hits()
        post(
            f"{BASE_URL}/reset_prefix_cache?reset_running_requests=true&reset_external=true",
            timeout=60,
        )
        fresh = completion()
        hits_after_fresh = cache_hits()

        stale_delta = hits_after_stale - hits_before_stale
        fresh_delta = hits_after_fresh - hits_after_stale
        reused = stale_delta > 0
        detail["hazard"] = {
            "seeded_text": seeded["text"],
            "pause_cleared_cache": False,
            "reset_skipped": True,
            "engine_version": weight_info(),
            "hits_with_stale_cache": stale_delta,
            "hits_with_fresh_cache": fresh_delta,
            "text_with_stale_cache": stale["text"],
            "text_with_fresh_cache": fresh["text"],
            "texts_differ": stale["text"] != fresh["text"],
            "kv_reused_across_weights": reused,
            "verdict": (
                "REUSE HAPPENED: KV computed before the weight change was handed "
                "back after it. RolloutCore's reset step is load-bearing."
                if reused
                else "NO REUSE: the engine did not hand back the old KV. The "
                "reset step is defence in depth here, not the only barrier."
            ),
        }
        print(f"[stale] hit delta={stale_delta:.0f}  fresh hit delta={fresh_delta:.0f}")
        print(f"[stale] texts differ: {detail['hazard']['texts_differ']}")
        print(f"[verdict] {detail['hazard']['verdict']}")

        record(checks, "server_never_restarted", server.poll() is None)
        report["states_visited"] = [s.value for s in result.states_visited]
        report["detail"] = detail
        report["checks"] = checks
        report["ok"] = all(checks.values())

        print("\n=== Phase 4C ===")
        print(
            f"  rc-0 cold/warm : {cold['text'][:40]!r} / hits "
            f"{hits_after_cold:.0f} -> {hits_after_warm:.0f}"
        )
        print(
            f"  rc-1 first     : {after1['text'][:40]!r}  "
            f"hits +{detail['post_update']['first_hit_delta']:.0f}"
        )
        print(f"  rc-1 second    : hits +{detail['post_update']['second_hit_delta']:.0f}")
        print(f"  phase 2        : {detail['hazard']['verdict']}")
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
        out = _REPO / "results" / "phase4c.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")
        _terminate(server)


if __name__ == "__main__":
    sys.exit(main())
