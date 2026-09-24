# SPDX-License-Identifier: Apache-2.0
"""A minimal in-process stand-in for ``vllm serve`` with dev mode enabled.

Purpose: let ``scripts/live_control_plane_smoke.py`` be exercised end to end --
including its failure paths -- on a machine with no GPU and no vLLM, so that
GPU time on the real server is not spent debugging the harness.

It models only the behaviours Phase 3A depends on, and models them faithfully:

* ``weight_version`` starts as the literal ``"default"`` and is only written by
  ``/update_weight_version``.
* ``/pause?mode=wait`` sets paused **first**, then blocks until ``active`` reaches
  zero -- so the pause genuinely waits for in-flight work, exactly like
  ``EngineCoreProc.pause_scheduler`` (``vllm/v1/engine/core.py:1984-2026``).
* ``/pause?mode=wait`` can be told to answer ``400`` the way the *in-process*
  engine does (``vllm/v1/engine/core.py:902``), so the harness's "multiprocessing
  is required" failure mode can be tested.
* ``/reset_prefix_cache`` returns ``{"success": false}`` while requests are
  active (``vllm/v1/core/sched/scheduler.py:2706-2715``).
* The weight-transfer endpoints are **not** implemented: Phase 3A must never call
  them, and a wrong call should fail loudly rather than be silently accepted.
* ``--drift-after-resume`` makes the model return different tokens once
  ``/resume`` has been called, which is how the harness's post-resume
  determinism check is shown to have teeth.

Run it standalone::

    PYTHONPATH=src:tests python3 tests/fake_dev_server.py --port 8123
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_VLLM_SHA = "00b7847c8036b667742b4efb21aab1de51fd4721"


class StubState:
    """Mutable engine state, guarded by one lock."""

    def __init__(
        self,
        *,
        vllm_sha: str = DEFAULT_VLLM_SHA,
        initial_label: str = "default",
        drift_after_resume: bool = False,
        token_delay: float = 0.02,
        reject_wait_mode: bool = False,
        pause_timeout: float = 60.0,
    ) -> None:
        self.lock = threading.Lock()
        self.vllm_sha = vllm_sha
        self.weight_version = initial_label
        self.paused = False
        self.active = 0
        self.resumed_once = False
        self.drift_after_resume = drift_after_resume
        self.token_delay = token_delay
        self.reject_wait_mode = reject_wait_mode
        self.pause_timeout = pause_timeout
        self.calls: list[str] = []

    def note(self, path: str) -> None:
        with self.lock:
            self.calls.append(path)

    def calls_for(self, path: str) -> int:
        with self.lock:
            return sum(1 for c in self.calls if c == path)

    def tokens_for(self, prompt: str, count: int) -> list[int]:
        with self.lock:
            drift = self.drift_after_resume and self.resumed_once
        salt = "post-resume" if drift else "baseline"
        tokens = []
        for index in range(count):
            digest = hashlib.sha256(f"{salt}|{prompt}|{index}".encode()).hexdigest()
            tokens.append(int(digest[:6], 16) % 50_000)
        return tokens


class StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-stub/3a"

    @property
    def state(self) -> StubState:
        state: StubState = self.server.state  # type: ignore[attr-defined]
        return state

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # quiet

    # ------------------------------------------------------------- responses

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _empty(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _query(self) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

    # --------------------------------------------------------------- routing

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        self.state.note(path)
        if path == "/health":
            self._empty()
        elif path == "/version":
            self._json(200, {"version": f"0.0.0.dev0+g{self.state.vllm_sha[:8]}"})
        elif path == "/weight_info":
            with self.state.lock:
                label = self.state.weight_version
            self._json(200, {"weight_version": label})
        elif path == "/is_paused":
            with self.state.lock:
                paused = self.state.paused
            self._json(200, {"is_paused": paused})
        elif path == "/get_world_size":
            self._json(200, {"world_size": 1})
        elif path == "/metrics":
            self._empty()
        else:
            self._json(404, {"error": f"no such endpoint: {path}"})

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        self.state.note(path)
        if path == "/update_weight_version":
            body = self._body()
            new_version = body.get("new_version")
            if not isinstance(new_version, str):
                self._json(400, {"error": "missing new_version"})
                return
            with self.state.lock:
                self.state.weight_version = new_version
            self._json(200, {"success": True, "new_version": new_version})
        elif path == "/pause":
            self._pause()
        elif path == "/resume":
            with self.state.lock:
                self.state.paused = False
                self.state.resumed_once = True
            self._json(200, {"status": "resumed"})
        elif path == "/abort_requests":
            with self.state.lock:
                self.state.active = 0
            self._json(200, {"status": "aborted"})
        elif path == "/reset_prefix_cache":
            with self.state.lock:
                busy = self.state.active > 0
            self._json(200, {"success": not busy})
        elif path in ("/reset_encoder_cache", "/reset_mm_cache"):
            self._empty()
        elif path in (
            "/init_weight_transfer_engine",
            "/start_weight_update",
            "/update_weights",
            "/finish_weight_update",
        ):
            # Phase 3A must not touch the weight-transfer surface at all.
            self._json(501, {"error": f"{path} is not implemented by the Phase 3A stub"})
        elif path == "/v1/completions":
            self._completions()
        else:
            self._json(404, {"error": f"no such endpoint: {path}"})

    # ------------------------------------------------------------- handlers

    def _pause(self) -> None:
        query = self._query()
        mode = (query.get("mode") or ["abort"])[0]
        if mode == "wait" and self.state.reject_wait_mode:
            # Faithful to the in-process engine: vllm/v1/engine/core.py:902
            self._json(400, {"error": "'wait' mode can't be used in inproc-engine mode"})
            return
        with self.state.lock:
            self.state.paused = True
        if mode == "wait":
            deadline = time.monotonic() + self.state.pause_timeout
            while True:
                with self.state.lock:
                    active = self.state.active
                if active == 0:
                    break
                if time.monotonic() > deadline:
                    self._json(500, {"error": "pause(wait) timed out"})
                    return
                time.sleep(0.005)
        self._json(200, {"status": "paused"})

    def _completions(self) -> None:
        body = self._body()
        prompt = str(body.get("prompt") or "")
        max_tokens = int(body.get("max_tokens") or 16)
        stream = bool(body.get("stream"))
        with self.state.lock:
            self.state.active += 1
        try:
            tokens = self.state.tokens_for(prompt, max_tokens)
            if stream:
                self._stream_completion(tokens)
            else:
                self._json(
                    200,
                    {
                        "choices": [
                            {
                                "text": " ".join(f"tok{t}" for t in tokens),
                                "token_ids": tokens,
                                "finish_reason": "length",
                                "index": 0,
                            }
                        ]
                    },
                )
        finally:
            with self.state.lock:
                self.state.active -= 1

    def _stream_completion(self, tokens: list[int]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for token in tokens:
            chunk = {
                "choices": [{"text": f"tok{token} ", "token_ids": [token], "index": 0}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(self.state.token_delay)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


class StubServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0, **state_kwargs: Any) -> None:
        super().__init__((host, port), StubHandler)
        self.state = StubState(**state_kwargs)

    @property
    def base_url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"


@contextlib.contextmanager
def running(**kwargs: Any) -> Iterator[StubServer]:
    """Run a stub server on a free port for the duration of the context."""
    server = StubServer(**kwargs)
    thread = threading.Thread(target=server.serve_forever, name="stub-dev-server", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    if str(here.parent.parent / "src") not in sys.path:
        sys.path.insert(0, str(here.parent.parent / "src"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--initial-label", default="default")
    ap.add_argument("--vllm-sha", default=DEFAULT_VLLM_SHA)
    ap.add_argument("--token-delay", type=float, default=0.02)
    ap.add_argument("--drift-after-resume", action="store_true")
    ap.add_argument("--reject-wait-mode", action="store_true")
    args = ap.parse_args(argv)

    server = StubServer(
        host=args.host,
        port=args.port,
        vllm_sha=args.vllm_sha,
        initial_label=args.initial_label,
        token_delay=args.token_delay,
        drift_after_resume=args.drift_after_resume,
        reject_wait_mode=args.reject_wait_mode,
    )
    print(f"stub vLLM dev server on {server.base_url} (label={args.initial_label!r})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
