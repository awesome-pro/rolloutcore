# SPDX-License-Identifier: Apache-2.0
"""Hygiene checks for the pod-only diagnostic scripts.

``diagnostics/*.py`` cannot be imported on a laptop -- they import ``torch`` and
vLLM at module scope -- so the properties that matter are enforced by *reading*
them rather than running them.

The rule below exists because Phase 3C sat in its readiness loop until the 900 s
deadline against a perfectly healthy server. ``/health`` is
``Response(status_code=200)`` with an **empty body**
(``serve/instrumentator/health.py:28``), so the script's JSON-parsing helper
raised on every single poll, and the ``except Exception: pass`` around it turned
that into silence. The server was ready the whole time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = [*sorted(REPO.glob("scripts/*.py")), *sorted(REPO.glob("diagnostics/*.py"))]


def _is_json_module(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "json") or (
        isinstance(node, ast.Attribute) and node.attr == "json"
    )


def json_parsing_helpers(text: str) -> set[str]:
    """Module-level helpers that decode a JSON response body.

    Parsed rather than pattern-matched: ``probe_ok``'s docstring *mentions*
    ``json.loads`` to explain why it must not use it, and a regex cannot tell
    that apart from calling it.
    """
    names: set[str] = set()
    for fn in ast.parse(text).body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attr, value = node.func.attr, node.func.value
            if attr == "json" or (attr == "loads" and _is_json_module(value)):
                names.add(fn.name)
    return names


def offending_lines(text: str) -> list[int]:
    """1-based lines that probe ``/health`` through a JSON-parsing helper."""
    helpers = json_parsing_helpers(text)
    if not helpers:
        return []
    call = re.compile(r"\b(" + "|".join(sorted(helpers)) + r")\s*\(")
    return [
        n for n, line in enumerate(text.splitlines(), 1) if "/health" in line and call.search(line)
    ]


def test_the_rule_catches_the_phase3c_bug() -> None:
    """The detector is only worth having if it fails on the real regression."""
    bug = (
        "def get(url: str) -> object:\n"
        "    return json.loads(urlopen(url).read())\n"
        "\n"
        "while True:\n"
        '    if get(f"{BASE}/health") is not None:\n'
        "        break\n"
    )
    assert json_parsing_helpers(bug) == {"get"}
    assert offending_lines(bug) == [5]


def test_health_is_never_probed_through_a_json_parser() -> None:
    offenders = [
        f"{path.relative_to(REPO)}:{n}"
        for path in SCRIPTS
        if (text := path.read_text(encoding="utf-8"))
        for n in offending_lines(text)
    ]
    assert not offenders, (
        "/health returns 200 with an empty body, so a JSON-parsing probe can "
        f"never succeed: {offenders}. Use a status-only probe."
    )


def test_the_status_only_probe_exists_where_health_is_polled() -> None:
    """Both long-running scripts must actually wait on a status-only probe."""
    for name in ("diagnostics/rolloutcore_cycle_2gpu.py", "diagnostics/upstream_nccl_2gpu.py"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "/health" in text, f"{name} no longer waits for readiness at all"
        assert not offending_lines(text), name


def killpg_without_own_session(text: str) -> bool:
    """True if a file signals a process *group* it never put in its own session.

    ``os.killpg`` needs the target to be a group leader, which is what
    ``start_new_session=True`` on the ``Popen`` buys. Without it the child
    inherits this process's group, so ``killpg`` --- meant to take down a
    ``vllm serve`` and its EngineCore child --- takes down the harness instead.

    That is not hypothetical: Phase 4B's first two pod runs ended with the shell
    printing "Terminated" because the script SIGTERM'd itself at the start of
    scenario C, which is also why no artifact was written.
    """
    return "os.killpg" in text and "start_new_session=True" not in text


def test_a_script_that_kills_a_process_group_creates_one() -> None:
    assert killpg_without_own_session("import os\nos.killpg(os.getpgid(pid), 9)\n")
    assert not killpg_without_own_session(
        "import os\nPopen(cmd, start_new_session=True)\nos.killpg(pgid, 9)\n"
    )


def test_no_script_kills_its_own_process_group() -> None:
    offenders = [
        path.relative_to(REPO).as_posix()
        for path in SCRIPTS
        if killpg_without_own_session(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"{offenders} call os.killpg on a child that shares their own process "
        "group, so the kill would hit the script itself. Start the child with "
        "start_new_session=True."
    )
