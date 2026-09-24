#!/usr/bin/env bash
# RolloutCore checks. Phase 1 and the fake-engine cycle are pure stdlib, so the
# test suite runs on a bare Python >= 3.11 with nothing installed.
#
#   ./scripts/test.sh          tests + lint + types (if the tools exist)
#   ./scripts/test.sh --fast   tests only
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer a local venv if one exists, else whatever python3 is on PATH.
PY=python3
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
fi

echo "==> tests ($PY)"
if "$PY" -c "import pytest" 2>/dev/null; then
    PYTHONPATH=src:tests "$PY" -m pytest "$@"
else
    PYTHONPATH=src python3 -m unittest discover -s tests -t tests -v
fi

[[ "${1:-}" == "--fast" ]] && exit 0

if "$PY" -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
    RUFF=ruff
    [[ -x .venv/bin/ruff ]] && RUFF=.venv/bin/ruff
    echo "==> ruff check"
    "$RUFF" check src tests
    echo "==> ruff format --check"
    "$RUFF" format --check src tests
else
    echo "==> ruff not available, skipping lint"
fi

if "$PY" -c "import mypy" 2>/dev/null; then
    echo "==> mypy"
    "$PY" -m mypy
else
    echo "==> mypy not available, skipping types"
fi

echo "==> demo"
PYTHONPATH=src "$PY" -m rolloutcore.demo >/dev/null
echo "OK"
