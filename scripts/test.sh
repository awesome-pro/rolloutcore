#!/usr/bin/env bash
# RolloutCore checks. Phase 1 and the fake-engine cycle are pure stdlib, so the
# test suite runs on a bare Python >= 3.11 with nothing installed.
#
#   ./scripts/test.sh          tests + lint + types (if the tools exist)
#   ./scripts/test.sh --fast   tests only
set -euo pipefail

cd "$(dirname "$0")/.."

# `--fast` is ours; everything else is forwarded to the test runner. (Passing it
# through to pytest made `./scripts/test.sh --fast` fail with "unrecognized
# arguments" while the README advertised it.)
FAST=0
TEST_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fast) FAST=1 ;;
        *) TEST_ARGS+=("$arg") ;;
    esac
done

# Prefer a local venv if one exists, else whatever python3 is on PATH.
PY=python3
if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
fi

echo "==> tests ($PY)"
if "$PY" -c "import pytest" 2>/dev/null; then
    PYTHONPATH=src:tests "$PY" -m pytest ${TEST_ARGS[@]+"${TEST_ARGS[@]}"}
else
    PYTHONPATH=src:tests "$PY" -m unittest discover -s tests -t tests ${TEST_ARGS[@]+"${TEST_ARGS[@]}"}
fi

if [[ "$FAST" == 1 ]]; then
    exit 0
fi

if "$PY" -c "import ruff" 2>/dev/null || command -v ruff >/dev/null 2>&1; then
    RUFF=ruff
    [[ -x .venv/bin/ruff ]] && RUFF=.venv/bin/ruff
    echo "==> ruff check"
    "$RUFF" check src tests scripts
    echo "==> ruff format --check"
    "$RUFF" format --check src tests scripts
else
    echo "==> ruff not available, skipping lint"
fi

if "$PY" -c "import mypy" 2>/dev/null; then
    echo "==> mypy"
    "$PY" -m mypy
else
    echo "==> mypy not available, skipping types"
fi

if [[ -n "${VLLM_CHECKOUT:-}" ]]; then
    echo "==> anchor verification (VLLM_CHECKOUT=$VLLM_CHECKOUT)"
    "$PY" scripts/verify_anchors.py --vllm "$VLLM_CHECKOUT"
else
    echo "==> VLLM_CHECKOUT not set, skipping anchor verification"
fi

echo "==> demo"
PYTHONPATH=src "$PY" -m rolloutcore.demo >/dev/null
echo "OK"
