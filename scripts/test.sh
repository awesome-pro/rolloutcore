#!/usr/bin/env bash
# Phase 1 test runner. No dependencies: Phase 1 is pure stdlib, so this works on
# a bare Python >= 3.11 with nothing installed.
set -euo pipefail

cd "$(dirname "$0")/.."

if python3 -c "import pytest" 2>/dev/null; then
    echo "==> running with pytest"
    PYTHONPATH=src:tests python3 -m pytest "$@"
else
    echo "==> pytest not installed; running with stdlib unittest"
    PYTHONPATH=src python3 -m unittest discover -s tests -t tests "${@:--v}"
fi
