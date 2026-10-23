#!/usr/bin/env bash
# The full check (brief §9): ruff, every test, then the validation script, which rewrites
# docs/VALIDATION.md. Run from anywhere:  bash scripts/check_all.sh
# Uses $PYTHON if set, otherwise `python` (activate the virtual environment first).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONUTF8=1  # Windows consoles default to cp1252, which cannot print °, ψ or δ

echo "== environment"
"$PY" --version
echo "== ruff"
"$PY" -m ruff check .
echo "== pytest"
"$PY" -m pytest -q
echo "== validation"
"$PY" scripts/validate.py
echo "== ALL CHECKS PASSED"
