#!/usr/bin/env bash
# Clean-machine smoke test on a synthetic fixture (no download, ~1-2 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
# Interpreter: $PY if given (a path or a command on PATH), else the project venv, else python3/python.
if [ -z "${PY:-}" ]; then
  for c in .venv/Scripts/python .venv/bin/python python3 python; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || { echo "no Python interpreter found (set PY=...)"; exit 1; }
WS=$(mktemp -d)
S="$PY -m sepsis --data-dir $WS/data --artifact-dir $WS/artifacts"
$S fixture --out "$WS/data/raw/physionet2019" --n-a 400 --n-b 200
$S validate --allow-partial
$S prepare --allow-partial --workers 2
RUN=$($S train --config configs/lightgbm.yaml --run-id smoke-lgbm)
$S calibrate --run "$RUN"
$S select-policy --run "$RUN"
$S evaluate --run "$RUN" --split a_test --n-boot 20 > /dev/null
$S evaluate --run "$RUN" --split b_external --n-boot 20 > /dev/null
$S report
echo "smoke OK: reports in $WS/artifacts/runs/$RUN/eval"
