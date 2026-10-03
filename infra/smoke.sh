#!/usr/bin/env bash
# Clean-machine smoke test on a synthetic fixture (no download, ~1-2 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
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
