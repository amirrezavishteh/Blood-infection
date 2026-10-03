# End-to-end open-data pipeline on native Windows (run from the project root).
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
& $py -m sepsis -v download --dataset physionet2019
& $py -m sepsis validate --config configs/physionet2019.yaml
& $py -m sepsis -v prepare --config configs/challenge2019_state.yaml
$runs = @()
foreach ($cfg in "configs/logistic.yaml", "configs/lightgbm.yaml", "configs/lightgbm_extended.yaml") {
    $run = (& $py -m sepsis -v train --config $cfg | Select-Object -Last 1).Trim()
    & $py -m sepsis calibrate --run $run
    & $py -m sepsis select-policy --run $run --split a_validation
    $runs += $run
}
& $py -m sepsis promote --runs @runs
# Final, frozen evaluations (each run once on each held-out split).
foreach ($run in $runs) {
    & $py -m sepsis evaluate --run $run --split a_test
    & $py -m sepsis evaluate --run $run --split b_external
}
& $py -m sepsis report
