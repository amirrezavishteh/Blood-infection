"""Adapter for the pinned official PhysioNet 2019 evaluator.

Writes one label file and one prediction file per record (the format the
official script expects), runs ``evaluate_sepsis_score`` from
``third_party/evaluation_2019`` (commit 467c49b, BSD-2) and keeps the files
and its output for audit.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.paths import PROJECT_ROOT

EVALUATOR_PATH = PROJECT_ROOT / "third_party" / "evaluation_2019" / "evaluate_sepsis_score.py"
EVALUATOR_COMMIT = "467c49b514542be7a4a0bafe40fa2c3b064dda2e"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_sepsis_score", EVALUATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_official_files(frame: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """``frame``: stay_id, hour_index, label, probability, prediction (complete hourly series)."""
    out_dir = Path(out_dir)
    lab_dir, pred_dir = out_dir / "labels", out_dir / "predictions"
    lab_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for sid, g in frame.sort_values(["stay_id", "hour_index"]).groupby("stay_id", sort=True):
        name = sid.replace(":", "_") + ".psv"
        (lab_dir / name).write_text("SepsisLabel\n" + "\n".join(str(int(v)) for v in g["label"]) + "\n")
        rows = [f"{p:.6f}|{int(b)}" for p, b in zip(g["probability"], g["prediction"])]
        (pred_dir / name).write_text("PredictedProbability|PredictedLabel\n" + "\n".join(rows) + "\n")
    return lab_dir, pred_dir


def run_official(frame: pd.DataFrame, out_dir: Path) -> dict:
    if frame[["probability", "prediction"]].isna().any().any():
        raise ValueError("official evaluation needs a numeric prediction for every hour")
    lab_dir, pred_dir = write_official_files(frame, out_dir)
    mod = load_evaluator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auroc, auprc, acc, f1, util = mod.evaluate_sepsis_score(str(lab_dir), str(pred_dir))
    res = {"AUROC": float(auroc), "AUPRC": float(auprc), "Accuracy": float(acc),
           "F-measure": float(f1), "Utility": float(util), "evaluator_commit": EVALUATOR_COMMIT,
           "records": int(frame["stay_id"].nunique()), "rows": int(len(frame))}
    (Path(out_dir) / "official_scores.json").write_text(json.dumps(res, indent=1))
    (Path(out_dir) / "official_scores.psv").write_text(
        "AUROC|AUPRC|Accuracy|F-measure|Utility\n"
        f"{auroc}|{auprc}|{acc}|{f1}|{util}\n")
    return res


def official_utility_only(labels_by_stay: list[np.ndarray], preds_by_stay: list[np.ndarray]) -> float:
    """Normalised utility computed with the official per-record function (for tests)."""
    mod = load_evaluator()
    obs = best = inaction = 0.0
    for lab, pr in zip(labels_by_stay, preds_by_stay):
        n = len(lab)
        bp = np.zeros(n)
        if np.any(lab):
            ts = int(np.argmax(lab)) + 6
            bp[max(0, ts - 12): min(ts + 3 + 1, n)] = 1
        obs += mod.compute_prediction_utility(lab, pr)
        best += mod.compute_prediction_utility(lab, bp)
        inaction += mod.compute_prediction_utility(lab, np.zeros(n))
    return (obs - inaction) / (best - inaction)
