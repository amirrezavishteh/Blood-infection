"""The vectorised utility equals the pinned official evaluator."""

import numpy as np
import pandas as pd

from sepsis.evaluation.official import load_evaluator, official_utility_only, run_official
from sepsis.evaluation.utility import normalized_utility, row_utilities, sepsis_times


def test_official_docstring_example():
    mod = load_evaluator()
    assert abs(mod.compute_prediction_utility(np.array([0, 0, 0, 0, 1, 1]),
                                              np.array([0, 0, 1, 1, 1, 1])) - 3.388888888888889) < 1e-12
    t, ts = sepsis_times(np.zeros(6), np.array([0, 0, 0, 0, 1, 1]))
    assert abs(row_utilities(t, ts, np.array([0, 0, 1, 1, 1, 1])).sum() - 3.388888888888889) < 1e-12


def test_vectorised_matches_official_random():
    rng = np.random.default_rng(42)
    labs, preds, codes = [], [], []
    for i in range(150):
        n = int(rng.integers(5, 80))
        lab = np.zeros(n)
        r = rng.random()
        if r < 0.3:
            lab[int(rng.integers(0, n)):] = 1
        elif r < 0.35:
            lab[:] = 1
        pr = (rng.random(n) < rng.random()).astype(float)
        labs.append(lab); preds.append(pr); codes += [i] * n
    ours = normalized_utility(np.array(codes), np.concatenate(labs), np.concatenate(preds))
    official = official_utility_only(labs, preds)
    assert abs(ours - official) < 1e-10


def test_run_official_end_to_end(tmp_path):
    rows = []
    for s, n, start in (("x:A:1", 20, 10), ("x:A:2", 15, None), ("x:A:3", 8, 0)):
        for h in range(n):
            lab = int(start is not None and h >= start)
            p = 0.9 if lab else 0.1
            rows.append((s, h, lab, p, int(p > 0.5)))
    fr = pd.DataFrame(rows, columns=["stay_id", "hour_index", "label", "probability", "prediction"])
    res = run_official(fr, tmp_path)
    ours = normalized_utility(fr["stay_id"].to_numpy(), fr["label"].to_numpy(), fr["prediction"].to_numpy())
    assert abs(res["Utility"] - ours) < 1e-10
    assert res["AUROC"] == 1.0
    assert (tmp_path / "official_scores.psv").exists()
