"""The weighted bootstrap reproduces the naive record-resampling bootstrap exactly."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis.evaluation.metrics import _Presorted, bootstrap_ci
from sepsis.evaluation.utility import normalized_utility


def _naive(codes, y, p, preds, n_boot, seed):
    stays = np.unique(codes)
    slices = [np.flatnonzero(codes == s) for s in stays]
    rng = np.random.default_rng(seed)
    au, ap, ut = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(slices), len(slices))
        idx = np.concatenate([slices[i] for i in pick])
        relabel = np.repeat(np.arange(len(pick)), [len(slices[i]) for i in pick])
        au.append(roc_auc_score(y[idx], p[idx]))
        ap.append(average_precision_score(y[idx], p[idx]))
        ut.append(normalized_utility(relabel, y[idx], preds[idx]))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return pct(au), pct(ap), pct(ut)


def _cohort(seed=3, n_stays=120):
    rng = np.random.default_rng(seed)
    codes, ys = [], []
    for s in range(n_stays):
        n = int(rng.integers(8, 60))
        lab = np.zeros(n)
        if rng.random() < 0.3:
            lab[int(rng.integers(0, n)):] = 1
        codes += [s] * n
        ys.append(lab)
    y = np.concatenate(ys).astype(int)
    p = np.round(np.clip(0.3 * y + rng.random(len(y)) * 0.7, 0, 1), 2)  # rounded -> many ties
    return np.array(codes), y, p, p > 0.6


def test_weighted_metrics_equal_sklearn_with_weights():
    codes, y, p, _ = _cohort()
    w = np.random.default_rng(1).integers(0, 4, len(y)).astype(float)
    a, b = _Presorted(y, p).metrics(w)
    assert abs(a - roc_auc_score(y, p, sample_weight=w)) < 1e-12
    assert abs(b - average_precision_score(y, p, sample_weight=w)) < 1e-12


def test_bootstrap_matches_naive_resampling():
    codes, y, p, preds = _cohort()
    fast = bootstrap_ci(codes, y, p, preds=preds, n_boot=40, seed=7)
    au, ap, ut = _naive(codes, y, p, preds, n_boot=40, seed=7)
    np.testing.assert_allclose(fast["auroc_ci"], au, atol=1e-12)
    np.testing.assert_allclose(fast["auprc_ci"], ap, atol=1e-12)
    np.testing.assert_allclose(fast["utility_ci"], ut, atol=1e-12)
