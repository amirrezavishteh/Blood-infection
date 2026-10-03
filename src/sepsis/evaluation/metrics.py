"""Hourly discrimination, calibration, subgroup and coverage metrics.

Uncertainty comes from bootstrapping whole records (stays); correlated hourly
rows are never resampled independently. PhysioNet 2019 has no person
linkage, so records are the resampling unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sepsis.evaluation.utility import normalized_utility, sepsis_times


def _stay_slices(stay_codes: np.ndarray) -> list[np.ndarray]:
    n = len(stay_codes)
    new = np.ones(n, dtype=bool)
    new[1:] = stay_codes[1:] != stay_codes[:-1]
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n)
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def _safe(fn, y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, p))


def bootstrap_ci(stay_codes, y, p, preds=None, n_boot: int = 200, seed: int = 0) -> dict:
    """Percentile 95% CIs for AUROC, AUPRC (and utility if binary preds given)."""
    slices = _stay_slices(stay_codes)
    rng = np.random.default_rng(seed)
    au, ap, ut = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(slices), len(slices))
        idx = np.concatenate([slices[i] for i in pick])
        # relabel stays so resampled duplicates stay distinct groups
        codes = np.repeat(np.arange(len(pick)), [len(slices[i]) for i in pick])
        yy, pp = y[idx], p[idx]
        au.append(_safe(roc_auc_score, yy, pp))
        ap.append(_safe(average_precision_score, yy, pp))
        if preds is not None:
            ut.append(normalized_utility(codes, yy, preds[idx]))
    out = {"auroc_ci": _pct(au), "auprc_ci": _pct(ap), "n_boot": n_boot, "unit": "record"}
    if preds is not None:
        out["utility_ci"] = _pct(ut)
    return out


def _pct(vals) -> list[float] | None:
    v = np.asarray([x for x in vals if x == x])
    if not len(v):
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]),
                    "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                    "observed_rate": float(y[m].mean())})
    return out


def hourly_metrics(frame: pd.DataFrame, n_boot: int = 200, seed: int = 0, probabilistic: bool = True) -> dict:
    """``frame``: stay_id, hour_index, label, score (NaN when unavailable), prediction."""
    f = frame.sort_values(["stay_id", "hour_index"], kind="stable")
    scored = f[f["score"].notna()]
    codes = scored["stay_id"].to_numpy()
    y = scored["label"].to_numpy().astype(int)
    p = scored["score"].to_numpy().astype(float)
    res = {
        "rows_scored": int(len(scored)),
        "positive_hour_prevalence": float(y.mean()) if len(y) else float("nan"),
        "auroc": _safe(roc_auc_score, y, p),
        "auprc": _safe(average_precision_score, y, p),
    }
    if probabilistic:
        res["brier"] = float(brier_score_loss(y, p)) if len(y) else float("nan")
        res["reliability"] = reliability(y, p)
    if n_boot:
        res.update(bootstrap_ci(codes, y, p, n_boot=n_boot, seed=seed))
    return res


def utility_with_ci(frame: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict:
    """Normalised utility on the complete hourly series (fallback rows included)."""
    f = frame.sort_values(["stay_id", "hour_index"], kind="stable")
    codes = f["stay_id"].to_numpy()
    y = f["label"].to_numpy().astype(int)
    pred = f["prediction"].to_numpy().astype(bool)
    t, ts = sepsis_times(codes, y)
    res = {"utility": normalized_utility(codes, y, pred, t, ts)}
    if n_boot:
        p = f["probability"].to_numpy().astype(float)
        res["utility_ci"] = bootstrap_ci(codes, y, p, preds=pred, n_boot=n_boot, seed=seed)["utility_ci"]
    return res


def coverage(frame: pd.DataFrame) -> dict:
    counts = frame["status"].value_counts().to_dict()
    n = len(frame)
    return {"scheduled_hours": int(n), "scored_hours": int(counts.get("scored", 0)),
            "coverage": float(counts.get("scored", 0) / n) if n else float("nan"),
            "unavailable_by_reason": {k: int(v) for k, v in counts.items() if k != "scored"}}


AGE_BANDS = [(0, 50, "<50"), (50, 65, "50-64"), (65, 80, "65-79"), (80, 200, ">=80")]


def subgroup_metrics(frame: pd.DataFrame, demo: pd.DataFrame, n_boot: int = 100,
                     min_events: int = 30) -> list[dict]:
    """Age band and recorded Gender (source code) subgroups with counts and CIs.

    ``demo``: stay_id, Age, Gender (first row values, unchanged source fields).
    Estimates with fewer than ``min_events`` septic records are flagged unstable.
    """
    f = frame.merge(demo, on="stay_id", how="left")
    groups = []
    for lo, hi, name in AGE_BANDS:
        groups.append((f"age {name}", (f["Age"] >= lo) & (f["Age"] < hi)))
    for g in sorted(f["Gender"].dropna().unique()):
        groups.append((f"Gender={int(g)} (source code)", f["Gender"] == g))
    out = []
    for name, m in groups:
        sub = f[m & f["score"].notna()].sort_values(["stay_id", "hour_index"])
        if sub.empty:
            continue
        events = int(sub.groupby("stay_id")["label"].max().sum())
        y, p = sub["label"].to_numpy().astype(int), sub["score"].to_numpy()
        row = {"subgroup": name, "stays": int(sub["stay_id"].nunique()), "septic_stays": events,
               "rows": int(len(sub)), "auroc": _safe(roc_auc_score, y, p),
               "auprc": _safe(average_precision_score, y, p), "unstable": events < min_events}
        if n_boot and events >= 2:
            row.update(bootstrap_ci(sub["stay_id"].to_numpy(), y, p, n_boot=n_boot))
        out.append(row)
    return out
