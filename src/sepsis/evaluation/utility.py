"""Vectorised PhysioNet 2019 utility (same constants and piecewise rules as the
pinned official evaluator in ``third_party/evaluation_2019``).

Used for fast threshold search and bootstrap; the official scorer itself is
run on the final hourly prediction files (see ``official.py``), and a test
asserts that both agree.
"""

from __future__ import annotations

import numpy as np

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0
_M1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
_B1 = -_M1 * DT_EARLY
_M2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
_B2 = -_M2 * DT_LATE
_M3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
_B3 = -_M3 * DT_OPTIMAL


def sepsis_times(stay_codes: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: position within stay (t) and the stay's t_sepsis (inf if never septic).

    ``stay_codes`` must be grouped (rows of a stay contiguous and hour-ordered).
    """
    n = len(labels)
    new = np.ones(n, dtype=bool)
    new[1:] = stay_codes[1:] != stay_codes[:-1]
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, n))
    start_of_row = np.repeat(starts, lengths)
    t = (np.arange(n) - start_of_row).astype(np.float64)
    ts_stay = np.full(len(starts), np.inf)
    lab = np.asarray(labels).astype(bool)
    pos_idx = np.flatnonzero(lab)
    if len(pos_idx):
        grp = np.searchsorted(starts, pos_idx, side="right") - 1
        first = np.full(len(starts), -1)
        # first positive row per stay
        order = np.unique(grp, return_index=True)
        first[order[0]] = pos_idx[order[1]]
        has = first >= 0
        ts_stay[has] = (first[has] - starts[has]) - DT_OPTIMAL
    return t, np.repeat(ts_stay, lengths)


def row_utilities(t: np.ndarray, ts: np.ndarray, preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds).astype(bool)
    septic = np.isfinite(ts)
    u = np.zeros(len(t))
    d = t - np.where(septic, ts, 0)
    in_scope = t <= ts + DT_LATE  # always true for non-septic (ts = inf)
    early = d <= DT_OPTIMAL
    late = (d > DT_OPTIMAL) & (d <= DT_LATE)
    tp = in_scope & septic & preds
    u[tp & early] = np.maximum(_M1 * d[tp & early] + _B1, U_FP)
    u[tp & late] = _M2 * d[tp & late] + _B2
    u[in_scope & ~septic & preds] = U_FP
    fn = in_scope & septic & ~preds
    u[fn & late] = _M3 * d[fn & late] + _B3
    return u


def best_predictions(t: np.ndarray, ts: np.ndarray, n_rows_of_row: np.ndarray | None = None) -> np.ndarray:
    septic = np.isfinite(ts)
    return septic & (t >= ts + DT_EARLY) & (t <= ts + DT_LATE)


def normalized_utility(stay_codes, labels, preds, t=None, ts=None) -> float:
    if t is None or ts is None:
        t, ts = sepsis_times(stay_codes, labels)
    obs = row_utilities(t, ts, preds).sum()
    best = row_utilities(t, ts, best_predictions(t, ts)).sum()
    inaction = row_utilities(t, ts, np.zeros(len(t), dtype=bool)).sum()
    return float((obs - inaction) / (best - inaction)) if best != inaction else float("nan")


def best_threshold(stay_codes, labels, scores, grid: np.ndarray | None = None) -> tuple[float, float, list]:
    """Threshold on the hourly score maximising normalised utility (selection data only)."""
    t, ts = sepsis_times(stay_codes, labels)
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1.0)
    if grid is None:
        qs = np.unique(np.quantile(s[s >= 0], np.linspace(0.5, 0.999, 120)))
        grid = qs
    curve = []
    best = (float("nan"), -np.inf)
    for thr in grid:
        u = normalized_utility(stay_codes, labels, s >= thr, t, ts)
        curve.append((float(thr), u))
        if u > best[1]:
            best = (float(thr), u)
    return best[0], best[1], curve
