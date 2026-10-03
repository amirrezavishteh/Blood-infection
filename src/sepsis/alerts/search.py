"""Alert-policy grid search under simulated alert budgets (validation data only).

For each budget (alerts per 100 observed patient-days) the configuration with
the highest early event sensitivity whose alert rate stays within budget is
chosen (ties broken by alert precision). Budgets are research comparison
points, not clinically approved limits; infeasible budgets are reported.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from sepsis.alerts.matching import StayAlerts, evaluate_alerts
from sepsis.alerts.policy import AlertPolicy
from sepsis.labels.targets import onset_from_challenge_labels


def fast_alert_hours(policy: AlertPolicy, hours: np.ndarray, scores: np.ndarray, scored: np.ndarray) -> list[int]:
    """Same semantics as ``policy.step`` iterated (verified by tests), without dict churn."""
    thr, k, cd = policy.threshold, policy.consecutive, policy.cooldown_h
    state = 0  # 0 monitoring, 1 alerted, 2 cooldown, 3 data_unavailable
    consec = 0
    cooldown_until = None
    out = []
    for h, s, ok in zip(hours, scores, scored):
        in_cd = cooldown_until is not None and h < cooldown_until
        if not ok:
            consec = 0
            state = 3
            continue
        above = s >= thr
        if state == 1:
            if not above:
                state = 2 if in_cd else 0
                consec = 0
            continue
        if (state == 2 or state == 3) and in_cd:
            state = 2
            consec = 0
            continue
        state = 0
        consec = consec + 1 if above else 0
        if consec >= k:
            state = 1
            consec = 0
            cooldown_until = h + cd if cd > 0 else None
            out.append(int(h))
    return out


def prepare_stays(frame: pd.DataFrame) -> list[dict]:
    """``frame``: stay_id, hour_index, score, status, label -> per-stay arrays."""
    f = frame.sort_values(["stay_id", "hour_index"], kind="stable")
    out = []
    for sid, g in f.groupby("stay_id", sort=False):
        lab = g["label"].to_numpy()
        info = onset_from_challenge_labels(lab)
        sc = g["score"].to_numpy(dtype=float)
        ok = (g["status"].to_numpy() == "scored") & ~np.isnan(sc)
        out.append({"stay_id": sid, "hours": g["hour_index"].to_numpy(), "scores": sc, "scored": ok,
                    "max": float(np.nanmax(np.where(ok, sc, np.nan))) if ok.any() else -1.0,
                    "info": info, "n": len(g)})
    return out


def alerts_for(policy: AlertPolicy, stays: list[dict]) -> list[StayAlerts]:
    res = []
    for s in stays:
        hrs = [] if s["max"] < policy.threshold else fast_alert_hours(policy, s["hours"], s["scores"], s["scored"])
        i = s["info"]
        res.append(StayAlerts(s["stay_id"], s["n"], hrs, i.onset_hour, i.septic, i.left_censored))
    return res


def search(frame: pd.DataFrame, budgets=(1, 2, 5, 10, 20), consecutive=(1, 2, 3),
           cooldowns=(0, 6, 12), n_thresholds: int = 40) -> dict:
    stays = prepare_stays(frame)
    sc = frame.loc[frame["status"] == "scored", "score"].to_numpy(dtype=float)
    thresholds = np.unique(np.round(np.quantile(sc, np.linspace(0.80, 0.9995, n_thresholds)), 6))
    grid = []
    for thr, k, cd in itertools.product(thresholds, consecutive, cooldowns):
        pol = AlertPolicy(float(thr), int(k), int(cd))
        m = evaluate_alerts(alerts_for(pol, stays))
        grid.append({**pol.to_dict(), **{key: m[key] for key in (
            "alerts_per_100_patient_days", "early_sensitivity", "six_hour_warning_rate",
            "alert_precision", "lead_time_median_h", "false_alerts_nonseptic_per_100_patient_days")}})
    chosen = {}
    for b in budgets:
        ok = [g for g in grid if g["alerts_per_100_patient_days"] <= b]
        if not ok:
            chosen[str(b)] = {"feasible": False}
            continue
        best = max(ok, key=lambda g: (g["early_sensitivity"], np.nan_to_num(g["alert_precision"])))
        chosen[str(b)] = {"feasible": True, **best}
    return {"grid": grid, "by_budget": chosen, "budgets": list(budgets)}
