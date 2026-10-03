"""Application alert metrics (an evaluation convention, not the challenge score).

Per record we know the emitted alert hours, the observed hours and the onset
proxy (challenge label start + 6 h). Windows are prespecified:

* early window ``[onset-12, onset)``      -> early event sensitivity, lead time
* warning window ``[onset-12, onset-6]``  -> at-least-six-hour warning
* match window ``[onset-12, onset+3]``    -> one-to-one alert/onset matching
  (pre-onset and late matches reported separately)

Left-censored positive records (positive from their first row) have no
usable onset; their alerts are counted separately and they are excluded from
lead-time summaries. Patient-days = observed monitoring hours / 24.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARLY = (-12, 0)  # [onset-12, onset)
WARNING = (-12, -6)  # [onset-12, onset-6]
MATCH = (-12, 3)  # [onset-12, onset+3]


@dataclass
class StayAlerts:
    stay_id: str
    n_hours: int
    alert_hours: list[int]
    onset: float | None
    septic: bool
    left_censored: bool


def evaluate_alerts(stays: list[StayAlerts]) -> dict:
    total_hours = sum(s.n_hours for s in stays)
    nonseptic_hours = sum(s.n_hours for s in stays if not s.septic)
    n_alerts = sum(len(s.alert_hours) for s in stays)
    evaluable = [s for s in stays if s.septic and not s.left_censored and s.onset is not None]
    detected_early, warned6, leads = 0, 0, []
    matched_pre, matched_late = 0, 0
    unmatched = 0
    censored_alerts = 0
    false_nonseptic = 0
    for s in stays:
        a = np.asarray(s.alert_hours, dtype=float)
        if not s.septic:
            false_nonseptic += len(a)
            unmatched += len(a)
            continue
        if s.left_censored or s.onset is None:
            censored_alerts += len(a)
            continue
        o = s.onset
        early = a[(a >= o + EARLY[0]) & (a < o + EARLY[1])]
        if len(early):
            detected_early += 1
            leads.append(o - early.min())
        if np.any((a >= o + WARNING[0]) & (a <= o + WARNING[1])):
            warned6 += 1
        in_match = a[(a >= o + MATCH[0]) & (a <= o + MATCH[1])]
        if len(in_match):  # one-to-one: the first alert in the window matches the onset
            first = in_match.min()
            if first < o:
                matched_pre += 1
            else:
                matched_late += 1
            unmatched += len(a) - 1
        else:
            unmatched += len(a)
    n_eval = len(evaluable)
    pd_all = total_hours / 24.0
    pd_non = nonseptic_hours / 24.0
    matched = matched_pre + matched_late
    precision_den = n_alerts - censored_alerts
    return {
        "stays": len(stays),
        "septic_onset_evaluable_stays": n_eval,
        "left_censored_positive_stays": sum(1 for s in stays if s.septic and s.left_censored),
        "patient_days": pd_all,
        "alerts_emitted": n_alerts,
        "alerts_per_100_patient_days": 100 * n_alerts / pd_all if pd_all else float("nan"),
        "unmatched_alerts": unmatched,
        "unmatched_per_100_patient_days": 100 * unmatched / pd_all if pd_all else float("nan"),
        "false_alerts_nonseptic": false_nonseptic,
        "false_alerts_nonseptic_per_100_patient_days":
            100 * false_nonseptic / pd_non if pd_non else float("nan"),
        "alerts_in_left_censored_stays": censored_alerts,
        "early_sensitivity": detected_early / n_eval if n_eval else float("nan"),
        "detected_early": detected_early,
        "missed": n_eval - detected_early,
        "six_hour_warning_rate": warned6 / n_eval if n_eval else float("nan"),
        "lead_time_median_h": float(np.median(leads)) if leads else float("nan"),
        "lead_time_iqr_h": [float(np.percentile(leads, 25)), float(np.percentile(leads, 75))] if leads else None,
        "matched_pre_onset": matched_pre,
        "matched_late": matched_late,
        "alert_precision": matched / precision_den if precision_den else float("nan"),
        "alert_precision_pre_onset": matched_pre / precision_den if precision_den else float("nan"),
    }
