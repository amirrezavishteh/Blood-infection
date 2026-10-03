"""Alert state machine, batch/fast equivalence and alert-matching metrics."""

import numpy as np
import pytest

from sepsis.alerts.matching import StayAlerts, evaluate_alerts
from sepsis.alerts.policy import AlertPolicy, acknowledge, initial_state, run_policy, step
from sepsis.alerts.search import fast_alert_hours


def test_threshold_consecutive_and_rearm():
    p = AlertPolicy(0.5, consecutive=2)
    scores = [0.6, 0.7, 0.8, 0.2, 0.9, 0.9]
    assert run_policy(p, range(6), scores, ["scored"] * 6) == [1, 5]


def test_cooldown_suppresses_realerts():
    p = AlertPolicy(0.5, consecutive=1, cooldown_h=6)
    scores = [0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9]
    assert run_policy(p, range(8), scores, ["scored"] * 8) == [0, 7]


def test_missing_data_is_not_reassuring_and_resets_counter():
    p = AlertPolicy(0.5, consecutive=2)
    st = initial_state()
    st, e = step(p, st, 0, 0.9, "scored")
    st, e = step(p, st, 1, None, "data_unavailable")
    assert st["state"] == "data_unavailable" and not e
    st, e = step(p, st, 2, 0.9, "scored")
    assert not e  # counter was reset by missing data
    st, e = step(p, st, 3, 0.9, "scored")
    assert e


def test_acknowledge_does_not_change_alert_timing():
    p = AlertPolicy(0.5)
    st, e = step(p, initial_state(), 0, 0.9, "scored")
    assert e and st["state"] == "alerted"
    st = acknowledge(st)
    assert st["state"] == "acknowledged"
    st, e = step(p, st, 1, 0.9, "scored")
    assert not e
    st, e = step(p, st, 2, 0.1, "scored")
    st, e = step(p, st, 3, 0.9, "scored")
    assert e


def test_policy_version_is_stable_and_distinct():
    assert AlertPolicy(0.5, 2, 6).version == AlertPolicy(0.5, 2, 6).version
    assert AlertPolicy(0.5, 2, 6).version != AlertPolicy(0.5, 2, 12).version
    with pytest.raises(ValueError):
        AlertPolicy(1.5)


def test_fast_search_loop_equals_state_machine():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        scores = rng.random(n)
        ok = rng.random(n) > 0.15
        statuses = np.where(ok, "scored", "data_unavailable")
        p = AlertPolicy(float(rng.random()), int(rng.integers(1, 4)), int(rng.choice([0, 3, 6, 12])))
        sc = np.where(ok, scores, np.nan)
        assert run_policy(p, range(n), sc, statuses) == fast_alert_hours(p, np.arange(n), sc, ok)


def test_alert_matching_metrics():
    stays = [
        StayAlerts("s1", 48, [20, 30], onset=30.0, septic=True, left_censored=False),  # early (lead 10) + late dup
        StayAlerts("s2", 48, [], onset=25.0, septic=True, left_censored=False),  # missed
        StayAlerts("s3", 48, [32], onset=30.0, septic=True, left_censored=False),  # late match only
        StayAlerts("s4", 96, [5, 50], onset=None, septic=False, left_censored=False),  # 2 false alerts
        StayAlerts("s5", 24, [1], onset=None, septic=True, left_censored=True),  # censored
    ]
    m = evaluate_alerts(stays)
    assert m["septic_onset_evaluable_stays"] == 3
    assert m["detected_early"] == 1 and m["missed"] == 2
    assert m["early_sensitivity"] == pytest.approx(1 / 3)
    assert m["six_hour_warning_rate"] == pytest.approx(1 / 3)  # alert at 20 in [18, 24]
    assert m["lead_time_median_h"] == 10
    assert m["matched_pre_onset"] == 1 and m["matched_late"] == 1
    assert m["false_alerts_nonseptic"] == 2
    assert m["unmatched_alerts"] == 1 + 2  # duplicate in s1 + two false alerts
    assert m["alerts_in_left_censored_stays"] == 1
    assert m["alert_precision"] == pytest.approx(2 / 5)
    assert m["patient_days"] == pytest.approx((48 * 3 + 96 + 24) / 24)
    assert m["alerts_per_100_patient_days"] == pytest.approx(100 * 6 / m["patient_days"])
