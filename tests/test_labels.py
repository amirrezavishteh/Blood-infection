"""Target definitions: no double shift; onset proxy; incident-horizon boundaries."""

import numpy as np
import pytest

from sepsis.labels.targets import (
    challenge_state_labels,
    incident_label,
    incident_labels_for_stay,
    onset_from_challenge_labels,
)


def test_challenge_labels_are_not_shifted_again(corpus):
    lab = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(challenge_state_labels(lab), lab)
    # stored outcomes equal the raw SepsisLabel column of the source file
    from sepsis.data.schema import read_psv
    from sepsis.paths import raw_dir

    sid = corpus.outcomes.groupby("stay_id")["label"].max().idxmax()
    rec = sid.split(":")[-1]
    raw = read_psv(raw_dir("physionet2019") / "training_setA" / f"{rec}.psv") if ":A:" in sid else \
        read_psv(raw_dir("physionet2019") / "training_setB" / f"{rec}.psv")
    stored = corpus.outcomes[corpus.outcomes.stay_id == sid].sort_values("hour_index")["label"].to_numpy()
    np.testing.assert_array_equal(stored, raw["SepsisLabel"].to_numpy())


def test_onset_proxy():
    o = onset_from_challenge_labels(np.array([0, 0, 0, 1, 1]))
    assert o.label_start == 3 and o.onset_hour == 9 and o.evaluable
    lc = onset_from_challenge_labels(np.array([1, 1, 1]))
    assert lc.left_censored and lc.onset_hour is None and not lc.evaluable
    neg = onset_from_challenge_labels(np.zeros(4))
    assert not neg.septic and neg.onset_hour is None


@pytest.mark.parametrize("t,onset,last,expected", [
    (4, 10, 50, 1),     # onset at t+6: inside (t, t+6]
    (3, 10, 50, 0),     # onset at t+7: just outside, horizon fully observed
    (3, None, 9, 0),    # horizon (3, 9] fully observed, no onset
    (4, None, 9, None),  # discharge before horizon end -> censored
    (4, 6, 5, 1),       # positive observed before censoring still counts
])
def test_incident_boundaries(t, onset, last, expected):
    assert incident_label(t, onset, last) == expected


def test_incident_not_at_risk_at_onset():
    with pytest.raises(ValueError):
        incident_label(10, 10, 50)


def test_incident_stay_vector_and_prevalent_exclusion():
    labels, at_risk, status = incident_labels_for_stay(12, onset=8)
    assert status == "included"
    assert not at_risk[8:].any() and at_risk[:8].all()
    np.testing.assert_array_equal(labels[:8], [0, 0, 1, 1, 1, 1, 1, 1])
    assert np.isnan(labels[8:]).all()
    _, at_risk, status = incident_labels_for_stay(12, onset=0)
    assert status == "prevalent_excluded" and not at_risk.any()
    labels, _, _ = incident_labels_for_stay(10, onset=None)
    np.testing.assert_array_equal(labels[:4], [0, 0, 0, 0])
    assert np.isnan(labels[4:]).all()  # last 6 hours have an unobserved horizon
