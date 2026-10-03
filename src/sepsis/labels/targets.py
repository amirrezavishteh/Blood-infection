"""Versioned prediction targets.

Target A, ``challenge2019_state``: the supplied ``SepsisLabel`` used unchanged.
Its positive state begins six hours before the challenge-defined onset and
persists afterwards, so it is *not* an incident-risk label. It is never
shifted again here.

Onset proxy (analysis convention only): for an observed 0->1 transition at
hour index k, the inferred onset is k + 6. Records positive from their first
row are left-censored: onset unknown, excluded from lead-time summaries,
retained in the official benchmark.

Target B, ``incident_sepsis_6h``: a horizon label for richer datasets.
y(t)=1 if the first eligible onset lies in (t, t+H]; y(t)=0 if the whole
horizon is observed without onset; otherwise censored (None). Hours at or
after onset are not at risk; stays already septic at their first hour are
excluded as prevalent cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHALLENGE_TARGET = "challenge2019_state"
INCIDENT_TARGET = "incident_sepsis_6h"
CHALLENGE_SHIFT_H = 6  # the supplied label already leads onset by this many hours


@dataclass(frozen=True)
class OnsetInfo:
    onset_hour: float | None  # inferred onset (hour index); None if unknown/absent
    label_start: int | None  # first positive hour index of the supplied label
    left_censored: bool
    septic: bool

    @property
    def evaluable(self) -> bool:
        """Usable for precise lead-time / early-warning summaries."""
        return self.septic and not self.left_censored


def challenge_state_labels(labels: np.ndarray) -> np.ndarray:
    """Target A: the supplied label, returned unchanged (no extra shift)."""
    return np.asarray(labels, dtype=np.int8).copy()


def onset_from_challenge_labels(labels: np.ndarray) -> OnsetInfo:
    lab = np.asarray(labels)
    if not lab.any():
        return OnsetInfo(None, None, False, False)
    start = int(np.argmax(lab))
    if start == 0:
        return OnsetInfo(None, 0, True, True)
    return OnsetInfo(float(start + CHALLENGE_SHIFT_H), start, False, True)


def incident_label(t: int, onset: float | None, last_observed: int, horizon: int = 6):
    """Label for a single at-risk hour ``t``.

    ``onset``: first eligible onset time (hour index) or None if none observed.
    ``last_observed``: last hour index with follow-up (end of observation).
    Returns 1, 0, or None (censored); raises if ``t`` is not at risk.
    """
    if onset is not None and onset <= t:
        raise ValueError("hour is not at risk (onset at or before t)")
    if onset is not None and onset <= t + horizon:
        return 1  # an observed positive inside the horizon counts even if follow-up ends later
    if t + horizon <= last_observed:
        return 0
    return None


def incident_labels_for_stay(n_hours: int, onset: float | None, horizon: int = 6,
                             last_observed: int | None = None):
    """Vectorised Target B for one stay.

    Returns (labels, at_risk, status) where labels is float (nan = censored /
    not at risk), at_risk is bool per hour and status is 'prevalent_excluded'
    when the stay is already septic at its first hour.
    """
    last_observed = n_hours - 1 if last_observed is None else last_observed
    labels = np.full(n_hours, np.nan)
    at_risk = np.ones(n_hours, dtype=bool)
    if onset is not None and onset <= 0:
        return labels, np.zeros(n_hours, dtype=bool), "prevalent_excluded"
    for t in range(n_hours):
        if onset is not None and onset <= t:
            at_risk[t:] = False
            break
        y = incident_label(t, onset, last_observed, horizon)
        labels[t] = np.nan if y is None else y
    return labels, at_risk, "included"
