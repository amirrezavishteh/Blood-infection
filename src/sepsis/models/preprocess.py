"""Train-partition-only preprocessing.

Every statistic here (dropped all-missing features, clipping quantiles,
imputation medians, scaling) is fitted on the training partition only and
then applied unchanged to validation, calibration, test and replay data.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropAllMissing(BaseEstimator, TransformerMixin):
    """Drop features that are entirely missing in the training data.

    A feature absent in training cannot be rescued by a test-set median; the
    decision is recorded in ``dropped_`` and reproduced at inference time.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.keep_ = ~np.all(np.isnan(X), axis=0)
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.kept_names_ = [n for n, k in zip(names, self.keep_) if k]
        self.dropped_ = [n for n, k in zip(names, self.keep_) if not k]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.keep_]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature to training-set quantiles; NaN is preserved."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):
            self.lo_ = np.nanquantile(X, self.lower, axis=0)
            self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lo_, self.hi_)


def fit_platt(raw_scores: np.ndarray, y: np.ndarray) -> dict:
    """Sigmoid calibration on the logit of the raw model score."""
    from sklearn.linear_model import LogisticRegression

    z = _logit(raw_scores).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z, y)
    return {"method": "sigmoid", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_calibration(raw_scores: np.ndarray, calib: dict | None) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if not calib:
        return raw_scores
    if calib["method"] != "sigmoid":
        raise ValueError(f"unsupported calibration {calib['method']}")
    return 1.0 / (1.0 + np.exp(-(calib["a"] * _logit(raw_scores) + calib["b"])))


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))
