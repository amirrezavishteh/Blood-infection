"""Seed-bagged LightGBM: several boosters with the same parameters and different seeds.

Predictions average the boosters' raw margins (log-odds) and then apply the
sigmoid, so per-feature contributions (TreeSHAP, also in log-odds) average
exactly to the bag's margin. Each booster keeps its own early-stopped
iteration count chosen on validation data.
"""

from __future__ import annotations

import numpy as np


class LGBMBag:
    def __init__(self, boosters: list, best_iterations: list[int]):
        if not boosters or len(boosters) != len(best_iterations):
            raise ValueError("need one best_iteration per booster")
        self.boosters = boosters
        self.best_iterations = [int(i) for i in best_iterations]

    def margin(self, X: np.ndarray) -> np.ndarray:
        return np.mean([b.predict(X, num_iteration=i, raw_score=True)
                        for b, i in zip(self.boosters, self.best_iterations)], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.margin(X)))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Mean TreeSHAP contributions per feature (last column = bias), log-odds scale."""
        return np.mean([b.predict(X, num_iteration=i, pred_contrib=True)
                        for b, i in zip(self.boosters, self.best_iterations)], axis=0)
