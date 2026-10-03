"""Shared fixtures: a small synthetic corpus processed and trained in temp dirs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ws")
    os.environ["SEPSIS_DATA_DIR"] = str(root / "data")
    os.environ["SEPSIS_ARTIFACT_DIR"] = str(root / "artifacts")
    return root


@pytest.fixture(scope="session")
def corpus(workspace):
    """Synthetic raw corpus -> parquet store -> splits."""
    from sepsis.data.prepare import prepare
    from sepsis.data.splits import make_splits, save_splits
    from sepsis.data.store import Store
    from sepsis.data.synthetic import make_fixture
    from sepsis.paths import raw_dir

    make_fixture(raw_dir("physionet2019"), n_a=160, n_b=60, seed=11)
    prepare("physionet2019", workers=1, require_expected_counts=False)
    store = Store.for_dataset("physionet2019")
    save_splits(make_splits(store.encounters, seed=2019), store.splits_path)
    return Store.for_dataset("physionet2019")


SMALL_LGBM = {
    "dataset": "physionet2019",
    "target_id": "challenge2019_state",
    "seed": 1,
    "features": {"feature_set": "basic"},
    "model": {"kind": "lightgbm", "params": {"max_rounds": 200, "early_stopping": 20, "num_threads": 2},
              "grid": {"num_leaves": [15], "min_child_samples": [20]}},
}


@pytest.fixture(scope="session")
def trained_run(corpus):
    """A calibrated LightGBM run with a frozen policy (small grid)."""
    from sepsis.evaluation.pipeline import select_policy
    from sepsis.models.train import calibrate, train

    run_dir = train(SMALL_LGBM, store=corpus, run_id="test-lgbm")
    calibrate(run_dir, store=corpus)
    select_policy(run_dir, store=corpus, budgets=(5, 20, 50), primary_budget=50,
                  consecutive=(1, 2), cooldowns=(0, 6))
    return run_dir
