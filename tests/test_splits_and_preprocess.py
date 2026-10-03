"""Train/test contamination: disjoint splits and train-only fitted statistics."""

import json

import numpy as np
import pandas as pd
import pytest

from sepsis.data.splits import ALL_SPLITS, load_splits, make_splits, save_splits
from sepsis.models.preprocess import DropAllMissing, QuantileClipper, apply_calibration, fit_platt


def _enc(n_a=1000, n_b=300, seed=0):
    rng = np.random.default_rng(seed)
    ids_a = [f"physionet2019:A:p{i:06d}" for i in range(n_a)]
    ids_b = [f"physionet2019:B:p{100000 + i:06d}" for i in range(n_b)]
    return pd.DataFrame({"stay_id": ids_a + ids_b, "hospital_key": ["A"] * n_a + ["B"] * n_b,
                         "ever_positive": rng.random(n_a + n_b) < 0.09})


def test_splits_disjoint_complete_stratified_deterministic(tmp_path):
    enc = _enc()
    s1, s2 = make_splits(enc, seed=3), make_splits(enc, seed=3)
    assert s1["split_hash"] == s2["split_hash"]
    allids = [i for k in ALL_SPLITS for i in s1["partitions"][k]]
    assert len(allids) == len(set(allids)) == len(enc)
    assert set(s1["partitions"]["b_external"]) == set(enc.loc[enc.hospital_key == "B", "stay_id"])
    assert not any(":B:" in i for k in ALL_SPLITS[:-1] for i in s1["partitions"][k])
    pos = set(enc.loc[enc.ever_positive, "stay_id"])
    rates = {k: np.mean([i in pos for i in s1["partitions"][k]]) for k in ALL_SPLITS[:-1]}
    assert max(rates.values()) - min(rates.values()) < 0.03
    assert abs(len(s1["partitions"]["a_train"]) / 1000 - 0.60) < 0.01
    assert make_splits(enc, seed=4)["split_hash"] != s1["split_hash"]


def test_split_file_tamper_detected(tmp_path):
    s = make_splits(_enc())
    p = tmp_path / "s.json"
    save_splits(s, p)
    d = json.loads(p.read_text())
    moved = d["partitions"]["a_test"].pop()
    d["partitions"]["a_train"].append(moved)
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        load_splits(p)


def test_corpus_split_ids_intersections_empty(corpus):
    parts = corpus.splits["partitions"]
    for a in parts:
        for b in parts:
            if a < b:
                assert not set(parts[a]) & set(parts[b])


def test_preprocessing_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (500, 3))
    Xtr[:, 2] = np.nan
    Xte = rng.normal(100, 1, (200, 3))  # wildly different test data
    drop = DropAllMissing(["a", "b", "c"]).fit(Xtr)
    assert drop.dropped_ == ["c"]
    assert drop.transform(Xte).shape[1] == 2  # dropped despite test values being present
    clip = QuantileClipper(0.01, 0.99).fit(drop.transform(Xtr))
    out = clip.transform(drop.transform(Xte))
    assert out.max() <= clip.hi_.max() + 1e-12  # test values clipped to train quantiles
    clip2 = QuantileClipper(0.01, 0.99).fit(drop.transform(Xtr))
    np.testing.assert_allclose(clip.lo_, clip2.lo_)


def test_training_pipeline_fitted_on_train_only(trained_run, corpus):
    """Imputation medians in the bundle equal medians of a_train features only."""
    import pickle

    from sepsis.features.causal import FeatureConfig, feature_names
    from sepsis.models.train import _fit_logistic

    cfg = FeatureConfig()
    names = feature_names(cfg)
    X, y, _ = corpus.matrix(cfg, "a_train", names)
    pipe = _fit_logistic(X, y, names, {"C": 0.1})
    keep = pipe.named_steps["drop"].keep_
    clipped = pipe.named_steps["clip"].transform(X[:, keep])
    np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, np.nanmedian(clipped, axis=0), rtol=1e-6)


def test_platt_calibration_monotone():
    rng = np.random.default_rng(1)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw ** 2).astype(int)
    c = fit_platt(raw, y)
    p = apply_calibration(np.array([0.1, 0.5, 0.9]), c)
    assert np.all(np.diff(p) > 0) and np.all((p > 0) & (p < 1))
