"""Look-ahead leakage and causal-feature invariants."""

import numpy as np
import pandas as pd
import pytest

from sepsis.data.schema import ADMINISTRATIVE, CLINICAL, DEMOGRAPHICS
from sepsis.features.causal import FeatureConfig, compute_features, feature_names, features_at

COLS = CLINICAL + DEMOGRAPHICS + ADMINISTRATIVE


def _stay(n=30, seed=0, sid="physionet2019:A:p1"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(np.where(rng.random((n, len(COLS))) < 0.4, np.nan,
                               rng.normal(50, 10, (n, len(COLS)))), columns=COLS)
    df["ICULOS"] = np.arange(1, n + 1, dtype=float)
    df.insert(0, "hour_index", np.arange(n))
    df.insert(0, "stay_id", sid)
    return df


@pytest.mark.parametrize("fs", ["basic", "extended"])
def test_future_rows_never_change_earlier_features(fs):
    cfg = FeatureConfig(feature_set=fs, include_admin=True)
    df = _stay(40)
    base = compute_features(df, cfg)
    rng = np.random.default_rng(5)
    for cut in (0, 5, 17, 38):
        # perturb every future value, and separately remove future rows entirely
        perturbed = df.copy()
        fut = perturbed["hour_index"] > cut
        perturbed.loc[fut, COLS] = rng.normal(500, 100, (fut.sum(), len(COLS)))
        removed = df[df["hour_index"] <= cut]
        for variant in (perturbed, removed):
            out = compute_features(variant, cfg)
            a = out[out["hour_index"] <= cut].drop(columns="stay_id").to_numpy()
            b = base[base["hour_index"] <= cut].drop(columns="stay_id").to_numpy()
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(a, b, equal_nan=True)


def test_online_features_match_batch():
    cfg = FeatureConfig(feature_set="extended")
    df = _stay(25, seed=3)
    batch = compute_features(df, cfg).drop(columns="stay_id").to_numpy()
    for t in range(25):
        online = features_at(df.iloc[: t + 1], cfg).drop("stay_id").to_numpy(dtype=float)
        np.testing.assert_allclose(online, batch[t], equal_nan=True)


def test_other_stays_do_not_leak_and_chunking_is_exact():
    cfg = FeatureConfig(feature_set="extended")
    a, b = _stay(20, 1, "s:A:1"), _stay(15, 2, "s:A:2")
    together = compute_features(pd.concat([a, b]), cfg)
    alone = compute_features(b, cfg)
    np.testing.assert_allclose(together[together.stay_id == "s:A:2"].drop(columns="stay_id").to_numpy(),
                               alone.drop(columns="stay_id").to_numpy(), equal_nan=True)


def test_no_backfill_and_expiry():
    cfg = FeatureConfig(vital_expiry_h=3)
    df = _stay(12)
    df["HR"] = np.nan
    df.loc[4, "HR"] = 99.0
    f = compute_features(df, cfg).set_index("hour_index")
    assert f.loc[0:3, "HR__last"].isna().all()  # nothing before the first observation
    assert (f.loc[4:7, "HR__last"] == 99).all()  # carried forward within expiry
    assert f.loc[8:, "HR__last"].isna().all()  # expired after 3 hours


def test_slope_and_std_need_two_observations():
    df = _stay(6)
    df["HR"] = [np.nan, 80, np.nan, np.nan, 86, np.nan]
    f = compute_features(df, FeatureConfig()).set_index("hour_index")
    assert np.isnan(f.loc[1, "HR__slope_3h"]) and np.isnan(f.loc[1, "HR__std_3h"])
    assert f.loc[4, "HR__slope_6h"] == pytest.approx(2.0)  # (86-80)/3h
    assert np.isnan(f.loc[4, "HR__slope_3h"])  # only one obs within the last 3 rows


def test_labels_and_ids_rejected_and_excluded():
    df = _stay(5)
    with pytest.raises(ValueError):
        compute_features(df.assign(SepsisLabel=0))
    names = feature_names(FeatureConfig(feature_set="extended"))
    for bad in ("SepsisLabel", "label", "stay_id", "hospital", "record", "Unit1", "Unit2", "HospAdmTime", "ICULOS"):
        assert bad not in names


def test_noncontiguous_hours_rejected():
    df = _stay(5).drop(index=2)
    with pytest.raises(ValueError, match="contiguous"):
        compute_features(df)
