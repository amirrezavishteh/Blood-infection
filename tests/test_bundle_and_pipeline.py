"""Model bundles, train/serve agreement and the frozen evaluation pipeline."""

import json
import shutil

import numpy as np
import pytest

from sepsis.features.causal import compute_features
from sepsis.models.bundle import BundleError, load_bundle


def test_bundle_contents(trained_run):
    b = load_bundle(trained_run)
    m = b.meta
    for key in ("model_version", "target_id", "feature_names", "feature_dtypes", "feature_schema_version",
                "calibration", "alert_policy", "benchmark_threshold", "provenance", "model_sha256"):
        assert m.get(key) is not None, key
    assert m["target_id"] == "challenge2019_state"
    assert m["calibration"]["fitted_on"] == "a_calibration"
    assert m["alert_policy"]["selected_on"] == "a_validation"


def test_bundle_refuses_tampering_and_untrusted_paths(trained_run, tmp_path, workspace):
    with pytest.raises(PermissionError):
        load_bundle(tmp_path)  # outside the trusted artifact directory
    copy = trained_run.parent / "tampered"
    shutil.copytree(trained_run, copy)
    with open(copy / "model.pkl", "ab") as f:
        f.write(b"x")
    with pytest.raises(BundleError, match="checksum"):
        load_bundle(copy)
    shutil.rmtree(copy)
    copy2 = trained_run.parent / "badschema"
    shutil.copytree(trained_run, copy2)
    meta = json.loads((copy2 / "bundle.json").read_text())
    meta["feature_names"] = meta["feature_names"][::-1]
    (copy2 / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(BundleError, match="schema"):
        load_bundle(copy2)
    shutil.rmtree(copy2)


def test_bundle_refuses_incompatible_feature_frame(trained_run, corpus):
    b = load_bundle(trained_run)
    feats = corpus.features(b.feature_config).head(5)
    with pytest.raises(BundleError):
        b.score_frame(feats.drop(columns=[b.feature_names[0]]))


def test_replay_prefix_scores_equal_batch(trained_run, corpus):
    """Train/serve drift: hour-by-hour scoring on growing history == batch inference."""
    b = load_bundle(trained_run)
    sid = corpus.stays("a_test")[0]
    hist = corpus.hourly[corpus.hourly.stay_id == sid].drop(columns=["hospital", "record"])
    batch = b.score_frame(compute_features(hist, b.feature_config))
    for t in range(len(hist)):
        online = b.score_frame(compute_features(hist.iloc[: t + 1], b.feature_config)).iloc[-1]
        assert online["status"] == batch.iloc[t]["status"]
        if online["status"] == "scored":
            assert online["score"] == pytest.approx(batch.iloc[t]["score"], abs=1e-9)
        else:
            assert np.isnan(online["score"])


def test_unavailable_input_gets_explicit_status(trained_run, corpus):
    b = load_bundle(trained_run)
    sid = corpus.stays("a_test")[0]
    hist = corpus.hourly[corpus.hourly.stay_id == sid].drop(columns=["hospital", "record"]).copy()
    from sepsis.data.schema import VITALS

    hist[VITALS] = np.nan
    s = b.score_frame(compute_features(hist, b.feature_config))
    assert (s["status"] == "data_unavailable").all() and s["score"].isna().all()


def test_contributions_are_deterministic(trained_run, corpus):
    b = load_bundle(trained_run)
    X, _, _ = corpus.matrix(b.feature_config, "a_test", b.feature_names)
    c1, c2 = b.contributions(X[10]), b.contributions(X[10])
    assert c1 == c2 and len(c1) == 8
    assert all(set(c) >= {"feature", "contribution", "value", "missing"} for c in c1)


def test_evaluation_requires_frozen_run_and_valid_split(trained_run, corpus):
    from sepsis.evaluation.pipeline import evaluate, select_policy

    with pytest.raises(ValueError):
        evaluate(trained_run, "a_train", store=corpus)
    with pytest.raises(ValueError):
        select_policy(trained_run, store=corpus, split="a_test")


def test_evaluate_held_out_splits(trained_run, corpus):
    from sepsis.evaluation.pipeline import evaluate

    for split in ("a_test", "b_external"):
        r = evaluate(trained_run, split, store=corpus, n_boot=20)
        assert r["records"] == len(corpus.stays(split))
        assert abs(r["official"]["Utility"] - r["benchmark"]["utility"]) < 1e-9
        assert 0 <= r["coverage"]["coverage"] <= 1
        assert (trained_run / "eval" / split / "report.md").exists()
    log = json.loads((trained_run / "evaluation_log.json").read_text())
    assert [e["split"] for e in log].count("b_external") >= 1


def test_seed_bag_bundle_scores_and_explains(corpus):
    """A seed-bagged LightGBM bundle saves, loads, scores and gives additive contributions."""
    import copy

    from conftest import SMALL_LGBM
    from sepsis.models.train import calibrate, train

    cfg = copy.deepcopy(SMALL_LGBM)
    cfg["model"]["n_seeds"] = 3
    run_dir = train(cfg, store=corpus, run_id="test-bag")
    calibrate(run_dir, store=corpus)
    b = load_bundle(run_dir)
    assert b.meta["model_kind"] == "lightgbm_bag" and b.meta["bag"]["n_seeds"] == 3
    X, _, _ = corpus.matrix(b.feature_config, "a_test", b.feature_names)
    raw = b.raw_scores(X[:50])
    contrib = b.model.contributions(X[:50])
    margin = np.log(raw / (1 - raw))
    np.testing.assert_allclose(contrib.sum(axis=1), margin, atol=1e-6)  # SHAP sums to the bag margin
    assert len(b.contributions(X[0])) == 8
