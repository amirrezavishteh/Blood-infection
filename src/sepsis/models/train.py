"""Training: regularised logistic baseline and LightGBM candidate.

* fit on ``a_train`` only (preprocessing included);
* LightGBM tuning uses a small predefined grid with early stopping on
  ``a_validation``; class weighting is a grid option and only affects
  training — evaluation always uses natural prevalence;
* the candidate is chosen by validation AUPRC (prespecified);
* calibration is a separate step on ``a_calibration`` (see ``calibrate``).
"""

from __future__ import annotations

import itertools
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis.data.store import Store
from sepsis.features.causal import FeatureConfig, feature_names
from sepsis.labels.targets import CHALLENGE_TARGET
from sepsis.models.bundle import bundle_root, load_bundle, save_bundle, write_meta
from sepsis.models.preprocess import fit_platt

log = logging.getLogger(__name__)


def code_revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=Path(__file__).parent, timeout=10)
        if rev.returncode == 0:
            return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        pass
    return "unknown"


def new_run_id(kind: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{kind}"


def _fit_logistic(X, y, names, params):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sepsis.models.preprocess import DropAllMissing, QuantileClipper

    pipe = Pipeline([
        ("drop", DropAllMissing(feature_names=names)),
        ("clip", QuantileClipper(params.get("clip_lower", 0.005), params.get("clip_upper", 0.995))),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=params.get("C", 0.1), class_weight=params.get("class_weight"),
                                   max_iter=params.get("max_iter", 2000), solver="lbfgs")),
    ])
    pipe.fit(X, y)
    return pipe


def _fit_lightgbm(X, y, Xv, yv, names, params, seed):
    import lightgbm as lgb

    keep = ~np.all(np.isnan(X), axis=0)  # train-only rule: drop features absent in training
    dropped = [n for n, k in zip(names, keep) if not k]
    base = {
        "objective": "binary", "learning_rate": params.get("learning_rate", 0.05),
        "num_leaves": params["num_leaves"], "min_child_samples": params["min_child_samples"],
        "feature_fraction": params.get("feature_fraction", 0.8),
        "bagging_fraction": params.get("bagging_fraction", 0.8), "bagging_freq": 1,
        "lambda_l2": params.get("lambda_l2", 1.0), "seed": seed, "deterministic": True,
        "num_threads": params.get("num_threads", 0), "verbose": -1,
        # Early stopping watches validation average precision: log-loss is distorted by class
        # weighting (weighted runs stopped after one round) and is not the selection metric.
        "metric": params.get("early_stopping_metric", "average_precision"), "first_metric_only": True,
    }
    if params.get("class_weight") == "balanced":
        base["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
    Xt = X.copy()
    Xt[:, ~keep] = np.nan
    dtrain = lgb.Dataset(Xt, y, feature_name=[_safe(n) for n in names], free_raw_data=True)
    dval = lgb.Dataset(Xv, yv, reference=dtrain)
    booster = lgb.train(base, dtrain, num_boost_round=params.get("max_rounds", 2000),
                        valid_sets=[dval], callbacks=[lgb.early_stopping(params.get("early_stopping", 100),
                                                                         verbose=False)])
    return booster, base, dropped


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def train(cfg: dict, store: Store | None = None, run_id: str | None = None) -> Path:
    """Train per ``cfg`` (see configs/*.yaml); returns the run directory."""
    store = store or Store.for_dataset(cfg.get("dataset", "physionet2019"))
    fcfg = FeatureConfig.from_dict(cfg.get("features", {}))
    names = feature_names(fcfg)
    kind = cfg["model"]["kind"]
    seed = int(cfg.get("seed", 2019))
    t0 = time.time()
    X, y, _ = store.matrix(fcfg, "a_train", names)
    Xv, yv, _ = store.matrix(fcfg, "a_validation", names)
    log.info("train rows=%d pos=%.4f | val rows=%d", len(y), y.mean(), len(yv))

    trials = []
    if kind == "logistic":
        grid = cfg["model"].get("grid", {"C": [0.1], "class_weight": [None]})
        best = None
        for combo in _expand(grid):
            params = {**cfg["model"].get("params", {}), **combo}
            m = _fit_logistic(X, y, names, params)
            pv = m.predict_proba(Xv)[:, 1]
            res = {"params": params, "val_auprc": float(average_precision_score(yv, pv)),
                   "val_auroc": float(roc_auc_score(yv, pv))}
            trials.append(res)
            log.info("logistic %s -> AUPRC %.4f", combo, res["val_auprc"])
            if best is None or res["val_auprc"] > best[0]["val_auprc"]:
                best = (res, m)
        res, model = best
        extra = {"dropped_features": model.named_steps["drop"].dropped_}
    elif kind == "lightgbm":
        grid = cfg["model"].get("grid", {"num_leaves": [31], "min_child_samples": [100]})
        best = None
        for combo in _expand(grid):
            params = {**cfg["model"].get("params", {}), **combo}
            booster, used, dropped = _fit_lightgbm(X, y, Xv, yv, names, params, seed)
            pv = booster.predict(Xv, num_iteration=booster.best_iteration)
            res = {"params": params, "best_iteration": int(booster.best_iteration),
                   "val_auprc": float(average_precision_score(yv, pv)),
                   "val_auroc": float(roc_auc_score(yv, pv))}
            trials.append(res)
            log.info("lightgbm %s -> AUPRC %.4f (iter %d)", combo, res["val_auprc"], res["best_iteration"])
            if best is None or res["val_auprc"] > best[0]["val_auprc"]:
                best = (res, booster, dropped)
        res, model, dropped = best
        extra = {"dropped_features": dropped, "best_iteration": res["best_iteration"]}
    else:
        raise ValueError(f"unknown model kind {kind}")

    run_id = run_id or new_run_id(kind)
    run_dir = bundle_root() / run_id
    meta = {
        "model_version": run_id,
        "model_kind": kind,
        "target_id": cfg.get("target_id", CHALLENGE_TARGET),
        "feature_config": fcfg.to_dict(),
        "feature_names": names,
        "selected_params": res["params"],
        "selection_metric": "a_validation AUPRC",
        "tuning_trials": trials,
        "train_positive_prevalence": float(y.mean()),
        "calibration": None,
        "alert_policy": None,
        "benchmark_threshold": None,
        "provenance": {
            "dataset": cfg.get("dataset", "physionet2019"),
            "dataset_quality_outputs": store.quality.get("outputs"),
            "raw_manifest_sha256": store.quality.get("raw_manifest_sha256"),
            "split_hash": store.splits["split_hash"],
            "code_revision": code_revision(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_rows": int(len(y)), "train_seconds": round(time.time() - t0, 1),
            "config": cfg,
        },
        **extra,
    }
    save_bundle(run_dir, model, meta)
    log.info("saved %s (val AUPRC %.4f)", run_dir, res["val_auprc"])
    return run_dir


def calibrate(run_dir: Path, store: Store | None = None) -> dict:
    """Fit a sigmoid calibrator on ``a_calibration`` and store it in the bundle."""
    b = load_bundle(run_dir)
    store = store or Store.for_dataset(b.meta["provenance"]["dataset"])
    _check_split(b, store)
    Xc, yc, _ = store.matrix(b.feature_config, "a_calibration", b.feature_names)
    raw = b.raw_scores(Xc)
    calib = fit_platt(raw, yc)
    calib.update({"fitted_on": "a_calibration", "rows": int(len(yc)), "positives": int(yc.sum())})
    meta = dict(b.meta)
    meta["calibration"] = calib
    write_meta(b.path, meta)
    return calib


def _check_split(b, store) -> None:
    if b.meta["provenance"]["split_hash"] != store.splits["split_hash"]:
        raise RuntimeError("bundle was trained on a different split; refusing")


def _expand(grid: dict):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def promote(run_dirs: list[Path], out: Path) -> dict:
    """Validation-based selection among calibrated candidate runs."""
    rows = []
    for rd in run_dirs:
        b = load_bundle(rd)
        best = max(t["val_auprc"] for t in b.meta["tuning_trials"])
        rows.append({"run": b.model_version, "kind": b.meta["model_kind"], "val_auprc": best,
                     "feature_set": b.feature_config.feature_set})
    chosen = max(rows, key=lambda r: r["val_auprc"])
    manifest = {"candidates": rows, "selected": chosen["run"], "rule": "max a_validation AUPRC",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(out).write_text(json.dumps(manifest, indent=1))
    return manifest
