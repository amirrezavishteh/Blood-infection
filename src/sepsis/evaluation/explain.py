"""Global model explanation and hospital-transfer diagnostics.

* Global importance: mean absolute contribution per feature (TreeSHAP for
  LightGBM, coefficient x standardised value for logistic regression) on a
  fixed random sample of A-validation hours. Features are grouped into
  families (vital, lab, observation pattern, organ score, demographic, time)
  so reliance on measurement-pattern features is visible.
* Transfer diagnostics: for the most important features, the standardised
  mean difference and the change in missingness between A training hours and
  hospital B hours. Only feature distributions are compared; hospital B
  outcomes are never read here.

These describe associations used by the model, not causal effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sepsis.data.schema import CLINICAL, DEMOGRAPHICS, LABS, VITALS
from sepsis.data.store import Store
from sepsis.features.causal import ORGAN_FEATURES, TIME_FEATURES
from sepsis.models.bundle import load_bundle

PATTERN_SUFFIXES = ("__observed", "__hours_since")


def feature_family(name: str) -> str:
    base = name.split("__")[0]
    if name.endswith(PATTERN_SUFFIXES) or name.startswith(("vitals__count", "labs__count")):
        return "observation pattern"
    if name in ORGAN_FEATURES or name in ("ShockIndex", "PulsePressure"):
        return "derived score"
    if name in TIME_FEATURES or name in ("Unit1", "Unit2"):
        return "time / administrative"
    if name in DEMOGRAPHICS:
        return "demographic"
    if base in VITALS:
        return "vital sign"
    if base in LABS:
        return "lab"
    return "other"


def _abs_contributions(bundle, X: np.ndarray) -> np.ndarray:
    kind = bundle.meta["model_kind"]
    if kind == "lightgbm_bag":
        return np.abs(bundle.model.contributions(X)[:, :-1])
    if kind == "lightgbm":
        return np.abs(bundle.model.predict(X, num_iteration=bundle.meta.get("best_iteration"),
                                           pred_contrib=True)[:, :-1])
    pipe = bundle.model  # logistic: fold indicator columns back onto their feature
    Z = X.astype(np.float64)
    for _, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    contrib = np.abs(Z * pipe.named_steps["clf"].coef_[0])
    drop, imp = pipe.named_steps["drop"], pipe.named_steps["impute"]
    kept = drop.kept_names_
    out = np.zeros((len(X), len(bundle.feature_names)))
    idx = {n: i for i, n in enumerate(bundle.feature_names)}
    for j, n in enumerate(kept):
        out[:, idx[n]] += contrib[:, j]
    for k, fi in enumerate(imp.indicator_.features_ if imp.add_indicator else []):
        out[:, idx[kept[fi]]] += contrib[:, len(kept) + k]
    return out


def explain(run_dir: Path, store: Store | None = None, sample: int = 20000, top: int = 25, seed: int = 0) -> dict:
    b = load_bundle(run_dir)
    store = store or Store.for_dataset(b.meta["provenance"]["dataset"])
    names = b.feature_names
    rng = np.random.default_rng(seed)
    Xv, _, _ = store.matrix(b.feature_config, "a_validation", names)
    Xs = Xv[rng.choice(len(Xv), min(sample, len(Xv)), replace=False)]
    imp = _abs_contributions(b, Xs).mean(axis=0)
    total = float(imp.sum()) or 1.0
    order = np.argsort(-imp)
    families: dict[str, float] = {}
    for n, v in zip(names, imp):
        families[feature_family(n)] = families.get(feature_family(n), 0.0) + float(v) / total

    feats = store.features(b.feature_config)
    a_ids, b_ids = set(store.stays("a_train")), set(store.stays("b_external"))
    fa = feats[feats["stay_id"].isin(a_ids)]
    fb = feats[feats["stay_id"].isin(b_ids)]
    rows = []
    for i in order[:top]:
        n = names[i]
        xa, xb = fa[n].to_numpy(dtype=float), fb[n].to_numpy(dtype=float)
        ma, mb = np.nanmean(xa) if np.isfinite(xa).any() else np.nan, np.nanmean(xb) if np.isfinite(xb).any() else np.nan
        sa, sb = np.nanstd(xa), np.nanstd(xb)
        pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
        rows.append({
            "feature": n, "family": feature_family(n), "importance_share": float(imp[i]) / total,
            "mean_a_train": _f(ma), "mean_b": _f(mb),
            "smd_b_vs_a": _f((mb - ma) / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else None,
            "missing_a_train": float(np.isnan(xa).mean()), "missing_b": float(np.isnan(xb).mean()),
        })
    res = {"run": b.model_version, "sample_rows": int(len(Xs)), "sample_split": "a_validation",
           "family_share": dict(sorted(families.items(), key=lambda kv: -kv[1])), "top_features": rows,
           "note": "mean |contribution| (log-odds) on A-validation hours; shift compares feature "
                   "distributions only (A training vs hospital B), never hospital B outcomes"}
    (Path(run_dir) / "explain.json").write_text(json.dumps(res, indent=1))
    return res


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)
