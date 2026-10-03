"""Versioned model bundles: save, verify, load, score, explain.

A bundle directory holds ``model.pkl`` plus ``bundle.json`` with the feature
order and dtypes, feature config + schema version, target id, preprocessing
settings, calibrator, training provenance, benchmark threshold and the
selected alert policy. Loading refuses bundles outside the trusted artifact
directory, checksum mismatches and incompatible feature schemas.
"""

from __future__ import annotations

import json
import pickle
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.data.schema import VITALS, sha256_file
from sepsis.features.causal import FeatureConfig, feature_names, schema_version
from sepsis.models.preprocess import apply_calibration
from sepsis.paths import artifact_dir, ensure_within

BUNDLE_FORMAT = 1


class BundleError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    path: Path
    meta: dict
    model: object = field(repr=False)

    # ---- identity -------------------------------------------------------
    @property
    def model_version(self) -> str:
        return self.meta["model_version"]

    @property
    def target_id(self) -> str:
        return self.meta["target_id"]

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig.from_dict(self.meta["feature_config"])

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta["feature_names"])

    @property
    def feature_schema_version(self) -> str:
        return self.meta["feature_schema_version"]

    @property
    def calibration(self) -> dict | None:
        return self.meta.get("calibration")

    @property
    def policy(self) -> dict | None:
        return self.meta.get("alert_policy")

    # ---- scoring --------------------------------------------------------
    def check_schema(self, names: list[str]) -> None:
        if list(names) != self.feature_names:
            raise BundleError("feature schema mismatch: refusing to score")

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        kind = self.meta["model_kind"]
        if kind == "lightgbm_bag":
            return self.model.predict(X)
        if kind == "lightgbm":
            return self.model.predict(X, num_iteration=self.meta.get("best_iteration"))
        return self.model.predict_proba(X)[:, 1]

    def scores(self, X: np.ndarray) -> np.ndarray:
        return apply_calibration(self.raw_scores(X), self.calibration)

    def quality_status(self, feats: pd.DataFrame) -> np.ndarray:
        """'scored' if at least one vital sign is valid (within expiry), else 'data_unavailable'."""
        cols = [f"{v}__last" for v in VITALS]
        ok = feats[cols].notna().any(axis=1).to_numpy()
        return np.where(ok, "scored", "data_unavailable")

    def score_frame(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Score a feature frame; unavailable rows get score NaN and status."""
        self.check_schema([c for c in feats.columns if c not in ("stay_id", "hour_index")])
        X = feats[self.feature_names].to_numpy(dtype=np.float32)
        s = self.scores(X) if len(X) else np.array([])
        status = self.quality_status(feats)
        s = np.where(status == "scored", s, np.nan)
        return pd.DataFrame({"stay_id": feats["stay_id"].to_numpy(),
                             "hour_index": feats["hour_index"].to_numpy(),
                             "score": s, "status": status})

    # ---- explanation ----------------------------------------------------
    def contributions(self, x_row: np.ndarray, top_k: int = 8) -> list[dict]:
        """Deterministic per-feature contributions (log-odds scale, uncalibrated).

        These describe associations the model uses; they are not causal
        explanations or treatment advice.
        """
        x_row = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
        names = self.feature_names
        if self.meta["model_kind"] == "lightgbm_bag":
            pairs = list(zip(names, self.model.contributions(x_row)[0][:-1]))
        elif self.meta["model_kind"] == "lightgbm":
            contrib = self.model.predict(x_row, num_iteration=self.meta.get("best_iteration"),
                                         pred_contrib=True)[0][:-1]
            pairs = list(zip(names, contrib))
        else:
            pipe = self.model
            drop = pipe.named_steps["drop"]
            Z = x_row.astype(np.float64)
            for name, step in pipe.steps[:-1]:
                Z = step.transform(Z)
            coef = pipe.named_steps["clf"].coef_[0]
            kept = drop.kept_names_
            imp = pipe.named_steps["impute"]
            ind_names = [f"{kept[i]} (missing)" for i in imp.indicator_.features_] if imp.add_indicator else []
            pairs = list(zip(kept + ind_names, (coef * Z[0])))
        pairs.sort(key=lambda p: -abs(p[1]))
        out = []
        for n, c in pairs[:top_k]:
            base = n.replace(" (missing)", "")
            val = float(x_row[0, names.index(base)]) if base in names else None
            out.append({"feature": n, "contribution": float(c),
                        "value": None if val is None or np.isnan(val) else val,
                        "missing": val is None or bool(np.isnan(val)) if base in names else None})
        return out


def bundle_root() -> Path:
    return artifact_dir() / "runs"


def save_bundle(run_dir: Path, model, meta: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = FeatureConfig.from_dict(meta["feature_config"])
    names = feature_names(cfg)
    if meta["feature_names"] != names:
        raise BundleError("feature names do not match the feature config")
    meta = dict(meta)
    meta.setdefault("bundle_format", BUNDLE_FORMAT)
    meta["feature_schema_version"] = schema_version(cfg)
    meta["feature_dtypes"] = {n: "float32" for n in names}
    meta.setdefault("environment", {"python": platform.python_version(), **_lib_versions()})
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta["model_sha256"] = sha256_file(run_dir / "model.pkl")
    write_meta(run_dir, meta)
    return run_dir


def write_meta(run_dir: Path, meta: dict) -> None:
    (Path(run_dir) / "bundle.json").write_text(json.dumps(meta, indent=1, sort_keys=True))


def load_bundle(run_dir: Path, trusted_root: Path | None = None) -> ModelBundle:
    trusted_root = trusted_root or artifact_dir()
    run_dir = ensure_within(Path(run_dir), trusted_root)
    meta_path = run_dir / "bundle.json"
    if not meta_path.exists():
        raise BundleError(f"no bundle at {run_dir}")
    meta = json.loads(meta_path.read_text())
    if meta.get("bundle_format") != BUNDLE_FORMAT:
        raise BundleError("unsupported bundle format")
    cfg = FeatureConfig.from_dict(meta["feature_config"])
    if schema_version(cfg) != meta["feature_schema_version"] or feature_names(cfg) != meta["feature_names"]:
        raise BundleError("bundle feature schema is inconsistent with this code version")
    if sha256_file(run_dir / "model.pkl") != meta["model_sha256"]:
        raise BundleError("model.pkl checksum mismatch")
    with open(run_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)  # only from the trusted, checksummed artifact directory
    return ModelBundle(run_dir, meta, model)


def _lib_versions() -> dict:
    import lightgbm
    import sklearn

    return {"numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__}
