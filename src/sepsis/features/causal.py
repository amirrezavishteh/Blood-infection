"""Shared causal feature transformer (batch training == online replay).

The feature row for hour ``t`` of a stay depends only on that stay's rows
with ``hour_index <= t``. Row ``t`` represents the hourly bin that has just
ended (the PSV gives no finer availability time; this is an approximation).
Nothing is back-filled, interpolated, centred, or normalised over a stay,
and no label, identifier, file path, hospital identity or eventual stay
length enters the features.

Feature families (``basic`` = prespecified primary set):

* vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp): latest valid value within
  an expiry, and trailing 3/6/12 h mean, min, max, std (>=2 obs) and slope
  (>=2 distinct observation times) over *observed* values only;
* labs (27 sparse measurements): latest valid value within an expiry and the
  change from the previous observation;
* demographics: Age, Gender (source definitions, unchanged);
* derived: shock index (HR/SBP) and pulse pressure from latest values.

``extended`` adds observation-pattern features (observed-now indicators,
hours since last observation, measurement counts), which can encode local
ordering workflow and are therefore studied as a separate variant.
``include_admin`` adds ICULOS, HospAdmTime and unit indicators (ablation only).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sepsis.data.schema import ADMINISTRATIVE, CLINICAL, DEMOGRAPHICS, LABS, VITALS

FEATURE_SCHEMA_PREFIX = "causal-v1"


@dataclass(frozen=True)
class FeatureConfig:
    feature_set: str = "basic"  # "basic" | "extended"
    include_admin: bool = False
    windows: tuple[int, ...] = (3, 6, 12)
    vital_expiry_h: int = 6
    lab_expiry_h: int = 48
    count_window_h: int = 24
    stats: tuple[str, ...] = ("mean", "min", "max", "std", "slope")
    expiry_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.feature_set not in ("basic", "extended"):
            raise ValueError(f"unknown feature_set {self.feature_set!r}")

    def expiry(self, var: str) -> int:
        if var in self.expiry_overrides:
            return int(self.expiry_overrides[var])
        return self.vital_expiry_h if var in VITALS else self.lab_expiry_h

    def to_dict(self) -> dict:
        d = asdict(self)
        d["windows"] = list(self.windows)
        d["stats"] = list(self.stats)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        d = dict(d or {})
        for k in ("windows", "stats"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def feature_names(cfg: FeatureConfig) -> list[str]:
    names: list[str] = []
    for v in VITALS:
        names.append(f"{v}__last")
        for w in cfg.windows:
            names += [f"{v}__{s}_{w}h" for s in cfg.stats]
    for v in LABS:
        names += [f"{v}__last", f"{v}__delta"]
    names += list(DEMOGRAPHICS)
    names += ["ShockIndex", "PulsePressure"]
    if cfg.feature_set == "extended":
        names += [f"{v}__observed" for v in CLINICAL]
        names += [f"{v}__hours_since" for v in CLINICAL]
        names += [f"vitals__count_{cfg.count_window_h}h", f"labs__count_{cfg.count_window_h}h"]
    if cfg.include_admin:
        names += list(ADMINISTRATIVE)
    return names


def schema_version(cfg: FeatureConfig) -> str:
    payload = json.dumps({"cfg": cfg.to_dict(), "names": feature_names(cfg)}, sort_keys=True)
    return f"{FEATURE_SCHEMA_PREFIX}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _group_layout(stay_ids: np.ndarray):
    """Group start index per row and position within group (data pre-sorted)."""
    n = len(stay_ids)
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = stay_ids[1:] != stay_ids[:-1]
    starts = np.flatnonzero(new)
    start_of_row = np.repeat(starts, np.diff(np.append(starts, n)))
    pos = np.arange(n) - start_of_row
    return start_of_row, pos


def _ffill_index(obs: np.ndarray, start_of_row: np.ndarray) -> np.ndarray:
    """Index of the latest observed row at or before each row (same group), else -1."""
    n = len(obs)
    idx = np.where(obs, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < start_of_row] = -1
    return idx


def _window_stack(x: np.ndarray, w: int, start_of_row: np.ndarray) -> np.ndarray:
    """(n, w) matrix of trailing values x[i-k], k=0..w-1, NaN outside the stay."""
    n = len(x)
    out = np.full((n, w), np.nan, dtype=np.float64)
    rows = np.arange(n)
    for k in range(w):
        src = rows - k
        ok = src >= start_of_row
        out[ok, k] = x[src[ok]]
    return out


def _window_stats(x: np.ndarray, w: int, start_of_row: np.ndarray, stats) -> dict[str, np.ndarray]:
    m = _window_stack(x, w, start_of_row)
    obs = ~np.isnan(m)
    cnt = obs.sum(axis=1)
    res: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if "mean" in stats:
            res["mean"] = np.nanmean(m, axis=1)
        if "min" in stats:
            res["min"] = np.nanmin(m, axis=1)
        if "max" in stats:
            res["max"] = np.nanmax(m, axis=1)
        if "std" in stats:
            sd = np.nanstd(m, axis=1, ddof=1)
            sd[cnt < 2] = np.nan
            res["std"] = sd
        if "slope" in stats:
            # time of column k is -k hours relative to the current row
            t = np.where(obs, -np.arange(w, dtype=np.float64)[None, :], np.nan)
            tm = np.nanmean(t, axis=1, keepdims=True)
            vm = np.nanmean(m, axis=1, keepdims=True)
            num = np.nansum((t - tm) * (m - vm), axis=1)
            den = np.nansum((t - tm) ** 2, axis=1)
            slope = np.where((cnt >= 2) & (den > 0), num / np.where(den > 0, den, 1), np.nan)
            res["slope"] = slope
    return res


def compute_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute causal features for every row of ``df``.

    ``df`` needs ``stay_id``, ``hour_index`` and the source input columns. It is
    sorted by (stay_id, hour_index) internally; hours must be contiguous per
    stay (validated at ingestion). Returns stay_id, hour_index + features.
    """
    cfg = cfg or FeatureConfig()
    if "SepsisLabel" in df.columns or "label" in df.columns:
        raise ValueError("label columns must not be passed to the feature transformer")
    df = df.sort_values(["stay_id", "hour_index"], kind="stable").reset_index(drop=True)
    sid = df["stay_id"].to_numpy()
    start_of_row, pos = _group_layout(sid)
    hours = df["hour_index"].to_numpy()
    if len(df) and not np.array_equal(hours - hours[start_of_row], pos):
        raise ValueError("hour_index must be contiguous within each stay")
    n = len(df)
    out: dict[str, np.ndarray] = {}
    rows = np.arange(n)
    obs_flags, hours_since = {}, {}

    for v in CLINICAL:
        x = df[v].to_numpy(dtype=np.float64)
        obs = ~np.isnan(x)
        last_idx = _ffill_index(obs, start_of_row)
        has = last_idx >= 0
        since = np.where(has, rows - last_idx, np.nan)
        last = np.where(has, x[np.maximum(last_idx, 0)], np.nan)
        last[since > cfg.expiry(v)] = np.nan
        out[f"{v}__last"] = last
        obs_flags[v], hours_since[v] = obs, since
        if v in VITALS:
            for w in cfg.windows:
                for s, arr in _window_stats(x, w, start_of_row, cfg.stats).items():
                    out[f"{v}__{s}_{w}h"] = arr
        else:
            # change between the latest observation and the one before it
            obs_rows = np.flatnonzero(obs)
            delta_at_obs = np.full(len(obs_rows), np.nan)
            if len(obs_rows) > 1:
                same = start_of_row[obs_rows[1:]] == start_of_row[obs_rows[:-1]]
                d = x[obs_rows[1:]] - x[obs_rows[:-1]]
                delta_at_obs[1:] = np.where(same, d, np.nan)
            delta_full = np.full(n, np.nan)
            delta_full[obs_rows] = delta_at_obs
            delta = np.where(has, delta_full[np.maximum(last_idx, 0)], np.nan)
            delta[since > cfg.expiry(v)] = np.nan
            out[f"{v}__delta"] = delta

    for v in DEMOGRAPHICS:
        out[v] = df[v].to_numpy(dtype=np.float64)
    hr, sbp, dbp = out["HR__last"], out["SBP__last"], out["DBP__last"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ShockIndex"] = np.where(sbp > 0, hr / sbp, np.nan)
    out["PulsePressure"] = sbp - dbp

    if cfg.feature_set == "extended":
        for v in CLINICAL:
            out[f"{v}__observed"] = obs_flags[v].astype(np.float64)
        for v in CLINICAL:
            out[f"{v}__hours_since"] = hours_since[v]
        w = cfg.count_window_h
        for name, group in (("vitals", VITALS), ("labs", LABS)):
            per_row = np.sum([obs_flags[v] for v in group], axis=0).astype(np.float64)
            csum = np.cumsum(per_row)
            prev = rows - w
            base = np.where(prev >= start_of_row, csum[np.maximum(prev, 0)], 0.0)
            # subtract the cumulative total before the stay began
            before = np.where(start_of_row > 0, csum[np.maximum(start_of_row - 1, 0)], 0.0)
            base = np.where(prev >= start_of_row, base, before)
            out[f"{name}__count_{w}h"] = csum - base

    if cfg.include_admin:
        for v in ADMINISTRATIVE:
            out[v] = df[v].to_numpy(dtype=np.float64)

    names = feature_names(cfg)
    feats = pd.DataFrame({k: out[k].astype(np.float32) for k in names})
    feats.insert(0, "hour_index", hours.astype(np.int32))
    feats.insert(0, "stay_id", sid)
    return feats


def features_at(history: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.Series:
    """Online path: feature row for the last hour of a single stay's history."""
    if history["stay_id"].nunique() != 1:
        raise ValueError("features_at expects one stay")
    return compute_features(history, cfg).iloc[-1]


# --- rule-based reference scores (partial; mental status / culture data absent) ---

def partial_qsofa(df: pd.DataFrame) -> np.ndarray:
    """Resp >= 22 and SBP <= 100 from latest values (0-2; GCS not available)."""
    return ((df["Resp__last"] >= 22).astype(float) + (df["SBP__last"] <= 100).astype(float)).to_numpy()


def partial_sirs(df: pd.DataFrame) -> np.ndarray:
    """Temp, HR, Resp, WBC SIRS criteria from latest values (0-4; PaCO2/bands not used)."""
    t, hr, rr, wbc = df["Temp__last"], df["HR__last"], df["Resp__last"], df["WBC__last"]
    return (((t > 38) | (t < 36)).astype(float) + (hr > 90).astype(float)
            + (rr > 20).astype(float) + ((wbc > 12) | (wbc < 4)).astype(float)).to_numpy()
