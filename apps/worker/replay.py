"""Replay engine: ordered hourly scoring with transactional checkpoints.

Each step (replay, hour h) runs in ONE database transaction:
load only measurements with hour_index <= h -> shared causal features ->
pinned model bundle -> data-quality status -> pinned alert policy using the
persisted prior state -> insert prediction / alert / audit event and advance
the replay checkpoint. A crash anywhere rolls the whole step back; a retried
or duplicated step is a no-op once the checkpoint has advanced, and unique
constraints make duplicate predictions/alerts impossible.

Outcome labels are never loaded here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from api.db import (
    Alert,
    Dataset,
    Prediction,
    Replay,
    ReplayStay,
    audit,
    dumps,
    transaction,
    utcnow,
)
from sepsis.alerts.policy import AlertPolicy, step as policy_step
from sepsis.data.store import SCORING_DROP, Store
from sepsis.features.causal import compute_features
from sepsis.models.bundle import bundle_root, load_bundle

log = logging.getLogger(__name__)

# Test hook: called after each stay is processed inside the step transaction.
FAULT_INJECTION = None


class OutOfOrderStep(RuntimeError):
    pass


def aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---- shared caches (per process) -------------------------------------------

@lru_cache(maxsize=4)
def get_bundle(model_version: str):
    return load_bundle(bundle_root() / model_version)


@lru_cache(maxsize=2)
def get_store(source: str) -> Store:
    if source not in REGISTERED_SOURCES:
        raise ValueError(f"unregistered dataset source {source!r}")
    return Store.for_dataset(REGISTERED_SOURCES[source])


REGISTERED_SOURCES = {"physionet2019": "physionet2019"}


@lru_cache(maxsize=8)
def _stay_frames(source: str, stay_ids: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    store = get_store(source)
    h = store.hourly
    sub = h[h["stay_id"].isin(set(stay_ids))].drop(columns=SCORING_DROP)
    return {sid: g.sort_values("hour_index").reset_index(drop=True) for sid, g in sub.groupby("stay_id")}


def visible_history(source: str, stay_ids: tuple[str, ...], stay_id: str, hour: int) -> pd.DataFrame:
    frame = _stay_frames(source, stay_ids)[stay_id]
    return frame[frame["hour_index"] <= hour]


def default_model_version() -> str | None:
    root = bundle_root()
    promo = root.parent / "promotion.json"
    if promo.exists():
        sel = json.loads(promo.read_text()).get("selected")
        if sel and (root / sel / "bundle.json").exists():
            return sel
    if not root.exists():
        return None
    ready = []
    for rd in sorted(root.iterdir()):
        bj = rd / "bundle.json"
        if bj.exists():
            m = json.loads(bj.read_text())
            if m.get("calibration") and m.get("alert_policy"):
                ready.append(rd.name)
    return ready[-1] if ready else None


# ---- step -------------------------------------------------------------------

def process_step(replay_id: str, target_hour: int, actor: str = "worker") -> dict:
    with transaction() as s:
        r = s.get(Replay, replay_id, with_for_update=True)
        if r is None:
            raise ValueError(f"unknown replay {replay_id}")
        if r.clock >= target_hour:
            return {"noop": True, "clock": r.clock}
        if target_hour != r.clock + 1:
            raise OutOfOrderStep(f"replay {replay_id} at hour {r.clock}; cannot step to {target_hour}")
        if r.status == "finished":
            return {"noop": True, "clock": r.clock}
        ds = s.get(Dataset, r.dataset_id)
        bundle = get_bundle(r.model_version)
        if bundle.feature_schema_version != r.feature_schema_version:
            raise RuntimeError("model bundle schema differs from the replay's pinned schema")
        policy = AlertPolicy.from_dict(json.loads(r.policy_json))
        if policy.version != r.policy_version:
            raise RuntimeError("policy version mismatch")
        stays = s.scalars(select(ReplayStay).where(ReplayStay.replay_id == replay_id)
                          .order_by(ReplayStay.position)).all()
        stay_ids = tuple(sorted(st.stay_id for st in stays))
        n_scored = n_alerts = 0
        for st in stays:
            if target_hour >= st.n_hours:
                continue  # record ended (discharge/end of data)
            hist = visible_history(ds.source, stay_ids, st.stay_id, target_hour)
            assert hist["hour_index"].max() <= target_hour
            feats = compute_features(hist, bundle.feature_config).iloc[[-1]]
            scored = bundle.score_frame(feats).iloc[0]
            score = None if np.isnan(scored["score"]) else float(scored["score"])
            x = feats[bundle.feature_names].to_numpy(dtype=np.float32)[0]
            contrib = bundle.contributions(x) if score is not None else None
            state, emitted = policy_step(policy, json.loads(st.policy_state_json), target_hour,
                                         score, scored["status"])
            exists = s.scalar(select(Prediction.id).where(
                Prediction.replay_id == replay_id, Prediction.stay_id == st.stay_id,
                Prediction.hour_index == target_hour, Prediction.model_version == r.model_version,
                Prediction.policy_version == r.policy_version))
            if exists is None:
                s.add(Prediction(replay_id=replay_id, stay_id=st.stay_id, hour_index=target_hour,
                                 score=score, status=scored["status"], policy_state=state["state"],
                                 target_id=r.target_id, model_version=r.model_version,
                                 policy_version=r.policy_version,
                                 feature_schema_version=r.feature_schema_version,
                                 input_hash=hashlib.sha256(x.tobytes()).hexdigest(),
                                 contributions_json=dumps(contrib) if contrib else None))
                n_scored += 1
            if emitted:
                s.add(Alert(replay_id=replay_id, stay_id=st.stay_id, trigger_hour=target_hour,
                            score=score, model_version=r.model_version, policy_version=r.policy_version))
                n_alerts += 1
                audit(s, actor, "alert_emitted", replay_id, stay_id=st.stay_id, hour=target_hour, score=score)
            st.policy_state_json = dumps(state)
            st.last_hour = target_hour
            if FAULT_INJECTION:
                FAULT_INJECTION(st.stay_id)
        r.clock = target_hour
        r.last_step_at = utcnow()
        if target_hour >= r.max_hour:
            r.status = "finished"
        audit(s, actor, "replay_step", replay_id, hour=target_hour, scored=n_scored, alerts=n_alerts)
        return {"clock": target_hour, "scored": n_scored, "alerts": n_alerts, "status": r.status}


def validate_dataset(dataset_id: int) -> dict:
    """Dataset import job: verify processed outputs, counts and split file."""
    from sepsis.data.schema import sha256_file

    with transaction() as s:
        ds = s.get(Dataset, dataset_id)
        ds.status = "validating"
    try:
        store = get_store(ds.source)
        q = store.quality
        for name, digest in (q.get("outputs") or {}).items():
            if sha256_file(store.root / name) != digest:
                raise RuntimeError(f"{name} checksum differs from quality.json")
        split = store.splits  # verifies the split hash
        summary = {
            "dataset": "physionet2019", "dataset_version": "1.0.0",
            "hospitals": {h: {k: v[k] for k in ("stays", "rows", "patient_days", "positive_hour_prevalence",
                                                "ever_positive_stays", "left_censored_positive_stays",
                                                "median_hours", "max_hours", "missing_fraction")}
                          for h, v in q["hospitals"].items()},
            "invalid_files": q["invalid_files"], "schema": q["schema"],
            "raw_manifest_sha256": q.get("raw_manifest_sha256"), "outputs": q.get("outputs"),
            "splits": split["counts"], "split_hash": split["split_hash"],
            "synthetic": bool(q.get("synthetic")),
            "license": ("synthetic fixture (no patient data)" if q.get("synthetic")
                        else "CC BY 4.0 (PhysioNet/CinC Challenge 2019)"),
        }
        with transaction() as s:
            ds = s.get(Dataset, dataset_id)
            ds.status, ds.quality_json, ds.dataset_version, ds.error = "ready", dumps(summary), "1.0.0", None
        return {"status": "ready"}
    except Exception as exc:
        with transaction() as s:
            ds = s.get(Dataset, dataset_id)
            ds.status, ds.error = "failed", str(exc)
        raise
