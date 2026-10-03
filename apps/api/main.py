"""FastAPI backend for the sepsis research simulator.

Run: ``uvicorn api.main:app --app-dir apps`` (binds to localhost by default
via the launch scripts). Every response carrying a score names the target,
model, policy and feature-schema versions. Active-replay endpoints never
return data beyond the replay clock and never return outcome labels; the
retrospective view is available only after a replay has finished.

Optional access control: if ``SEPSIS_API_TOKEN`` is set, every /v1 request
must send ``Authorization: Bearer <token>``. Reviewer identity comes from
the ``X-User`` header (default ``local-user``).
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from api import jobs
from api.db import (
    Alert,
    AuditEvent,
    Dataset,
    Job,
    Prediction,
    Replay,
    ReplayStay,
    WorkerHeartbeat,
    audit,
    dumps,
    ping,
    transaction,
    utcnow,
)
from sepsis import __version__
from sepsis.alerts.policy import AlertPolicy, initial_state
from sepsis.data.schema import CLINICAL, MAX_FILE_BYTES, PSVValidationError, UNITS, parse_psv_bytes
from sepsis.data.splits import HELD_OUT_SPLITS
from sepsis.labels.targets import onset_from_challenge_labels
from sepsis.models.bundle import bundle_root
from sepsis.paths import PROJECT_ROOT, artifact_dir, data_dir, ensure_within
from worker.replay import REGISTERED_SOURCES, aware, default_model_version, get_bundle, get_store

API_VERSION = "v1"
MAX_REPLAY_STAYS = 100
DISCLAIMER = ("Research sepsis score for the PhysioNet 2019 benchmark target (challenge2019_state). "
              "Retrospective simulation only; not a cleared diagnostic device or a clinically validated "
              "probability, and not for patient care.")

app = FastAPI(title="Sepsis early-warning research simulator", version=__version__,
              description=DISCLAIMER)


# ---- access ------------------------------------------------------------------

def require_access(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("SEPSIS_API_TOKEN")
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer ") or \
            not secrets.compare_digest(authorization[7:], token):
        raise HTTPException(401, "missing or invalid bearer token")


def current_user(x_user: str | None = Header(default=None)) -> str:
    user = (x_user or "local-user").strip()[:64]
    if not user.replace("-", "").replace("_", "").replace(".", "").replace("@", "").isalnum():
        raise HTTPException(400, "invalid X-User")
    return user


V1 = [Depends(require_access)]


@app.exception_handler(PSVValidationError)
async def _psv_error(_: Request, exc: PSVValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ---- health ------------------------------------------------------------------

@app.get("/health/live")
def live():
    return {"status": "live", "version": __version__}


@app.get("/health/ready")
def ready():
    checks = {}
    try:
        checks["database"] = ping()
    except Exception as exc:
        checks["database"] = False
        checks["database_error"] = type(exc).__name__
    with transaction() as s:
        last = s.scalar(select(func.max(WorkerHeartbeat.last_seen)))
    checks["worker"] = bool(last and utcnow() - aware(last) < timedelta(seconds=30))
    mv = default_model_version()
    checks["model"] = False
    if mv:
        try:
            get_bundle(mv)
            checks["model"] = True
            checks["model_version"] = mv
        except Exception as exc:
            checks["model_error"] = str(exc)
    ok = checks["database"] and checks["worker"] and checks["model"]
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})


@app.get("/v1/meta", dependencies=V1)
def meta():
    return {"api_version": API_VERSION, "version": __version__, "disclaimer": DISCLAIMER,
            "registered_sources": sorted(REGISTERED_SOURCES), "held_out_splits": list(HELD_OUT_SPLITS),
            "default_model_version": default_model_version(), "units": UNITS}


# ---- datasets ----------------------------------------------------------------

class ImportRequest(BaseModel):
    source: str = Field(description="registered server-side source key, e.g. physionet2019")
    name: str | None = None


def _dataset_out(d: Dataset) -> dict:
    return {"id": d.id, "name": d.name, "source": d.source, "dataset_version": d.dataset_version,
            "status": d.status, "error": d.error, "created_at": d.created_at.isoformat()}


@app.post("/v1/datasets/import", dependencies=V1, status_code=202)
def import_dataset(req: ImportRequest, user: str = Depends(current_user)):
    if req.source not in REGISTERED_SOURCES:  # never arbitrary filesystem paths
        raise HTTPException(400, f"unknown source; registered: {sorted(REGISTERED_SOURCES)}")
    with transaction() as s:
        d = Dataset(name=req.name or req.source, source=req.source, status="pending")
        s.add(d)
        s.flush()
        job_id = jobs.enqueue("dataset_validate", {"dataset_id": d.id}, session=s, max_attempts=1)
        audit(s, user, "dataset_import", None, dataset_id=d.id, source=req.source)
        return {"dataset": _dataset_out(d), "job_id": job_id}


@app.post("/v1/datasets/upload", dependencies=V1)
async def upload_psv(request: Request, filename: str = Query(..., pattern=r"^[A-Za-z0-9_.-]{1,64}\.psv$"),
                     user: str = Depends(current_user)):
    """Validate one uploaded challenge-format PSV file and report its quality.

    The file is stored under data/uploads for audit; uploaded files are not
    replayable because they have no held-out split assignment.
    """
    raw = await request.body()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, "file too large")
    df = parse_psv_bytes(raw, filename)
    with transaction() as s:
        d = Dataset(name=filename, source="upload", status="ready", dataset_version="upload")
        s.add(d)
        s.flush()
        up = data_dir() / "uploads" / str(d.id)
        up.mkdir(parents=True, exist_ok=True)
        (up / filename).write_bytes(raw)
        quality = {"rows": int(len(df)), "columns": int(df.shape[1]),
                   "missing_fraction": {c: float(df[c].isna().mean()) for c in CLINICAL},
                   "label_present": True, "label_hidden_from_scoring": True}
        d.quality_json = dumps(quality)
        audit(s, user, "dataset_upload", None, dataset_id=d.id, filename=filename)
        return {"dataset": _dataset_out(d), "quality": quality}


@app.get("/v1/datasets", dependencies=V1)
def list_datasets():
    with transaction() as s:
        return [_dataset_out(d) for d in s.scalars(select(Dataset).order_by(Dataset.id.desc())).all()]


@app.get("/v1/datasets/{dataset_id}/quality", dependencies=V1)
def dataset_quality(dataset_id: int):
    with transaction() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404, "dataset not found")
        return {**_dataset_out(d), "quality": json.loads(d.quality_json) if d.quality_json else None}


@app.get("/v1/jobs/{job_id}", dependencies=V1)
def job_status(job_id: int):
    with transaction() as s:
        j = s.get(Job, job_id)
        if not j:
            raise HTTPException(404, "job not found")
        return {"id": j.id, "kind": j.kind, "status": j.status, "attempts": j.attempts,
                "last_error": j.last_error, "result": json.loads(j.result_json) if j.result_json else None}


# ---- replays -----------------------------------------------------------------

class ReplayRequest(BaseModel):
    dataset_id: int
    split: str = "a_test"
    stay_ids: list[str] | None = None
    n_stays: int = Field(default=12, ge=1, le=MAX_REPLAY_STAYS)
    seed: int = 0
    enrich_septic_fraction: float | None = Field(
        default=None, ge=0, le=1,
        description="optional demo enrichment: sample this fraction of records that are ever positive. "
                    "Labels are used for record selection only and remain hidden from scoring.")
    model_version: str | None = None
    hours_per_second: float = Field(default=2.0, gt=0, le=20)


def _replay_out(r: Replay) -> dict:
    return {"id": r.id, "dataset_id": r.dataset_id, "split": r.split, "target_id": r.target_id,
            "model_version": r.model_version, "policy_version": r.policy_version,
            "policy": json.loads(r.policy_json), "feature_schema_version": r.feature_schema_version,
            "clock": r.clock, "max_hour": r.max_hour, "status": r.status,
            "hours_per_second": round(1.0 / r.tick_seconds, 3), "parent_replay_id": r.parent_replay_id,
            "created_by": r.created_by, "created_at": r.created_at.isoformat(),
            "simulated_clock_note": "simulated ICU hours since admission; distinct from wall-clock time"}


def _get_replay(s, replay_id: str) -> Replay:
    r = s.get(Replay, replay_id)
    if not r:
        raise HTTPException(404, "replay not found")
    return r


def _create_replay(s, ds: Dataset, split: str, stay_ids: list[str], model_version: str,
                   hours_per_second: float, user: str, parent: str | None = None) -> Replay:
    try:
        bundle = get_bundle(model_version)
    except Exception as exc:
        raise HTTPException(400, f"model bundle unavailable: {exc}")
    if not bundle.calibration or not bundle.policy:
        raise HTTPException(400, "model bundle has no calibration or frozen alert policy")
    store = get_store(ds.source)
    if bundle.meta["provenance"]["split_hash"] != store.splits["split_hash"]:
        raise HTTPException(400, "model was trained on a different split")
    enc = store.encounters.set_index("stay_id")
    policy = AlertPolicy.from_dict(bundle.policy)
    n_hours = [int(enc.loc[sid, "n_hours"]) for sid in stay_ids]
    r = Replay(id=f"rp-{uuid.uuid4().hex[:12]}", dataset_id=ds.id, split=split, target_id=bundle.target_id,
               model_version=model_version, policy_version=policy.version, policy_json=dumps(policy.to_dict()),
               feature_schema_version=bundle.feature_schema_version, tick_seconds=1.0 / hours_per_second,
               clock=-1, max_hour=max(n_hours) - 1, status="paused", created_by=user, parent_replay_id=parent)
    s.add(r)
    s.flush()
    for i, (sid, n) in enumerate(zip(stay_ids, n_hours)):
        s.add(ReplayStay(replay_id=r.id, stay_id=sid, position=i, n_hours=n,
                         policy_state_json=dumps(initial_state())))
    audit(s, user, "replay_created", r.id, split=split, stays=len(stay_ids), model_version=model_version,
          policy_version=policy.version, parent=parent)
    return r


@app.post("/v1/replays", dependencies=V1, status_code=201)
def create_replay(req: ReplayRequest, user: str = Depends(current_user)):
    if req.split not in HELD_OUT_SPLITS:
        raise HTTPException(400, f"replays are limited to held-out splits {HELD_OUT_SPLITS}")
    with transaction() as s:
        ds = s.get(Dataset, req.dataset_id)
        if not ds or ds.status != "ready" or ds.source not in REGISTERED_SOURCES:
            raise HTTPException(400, "dataset must be an imported, validated registered source")
        store = get_store(ds.source)
        pool = store.stays(req.split)
        if req.stay_ids:
            bad = sorted(set(req.stay_ids) - set(pool))
            if bad:
                raise HTTPException(400, f"records not in split {req.split}: {bad[:5]}")
            if len(req.stay_ids) > MAX_REPLAY_STAYS:
                raise HTTPException(400, "too many records")
            stay_ids = list(dict.fromkeys(req.stay_ids))
        else:
            rng = np.random.default_rng(req.seed)
            n = min(req.n_stays, len(pool))
            if req.enrich_septic_fraction is not None:
                ever = store.encounters.set_index("stay_id")["ever_positive"]
                pos = [p for p in pool if ever[p]]
                neg = [p for p in pool if not ever[p]]
                k = min(len(pos), int(round(n * req.enrich_septic_fraction)))
                stay_ids = list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, n - k, replace=False))
                rng.shuffle(stay_ids)
            else:
                stay_ids = list(rng.choice(pool, n, replace=False))
            stay_ids = [str(x) for x in stay_ids]
        mv = req.model_version or default_model_version()
        if not mv:
            raise HTTPException(400, "no trained, calibrated model bundle with a frozen policy is available")
        r = _create_replay(s, ds, req.split, stay_ids, mv, req.hours_per_second, user)
        return _replay_out(r)


@app.get("/v1/replays", dependencies=V1)
def list_replays():
    with transaction() as s:
        return [_replay_out(r) for r in s.scalars(select(Replay).order_by(Replay.created_at.desc())).all()]


@app.get("/v1/replays/{replay_id}", dependencies=V1)
def get_replay(replay_id: str):
    with transaction() as s:
        return _replay_out(_get_replay(s, replay_id))


class StepRequest(BaseModel):
    target_hour: int | None = Field(default=None, description="idempotency: the hour this step should reach")


@app.post("/v1/replays/{replay_id}/step", dependencies=V1, status_code=202)
def step_replay(replay_id: str, req: StepRequest | None = None, user: str = Depends(current_user)):
    with transaction() as s:
        r = _get_replay(s, replay_id)
        if r.status == "finished":
            raise HTTPException(409, "replay finished")
        pending = s.scalars(select(Job).where(Job.kind == "replay_step", Job.status.in_(("queued", "running")),
                                              Job.dedupe_key.like(f"replay:{replay_id}:step:%"))).all()
        next_hour = max([r.clock] + [json.loads(j.payload_json)["target_hour"] for j in pending]) + 1
        target = req.target_hour if req and req.target_hour is not None else next_hour
        if target <= r.clock:  # already done: idempotent no-op
            done = s.scalar(select(Job.id).where(Job.dedupe_key == f"replay:{replay_id}:step:{target}"))
            return {"job_id": done, "target_hour": target, "already_completed": True}
        if target > next_hour:
            raise HTTPException(409, f"steps are sequential; next hour is {next_hour}")
        if target > r.max_hour:
            raise HTTPException(409, "beyond the end of the replay")
        job_id = jobs.enqueue("replay_step", {"replay_id": replay_id, "target_hour": target},
                              dedupe_key=f"replay:{replay_id}:step:{target}", session=s)
        audit(s, user, "replay_step_requested", replay_id, target_hour=target)
        return {"job_id": job_id, "target_hour": target, "already_completed": False}


@app.post("/v1/replays/{replay_id}/play", dependencies=V1)
def play_replay(replay_id: str, user: str = Depends(current_user)):
    with transaction() as s:
        r = _get_replay(s, replay_id)
        if r.status == "finished":
            raise HTTPException(409, "replay finished")
        r.status = "playing"
        audit(s, user, "replay_play", replay_id)
        return _replay_out(r)


@app.post("/v1/replays/{replay_id}/pause", dependencies=V1)
def pause_replay(replay_id: str, user: str = Depends(current_user)):
    """Stops scheduling further steps; an in-flight step completes atomically."""
    with transaction() as s:
        r = _get_replay(s, replay_id)
        if r.status == "playing":
            r.status = "paused"
        audit(s, user, "replay_pause", replay_id)
        return _replay_out(r)


@app.post("/v1/replays/{replay_id}/reset", dependencies=V1, status_code=201)
def reset_replay(replay_id: str, user: str = Depends(current_user)):
    """Fresh replay id with the same pinned configuration; the original run is preserved."""
    with transaction() as s:
        old = _get_replay(s, replay_id)
        if old.status == "playing":
            old.status = "paused"
        stays = s.scalars(select(ReplayStay).where(ReplayStay.replay_id == replay_id)
                          .order_by(ReplayStay.position)).all()
        ds = s.get(Dataset, old.dataset_id)
        r = _create_replay(s, ds, old.split, [x.stay_id for x in stays], old.model_version,
                           1.0 / old.tick_seconds, user, parent=old.id)
        return _replay_out(r)


@app.get("/v1/replays/{replay_id}/patients", dependencies=V1)
def replay_patients(replay_id: str):
    with transaction() as s:
        r = _get_replay(s, replay_id)
        stays = s.scalars(select(ReplayStay).where(ReplayStay.replay_id == replay_id)
                          .order_by(ReplayStay.position)).all()
        out = []
        for st in stays:
            latest = s.scalars(select(Prediction).where(Prediction.replay_id == replay_id,
                                                        Prediction.stay_id == st.stay_id,
                                                        Prediction.hour_index <= r.clock)
                               .order_by(Prediction.hour_index.desc()).limit(1)).first()
            n_alerts = s.scalar(select(func.count(Alert.id)).where(
                Alert.replay_id == replay_id, Alert.stay_id == st.stay_id, Alert.trigger_hour <= r.clock))
            open_alerts = s.scalar(select(func.count(Alert.id)).where(
                Alert.replay_id == replay_id, Alert.stay_id == st.stay_id, Alert.trigger_hour <= r.clock,
                Alert.state == "open"))
            visible = min(r.clock, st.n_hours - 1) + 1 if r.clock >= 0 else 0
            out.append({
                "stay_id": st.stay_id, "visible_hours": visible,
                "active": r.clock < st.n_hours - 1,
                # total length is revealed only once the record has ended at the replay clock
                "ended": r.clock >= st.n_hours - 1,
                "latest_hour": latest.hour_index if latest else None,
                "latest_score": latest.score if latest else None,
                "latest_status": latest.status if latest else None,
                "policy_state": latest.policy_state if latest else None,
                "alerts": n_alerts, "open_alerts": open_alerts,
            })
        return {"replay": _replay_out(r), "patients": out}


def _clean(v):
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f


@app.get("/v1/replays/{replay_id}/patients/{stay_id}/timeline", dependencies=V1)
def patient_timeline(replay_id: str, stay_id: str, until_hour: int | None = None):
    with transaction() as s:
        r = _get_replay(s, replay_id)
        st = s.get(ReplayStay, (replay_id, stay_id))
        if not st:
            raise HTTPException(404, "record not in this replay")
        clock = r.clock
        if until_hour is not None and until_hour > clock:
            raise HTTPException(403, f"hour {until_hour} is beyond the replay clock ({clock})")
        limit = clock if until_hour is None else until_hour
        ended = clock >= st.n_hours - 1
        # once a record has ended at the replay clock, freshness is relative to its last hour
        limit = min(limit, st.n_hours - 1)
        preds = s.scalars(select(Prediction).where(Prediction.replay_id == replay_id,
                                                   Prediction.stay_id == stay_id,
                                                   Prediction.hour_index <= limit)
                          .order_by(Prediction.hour_index)).all()
        alerts = s.scalars(select(Alert).where(Alert.replay_id == replay_id, Alert.stay_id == stay_id,
                                               Alert.trigger_hour <= limit).order_by(Alert.trigger_hour)).all()
        ds = s.get(Dataset, r.dataset_id)
    store = get_store(ds.source)
    h = store.hourly
    rows = h[(h["stay_id"] == stay_id) & (h["hour_index"] <= limit)].sort_values("hour_index")
    observations = {}
    freshness = {}
    for v in CLINICAL:
        col = rows[v].to_numpy(dtype=float)
        pts = [{"hour": int(hh), "value": float(f"{x:.6g}")}  # float32 storage -> source precision
               for hh, x in zip(rows["hour_index"], col) if not np.isnan(x)]
        if pts:
            observations[v] = pts
            freshness[v] = {"last_hour": pts[-1]["hour"], "hours_since": int(limit - pts[-1]["hour"])}
    demo = rows.iloc[0][["Age", "Gender"]].to_dict() if len(rows) else {}
    latest = preds[-1] if preds else None
    return {
        "replay_id": replay_id, "stay_id": stay_id, "clock": clock, "through_hour": limit, "record_ended": ended,
        "target_id": r.target_id, "model_version": r.model_version, "policy_version": r.policy_version,
        "feature_schema_version": r.feature_schema_version,
        "demographics": {"Age": _clean(demo.get("Age")), "Gender_source_code": _clean(demo.get("Gender"))},
        "units": {v: UNITS[v] for v in observations},
        "observations": observations, "freshness": freshness,
        "scores": [{"hour": p.hour_index, "score": p.score, "status": p.status, "policy_state": p.policy_state,
                    "input_hash": p.input_hash} for p in preds],
        "contributions": json.loads(latest.contributions_json) if latest and latest.contributions_json else None,
        "contributions_note": "associations used by the model (log-odds, before calibration); "
                              "not causal explanations or treatment advice",
        "alerts": [_alert_out(a) for a in alerts],
        "threshold": json.loads(r.policy_json)["threshold"],
        "disclaimer": DISCLAIMER,
    }


@app.get("/v1/replays/{replay_id}/patients/{stay_id}/retrospective", dependencies=V1)
def retrospective(replay_id: str, stay_id: str):
    """Labels and onset proxy, available only after the replay has finished."""
    with transaction() as s:
        r = _get_replay(s, replay_id)
        if r.status != "finished":
            raise HTTPException(403, "retrospective labels are available only after the replay finishes")
        if not s.get(ReplayStay, (replay_id, stay_id)):
            raise HTTPException(404, "record not in this replay")
        ds = s.get(Dataset, r.dataset_id)
    store = get_store(ds.source)
    o = store.outcomes
    lab = o[o["stay_id"] == stay_id].sort_values("hour_index")["label"].to_numpy()
    info = onset_from_challenge_labels(lab)
    return {"stay_id": stay_id, "labels": lab.tolist(), "label_version": "challenge2019_state@1",
            "label_start_hour": info.label_start, "onset_proxy_hour": info.onset_hour,
            "left_censored": info.left_censored,
            "note": "onset proxy = first positive label hour + 6 (analysis convention, not clinical confirmation)"}


# ---- alerts ------------------------------------------------------------------

def _alert_out(a: Alert) -> dict:
    return {"id": a.id, "replay_id": a.replay_id, "stay_id": a.stay_id, "trigger_hour": a.trigger_hour,
            "score": a.score, "state": a.state, "reviewed_by": a.reviewed_by,
            "reviewed_at": a.reviewed_at.isoformat() if a.reviewed_at else None, "annotation": a.annotation,
            "model_version": a.model_version, "policy_version": a.policy_version}


@app.get("/v1/replays/{replay_id}/alerts", dependencies=V1)
def replay_alerts(replay_id: str):
    with transaction() as s:
        r = _get_replay(s, replay_id)
        alerts = s.scalars(select(Alert).where(Alert.replay_id == replay_id, Alert.trigger_hour <= r.clock)
                           .order_by(Alert.trigger_hour, Alert.id)).all()
        return [_alert_out(a) for a in alerts]


class AckRequest(BaseModel):
    annotation: str | None = Field(default=None, max_length=2000)


@app.post("/v1/alerts/{alert_id}/acknowledge", dependencies=V1)
def acknowledge_alert(alert_id: int, req: AckRequest | None = None, user: str = Depends(current_user)):
    """Records simulated review only; it is not an outcome label and never retrains the model."""
    with transaction() as s:
        a = s.get(Alert, alert_id)
        if not a:
            raise HTTPException(404, "alert not found")
        if a.state == "acknowledged":
            return _alert_out(a)
        a.state, a.reviewed_by, a.reviewed_at = "acknowledged", user, utcnow()
        a.annotation = req.annotation if req else None
        st = s.get(ReplayStay, (a.replay_id, a.stay_id))
        state = json.loads(st.policy_state_json)
        if state.get("state") == "alerted" and state.get("last_alert_hour") == a.trigger_hour:
            state["state"] = "acknowledged"
            st.policy_state_json = dumps(state)
        audit(s, user, "alert_acknowledged", a.replay_id, alert_id=a.id, stay_id=a.stay_id)
        return _alert_out(a)


@app.get("/v1/replays/{replay_id}/audit", dependencies=V1)
def replay_audit(replay_id: str, limit: int = Query(200, le=1000)):
    with transaction() as s:
        ev = s.scalars(select(AuditEvent).where(AuditEvent.replay_id == replay_id)
                       .order_by(AuditEvent.id.desc()).limit(limit)).all()
        return [{"at": e.at.isoformat(), "actor": e.actor, "kind": e.kind,
                 "payload": json.loads(e.payload_json) if e.payload_json else None} for e in ev]


# ---- experiments -------------------------------------------------------------

@app.get("/v1/experiments", dependencies=V1)
def list_experiments():
    from sepsis.evaluation.report import collect_runs

    runs = collect_runs()
    for r in runs:  # summary list: keep evaluations compact
        r["evaluations"] = {k: _eval_summary(v) for k, v in r["evaluations"].items()}
    return {"runs": runs, "default_model_version": default_model_version()}


def _eval_summary(m: dict) -> dict:
    return {"auroc": m["hourly"]["auroc"], "auprc": m["hourly"]["auprc"],
            "utility": m["benchmark"]["utility"], "early_sensitivity": m["alerts"]["early_sensitivity"],
            "alerts_per_100_patient_days": m["alerts"]["alerts_per_100_patient_days"],
            "records": m["records"]}


@app.get("/v1/experiments/{run_id}", dependencies=V1)
def get_experiment(run_id: str):
    from sepsis.evaluation.report import collect_runs

    for r in collect_runs():
        if r["run_id"] == run_id:
            rd = ensure_within(bundle_root() / run_id, artifact_dir())
            meta = json.loads((rd / "bundle.json").read_text())
            r["selected_params"] = meta.get("selected_params")
            r["tuning_trials"] = meta.get("tuning_trials")
            r["calibration"] = meta.get("calibration")
            r["dropped_features"] = meta.get("dropped_features")
            return r
    raise HTTPException(404, "experiment not found")


# ---- dashboard (built React app) ---------------------------------------------

_WEB_DIST = PROJECT_ROOT / "apps" / "web" / "dist"
if (_WEB_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        # index.html must revalidate so a rebuilt dashboard (new hashed assets) is picked up
        return FileResponse(_WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
