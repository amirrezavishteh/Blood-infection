"""API + worker integration: idempotent replay, recovery, clock-limited access."""

import json
from datetime import timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select


@pytest.fixture(scope="module")
def client(trained_run, corpus):
    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def dataset_id(client):
    from worker.runner import run

    r = client.post("/v1/datasets/import", json={"source": "physionet2019"})
    assert r.status_code == 202, r.text
    run(once=True, wid="test-worker")
    ds = client.get(f"/v1/datasets/{r.json()['dataset']['id']}/quality").json()
    assert ds["status"] == "ready", ds
    return ds["id"]


def _replay(client, dataset_id, **kw):
    body = {"dataset_id": dataset_id, "split": "a_test", "n_stays": 5, "seed": 1,
            "enrich_septic_fraction": 0.4, "model_version": "test-lgbm", **kw}
    r = client.post("/v1/replays", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _drain(wid="test-worker"):
    from worker.runner import run

    run(once=True, wid=wid)


def test_import_rejects_unregistered_source(client):
    assert client.post("/v1/datasets/import", json={"source": "../../etc"}).status_code == 400


def test_dataset_quality_reports_counts(client, dataset_id):
    q = client.get(f"/v1/datasets/{dataset_id}/quality").json()["quality"]
    assert set(q["hospitals"]) == {"A", "B"} and q["invalid_files"] == []
    assert sum(q["splits"].values()) == q["hospitals"]["A"]["stays"] + q["hospitals"]["B"]["stays"]


def test_replay_only_on_held_out_splits(client, dataset_id):
    r = client.post("/v1/replays", json={"dataset_id": dataset_id, "split": "a_train", "model_version": "test-lgbm"})
    assert r.status_code == 400


def test_step_is_idempotent_and_clock_limits_access(client, dataset_id):
    rp = _replay(client, dataset_id)
    rid = rp["id"]
    a = client.post(f"/v1/replays/{rid}/step", json={"target_hour": 0}).json()
    b = client.post(f"/v1/replays/{rid}/step", json={"target_hour": 0}).json()
    assert a["job_id"] == b["job_id"]  # duplicate request -> same job
    assert client.post(f"/v1/replays/{rid}/step", json={"target_hour": 5}).status_code == 409
    _drain()
    assert client.get(f"/v1/replays/{rid}").json()["clock"] == 0
    again = client.post(f"/v1/replays/{rid}/step", json={"target_hour": 0}).json()
    assert again["already_completed"]

    pats = client.get(f"/v1/replays/{rid}/patients").json()["patients"]
    sid = pats[0]["stay_id"]
    tl = client.get(f"/v1/replays/{rid}/patients/{sid}/timeline").json()
    assert tl["through_hour"] == 0
    assert all(p["hour"] <= 0 for v in tl["observations"].values() for p in v)
    assert [s["hour"] for s in tl["scores"]] == [0]
    assert client.get(f"/v1/replays/{rid}/patients/{sid}/timeline?until_hour=1").status_code == 403
    assert client.get(f"/v1/replays/{rid}/patients/{sid}/retrospective").status_code == 403
    blob = json.dumps(tl) + json.dumps(pats)
    assert "SepsisLabel" not in blob and '"label"' not in blob and "n_hours" not in blob
    for k in ("target_id", "model_version", "policy_version", "feature_schema_version"):
        assert tl[k]


def test_duplicate_execution_and_crash_rollback(client, dataset_id):
    from api.db import Alert, Prediction, transaction
    from worker import replay as rep

    rid = _replay(client, dataset_id, seed=2)["id"]
    # crash after the second stay inside the transaction: nothing may persist
    calls = []

    def boom(stay_id):
        calls.append(stay_id)
        if len(calls) == 2:
            raise RuntimeError("simulated crash mid-step")

    rep.FAULT_INJECTION = boom
    try:
        with pytest.raises(RuntimeError):
            rep.process_step(rid, 0)
    finally:
        rep.FAULT_INJECTION = None
    with transaction() as s:
        assert s.scalar(select(func.count()).select_from(Prediction).where(Prediction.replay_id == rid)) == 0
    assert client.get(f"/v1/replays/{rid}").json()["clock"] == -1

    for h in range(4):
        rep.process_step(rid, h)
        assert rep.process_step(rid, h)["noop"]  # duplicate execution is a no-op
    with pytest.raises(rep.OutOfOrderStep):
        rep.process_step(rid, 9)
    with transaction() as s:
        n = s.scalar(select(func.count()).select_from(Prediction).where(Prediction.replay_id == rid))
        dup = s.execute(select(Prediction.stay_id, Prediction.hour_index, func.count())
                        .where(Prediction.replay_id == rid)
                        .group_by(Prediction.stay_id, Prediction.hour_index).having(func.count() > 1)).all()
        n_alerts = s.scalar(select(func.count()).select_from(Alert).where(Alert.replay_id == rid))
    assert dup == [] and n > 0
    # replaying the same hours again cannot add alerts
    for h in range(4):
        rep.process_step(rid, h)
    with transaction() as s:
        assert s.scalar(select(func.count()).select_from(Alert).where(Alert.replay_id == rid)) == n_alerts


def test_expired_lease_is_reclaimed_by_another_worker(client, dataset_id):
    from api import jobs
    from api.db import Job, transaction, utcnow

    rid = _replay(client, dataset_id, seed=3)["id"]
    jid = client.post(f"/v1/replays/{rid}/step", json={}).json()["job_id"]
    job = jobs.claim("dead-worker")  # claims but never finishes (simulated crash)
    assert job.id == jid
    assert jobs.claim("other") is None  # lease still valid
    with transaction() as s:
        s.get(Job, jid).lease_expires_at = utcnow() - timedelta(seconds=1)
    _drain("recovery-worker")
    with transaction() as s:
        j = s.get(Job, jid)
        assert j.status == "done" and j.lease_owner == "recovery-worker" and j.attempts == 2
    jobs.complete(jid, "dead-worker", {"late": True})  # late completion from lost lease is ignored
    with transaction() as s:
        assert json.loads(s.get(Job, jid).result_json).get("late") is None


def test_replay_scores_match_batch_inference(client, dataset_id, corpus):
    from api.db import Prediction, transaction
    from sepsis.models.bundle import load_bundle
    from worker import replay as rep

    rid = _replay(client, dataset_id, seed=4)["id"]
    for h in range(10):
        rep.process_step(rid, h)
    b = load_bundle(rep.get_bundle("test-lgbm").path)
    feats = corpus.features(b.feature_config)
    with transaction() as s:
        preds = s.scalars(select(Prediction).where(Prediction.replay_id == rid)).all()
    batch = b.score_frame(feats[feats.stay_id.isin({p.stay_id for p in preds})][["stay_id", "hour_index"] + b.feature_names])
    batch = batch.set_index(["stay_id", "hour_index"])
    for p in preds:
        row = batch.loc[(p.stay_id, p.hour_index)]
        assert p.status == row["status"]
        if p.score is None:
            assert np.isnan(row["score"])
        else:
            assert p.score == pytest.approx(row["score"], abs=1e-9)


def test_play_to_finish_then_retrospective_and_reset(client, dataset_id):
    from api.db import Replay, transaction
    from worker.runner import run, tick_playing_replays

    rid = _replay(client, dataset_id, seed=5, n_stays=3, hours_per_second=20)["id"]
    assert client.post(f"/v1/replays/{rid}/play").json()["status"] == "playing"
    for _ in range(500):
        with transaction() as s:
            r = s.get(Replay, rid)
            r.last_step_at = None  # skip waiting for the tick interval in tests
            if r.status == "finished":
                break
        tick_playing_replays()
        run(once=True, wid="test-worker")
    rp = client.get(f"/v1/replays/{rid}").json()
    assert rp["status"] == "finished" and rp["clock"] == rp["max_hour"]
    sid = client.get(f"/v1/replays/{rid}/patients").json()["patients"][0]["stay_id"]
    retro = client.get(f"/v1/replays/{rid}/patients/{sid}/retrospective")
    assert retro.status_code == 200 and "labels" in retro.json()
    new = client.post(f"/v1/replays/{rid}/reset").json()
    assert new["id"] != rid and new["parent_replay_id"] == rid and new["clock"] == -1
    assert client.get(f"/v1/replays/{rid}").json()["clock"] == rp["clock"]  # original preserved


def test_alert_acknowledgement_is_audited(client, dataset_id):
    from worker import replay as rep

    rid = _replay(client, dataset_id, seed=6, n_stays=8, enrich_septic_fraction=0.8)["id"]
    rp = client.get(f"/v1/replays/{rid}").json()
    for h in range(rp["max_hour"] + 1):
        rep.process_step(rid, h)
    alerts = client.get(f"/v1/replays/{rid}/alerts").json()
    if not alerts:
        pytest.skip("no alerts emitted for this synthetic sample")
    a = alerts[0]
    r = client.post(f"/v1/alerts/{a['id']}/acknowledge", json={"annotation": "reviewed in simulation"},
                    headers={"X-User": "reviewer-1"}).json()
    assert r["state"] == "acknowledged" and r["reviewed_by"] == "reviewer-1"
    kinds = [e["kind"] for e in client.get(f"/v1/replays/{rid}/audit").json()]
    assert "alert_acknowledged" in kinds and "alert_emitted" in kinds


def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("SEPSIS_API_TOKEN", "s3cret")
    assert client.get("/v1/replays").status_code == 401
    assert client.get("/v1/replays", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/health/live").status_code == 200


def test_readiness_and_experiments(client, dataset_id):
    r = client.get("/health/ready").json()
    assert r["checks"]["database"] and r["checks"]["model"]
    ex = client.get("/v1/experiments").json()
    assert any(run["run_id"] == "test-lgbm" for run in ex["runs"])
    assert client.get("/v1/experiments/test-lgbm").json()["calibrated"]
    assert client.get("/v1/experiments/../../etc").status_code == 404
