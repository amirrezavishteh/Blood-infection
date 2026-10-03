"""Job queue and replay engine on PostgreSQL (embedded server via ``pgserver``).

Skipped when pgserver/psycopg are not installed (``pip install pgserver psycopg[binary]``).
"""

import threading

import pytest

pgserver = pytest.importorskip("pgserver")
pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def pg_url(tmp_path_factory):
    srv = pgserver.get_server(tmp_path_factory.mktemp("pg") / "data", cleanup_mode="stop")
    uri = srv.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    yield uri


@pytest.fixture
def on_postgres(pg_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    from api.db import database_url, get_engine

    assert database_url().startswith("postgresql+psycopg://")
    assert get_engine().dialect.name == "postgresql"
    return pg_url


def test_concurrent_claims_never_double_assign(on_postgres):
    from api import jobs

    ids = [jobs.enqueue("noop", {"i": i}) for i in range(40)]
    claimed: dict[int, str] = {}
    lock = threading.Lock()
    errors = []

    def worker(name):
        try:
            while True:
                j = jobs.claim(name)
                if j is None:
                    return
                with lock:
                    assert j.id not in claimed, f"{j.id} claimed twice"
                    claimed[j.id] = name
                jobs.complete(j.id, name, {})
        except Exception as exc:  # surface assertion failures from threads
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert set(ids) <= set(claimed)


def test_dedupe_key_is_unique(on_postgres):
    from api import jobs

    a = jobs.enqueue("replay_step", {"x": 1}, dedupe_key="pg:dedupe:1")
    b = jobs.enqueue("replay_step", {"x": 1}, dedupe_key="pg:dedupe:1")
    assert a == b


def test_replay_steps_on_postgres(on_postgres, trained_run, corpus):
    from sqlalchemy import func, select

    from api.db import Dataset, Prediction, Replay, ReplayStay, dumps, transaction
    from sepsis.alerts.policy import AlertPolicy, initial_state
    from worker import replay as rep

    b = rep.get_bundle("test-lgbm")
    pol = AlertPolicy.from_dict(b.policy)
    stays = corpus.stays("a_test")[:3]
    enc = corpus.encounters.set_index("stay_id")
    with transaction() as s:
        ds = Dataset(name="pg", source="physionet2019", status="ready")
        s.add(ds)
        s.flush()
        s.add(Replay(id="rp-pg-1", dataset_id=ds.id, split="a_test", target_id=b.target_id, model_version="test-lgbm",
                     policy_version=pol.version, policy_json=dumps(pol.to_dict()),
                     feature_schema_version=b.feature_schema_version, max_hour=5, created_by="t"))
        s.flush()
        for i, sid in enumerate(stays):
            s.add(ReplayStay(replay_id="rp-pg-1", stay_id=sid, position=i, n_hours=int(enc.loc[sid, "n_hours"]),
                             policy_state_json=dumps(initial_state())))
    for h in range(6):
        rep.process_step("rp-pg-1", h)
        assert rep.process_step("rp-pg-1", h)["noop"]
    with transaction() as s:
        n = s.scalar(select(func.count()).select_from(Prediction).where(Prediction.replay_id == "rp-pg-1"))
        assert s.get(Replay, "rp-pg-1").status == "finished"
    assert n == sum(min(6, int(enc.loc[sid, "n_hours"])) for sid in stays)
