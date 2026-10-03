"""Application-state persistence (SQLAlchemy 2.0; PostgreSQL or SQLite).

``DATABASE_URL`` selects the database, e.g.
``postgresql+psycopg://sepsis:sepsis@localhost:5432/sepsis``. Without it a
local SQLite file under ``data/app/`` is used (WAL mode) for single-machine
development. Schemas are portable: JSON payloads are stored as text and the
job claim uses a conditional ``UPDATE ... RETURNING`` that is safe on both.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from sepsis.paths import data_dir


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32))  # registered source key or "upload"
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    quality_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Replay(Base):
    __tablename__ = "replays"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    split: Mapped[str] = mapped_column(String(24))
    target_id: Mapped[str] = mapped_column(String(48))
    model_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(32))
    policy_json: Mapped[str] = mapped_column(Text)
    feature_schema_version: Mapped[str] = mapped_column(String(48))
    speed_hours_per_tick: Mapped[float] = mapped_column(Float, default=1.0)
    tick_seconds: Mapped[float] = mapped_column(Float, default=1.0)
    clock: Mapped[int] = mapped_column(Integer, default=-1)  # last completed simulated hour
    max_hour: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="paused")  # paused|playing|finished|failed
    parent_replay_id: Mapped[str | None] = mapped_column(String(40))
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_step_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReplayStay(Base):
    __tablename__ = "replay_stays"
    replay_id: Mapped[str] = mapped_column(ForeignKey("replays.id"), primary_key=True)
    stay_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)
    n_hours: Mapped[int] = mapped_column(Integer)
    policy_state_json: Mapped[str] = mapped_column(Text)
    last_hour: Mapped[int] = mapped_column(Integer, default=-1)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("replay_id", "stay_id", "hour_index", "model_version", "policy_version",
                         name="uq_prediction_step"),
        Index("ix_pred_replay_stay", "replay_id", "stay_id", "hour_index"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    replay_id: Mapped[str] = mapped_column(ForeignKey("replays.id"))
    stay_id: Mapped[str] = mapped_column(String(64))
    hour_index: Mapped[int] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24))
    policy_state: Mapped[str] = mapped_column(String(24))
    target_id: Mapped[str] = mapped_column(String(48))
    model_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(32))
    feature_schema_version: Mapped[str] = mapped_column(String(48))
    input_hash: Mapped[str] = mapped_column(String(64))
    contributions_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("replay_id", "stay_id", "trigger_hour", "policy_version",
                                       name="uq_alert_trigger"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    replay_id: Mapped[str] = mapped_column(ForeignKey("replays.id"))
    stay_id: Mapped[str] = mapped_column(String(64))
    trigger_hour: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(32))
    state: Mapped[str] = mapped_column(String(16), default="open")  # open|acknowledged
    reviewed_by: Mapped[str | None] = mapped_column(String(64))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    annotation: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(48))
    replay_id: Mapped[str | None] = mapped_column(String(40))
    payload_json: Mapped[str | None] = mapped_column(Text)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[str] = mapped_column(Text)
    dedupe_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=5)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    lease_owner: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"
    worker_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    pid: Mapped[int] = mapped_column(Integer)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    path = data_dir() / "app" / "app.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


_engines: dict[str, object] = {}


def get_engine(url: str | None = None):
    url = url or database_url()
    if url not in _engines:
        kwargs = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        eng = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(eng, "connect")
            def _pragmas(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        Base.metadata.create_all(eng)
        _engines[url] = eng
    return _engines[url]


def session_factory(url: str | None = None):
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)


@contextmanager
def transaction(url: str | None = None):
    """One atomic unit of work; rolls back on any exception."""
    s = session_factory(url)()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def dumps(o) -> str:
    return json.dumps(o, default=float, separators=(",", ":"))


def audit(session, actor: str, kind: str, replay_id: str | None = None, **payload) -> None:
    session.add(AuditEvent(actor=actor, kind=kind, replay_id=replay_id, payload_json=dumps(payload)))


def ping(url: str | None = None) -> bool:
    with get_engine(url).connect() as c:
        c.execute(text("SELECT 1"))
    return True
