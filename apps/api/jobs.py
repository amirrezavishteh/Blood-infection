"""Persistent job queue on the application database.

Claiming is a single conditional ``UPDATE ... RETURNING`` so two workers can
never own the same job: the outer WHERE re-checks the claimable condition.
Leases expire; a job whose worker died is reclaimed after its lease. Failed
attempts are retried with backoff up to ``max_attempts``.
"""

from __future__ import annotations

import json
from datetime import timedelta

from sqlalchemy import DateTime, bindparam, select, text
from sqlalchemy.exc import IntegrityError

from api.db import Job, dumps, transaction, utcnow

LEASE_SECONDS = 60

_CLAIM_SQL = text("""
UPDATE jobs SET status='running', lease_owner=:worker, lease_expires_at=:lease,
       heartbeat_at=:now, attempts=attempts+1
WHERE id = (
    SELECT id FROM jobs
    WHERE (status='queued' AND run_after <= :now)
       OR (status='running' AND lease_expires_at < :now)
    ORDER BY id LIMIT 1
)
AND ((status='queued' AND run_after <= :now) OR (status='running' AND lease_expires_at < :now))
RETURNING id
""").bindparams(bindparam("lease", type_=DateTime(timezone=True)),
                bindparam("now", type_=DateTime(timezone=True)))


def enqueue(kind: str, payload: dict, dedupe_key: str | None = None, max_attempts: int = 5,
            session=None) -> int:
    """Insert a job; with ``dedupe_key`` an existing job is returned instead."""
    def _do(s):
        if dedupe_key:
            existing = s.scalar(select(Job).where(Job.dedupe_key == dedupe_key))
            if existing:
                return existing.id
        job = Job(kind=kind, payload_json=dumps(payload), dedupe_key=dedupe_key, max_attempts=max_attempts)
        s.add(job)
        s.flush()
        return job.id

    if session is not None:
        return _do(session)
    try:
        with transaction() as s:
            return _do(s)
    except IntegrityError:  # concurrent insert with the same dedupe key
        with transaction() as s:
            return s.scalar(select(Job.id).where(Job.dedupe_key == dedupe_key))


def claim(worker_id: str) -> Job | None:
    now = utcnow()
    with transaction() as s:
        row = s.execute(_CLAIM_SQL, {"worker": worker_id, "lease": now + timedelta(seconds=LEASE_SECONDS),
                                     "now": now}).first()
        if not row:
            return None
        job = s.get(Job, row[0])
        if job.attempts > job.max_attempts:
            job.status = "failed"
            job.last_error = (job.last_error or "") + " | exceeded max attempts (lease expiries)"
            job.finished_at = now
            return None
        return job


def heartbeat(job_id: int, worker_id: str) -> None:
    now = utcnow()
    with transaction() as s:
        job = s.get(Job, job_id)
        if job and job.lease_owner == worker_id and job.status == "running":
            job.heartbeat_at = now
            job.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)


def complete(job_id: int, worker_id: str, result: dict | None = None) -> None:
    with transaction() as s:
        job = s.get(Job, job_id)
        if job.lease_owner != worker_id:
            return  # lease lost; another worker owns the outcome
        job.status = "done"
        job.result_json = dumps(result or {})
        job.finished_at = utcnow()


def fail(job_id: int, worker_id: str, error: str, retryable: bool = True) -> None:
    with transaction() as s:
        job = s.get(Job, job_id)
        if job.lease_owner != worker_id:
            return
        job.last_error = error[-2000:]
        if retryable and job.attempts < job.max_attempts:
            job.status = "queued"
            job.run_after = utcnow() + timedelta(seconds=min(2 ** job.attempts, 60))
            job.lease_owner = None
        else:
            job.status = "failed"
            job.finished_at = utcnow()


def payload(job: Job) -> dict:
    return json.loads(job.payload_json)
