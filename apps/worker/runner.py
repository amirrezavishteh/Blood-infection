"""Background worker: claims jobs, heartbeats leases, drives playing replays.

Run: ``python -m worker`` (``--once`` processes available work then exits).
Training is deliberately not a worker job; it runs through the CLI.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from datetime import timedelta

from sqlalchemy import select

from api import jobs
from api.db import Job, Replay, WorkerHeartbeat, transaction, utcnow
from worker.replay import OutOfOrderStep, aware, process_step, validate_dataset

log = logging.getLogger("worker")
_stop = threading.Event()


def worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"


def beat(wid: str) -> None:
    with transaction() as s:
        hb = s.get(WorkerHeartbeat, wid)
        if hb is None:
            s.add(WorkerHeartbeat(worker_id=wid, pid=os.getpid(), last_seen=utcnow()))
        else:
            hb.last_seen = utcnow()


def tick_playing_replays() -> int:
    """Enqueue the next step for playing replays whose tick interval elapsed."""
    n = 0
    now = utcnow()
    with transaction() as s:
        playing = s.scalars(select(Replay).where(Replay.status == "playing")).all()
        for r in playing:
            if r.clock >= r.max_hour:
                r.status = "finished"
                continue
            last = aware(r.last_step_at)
            if last is not None and now - last < timedelta(seconds=r.tick_seconds):
                continue
            jobs.enqueue("replay_step", {"replay_id": r.id, "target_hour": r.clock + 1},
                         dedupe_key=f"replay:{r.id}:step:{r.clock + 1}", session=s)
            n += 1
    return n


def run_job(job: Job, wid: str) -> None:
    p = jobs.payload(job)
    stop_hb = threading.Event()

    def _hb():
        while not stop_hb.wait(jobs.LEASE_SECONDS / 3):
            jobs.heartbeat(job.id, wid)

    t = threading.Thread(target=_hb, daemon=True)
    t.start()
    try:
        if job.kind == "replay_step":
            res = process_step(p["replay_id"], int(p["target_hour"]))
        elif job.kind == "dataset_validate":
            res = validate_dataset(int(p["dataset_id"]))
        else:
            raise ValueError(f"unknown job kind {job.kind}")
        jobs.complete(job.id, wid, res)
    except OutOfOrderStep as exc:
        jobs.fail(job.id, wid, str(exc), retryable=True)
    except Exception as exc:
        log.error("job %s failed: %s", job.id, exc)
        jobs.fail(job.id, wid, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
                  retryable=job.kind != "dataset_validate")
    finally:
        stop_hb.set()


def run(poll: float = 0.25, once: bool = False, wid: str | None = None) -> None:
    wid = wid or worker_id()
    log.info("worker %s started", wid)
    last_beat = 0.0
    while not _stop.is_set():
        if time.time() - last_beat > 5:
            beat(wid)
            last_beat = time.time()
        tick_playing_replays()
        job = jobs.claim(wid)
        if job is None:
            if once:
                return
            _stop.wait(poll)
            continue
        run_job(job, wid)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m worker")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=float, default=0.25)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: _stop.set())
    run(poll=a.poll, once=a.once)
