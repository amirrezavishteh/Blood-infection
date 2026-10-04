"""Resumable, verified downloader for the open PhysioNet 2019 challenge files.

Works on native Windows (no wget needed). For each hospital directory it reads
the server listing (names + byte sizes), downloads missing/incomplete files
with modest concurrency and exponential backoff, rejects HTML or oversized
payloads, validates the PSV schema, writes atomically, and records a manifest
with byte counts and SHA-256 checksums. It fails loudly unless the expected
20,336 (A) and 20,000 (B) files are present and valid.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from sepsis.data.schema import (
    EXPECTED_COUNTS,
    HOSPITALS,
    MAX_FILE_BYTES,
    PSVValidationError,
    looks_like_html,
    parse_psv_bytes,
    sha256_file,
)
from sepsis.paths import raw_dir

log = logging.getLogger(__name__)

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/"
PROJECT_URL = "https://physionet.org/content/challenge-2019/1.0.0/"
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
CITATION = (
    "Reyna MA, Josef CS, Jeter R, Shashikumar SP, Westover MB, Nemati S, Clifford GD, "
    "Sharma A. Early Prediction of Sepsis From Clinical Data: the PhysioNet/Computing in "
    "Cardiology Challenge 2019. Critical Care Medicine 48(2):210-217 (2020). "
    "Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23) (2000)."
)
USER_AGENT = "sepsis-research-downloader/0.1 (+research use)"
_LISTING_RE = re.compile(r'<a href="(p\d{6}\.psv)">[^<]*</a>\s+\S+\s+\S+\s+(\d+)')


class DownloadError(RuntimeError):
    pass


_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _local.s = s
    return _local.s


def fetch_listing(hospital: str, base_url: str = BASE_URL, timeout: float = 60) -> dict[str, int]:
    url = f"{base_url}{HOSPITALS[hospital]}/"
    r = _session().get(url, timeout=timeout)
    r.raise_for_status()
    entries = {name: int(size) for name, size in _LISTING_RE.findall(r.text)}
    if not entries:
        raise DownloadError(f"could not parse directory listing at {url}")
    return entries


def _get_with_retries(url: str, retries: int, timeout: float) -> bytes:
    delay = 1.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            deadline = time.monotonic() + timeout  # total per-file budget (stalled/trickling connections)
            with _session().get(url, timeout=(10, min(timeout, 30)), stream=True) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise DownloadError(f"HTTP {r.status_code}")
                r.raise_for_status()
                chunks, total = [], 0
                for chunk in r.iter_content(65536):
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise PSVValidationError(f"{url}: exceeds {MAX_FILE_BYTES} bytes")
                    if time.monotonic() > deadline:
                        raise DownloadError(f"{url}: exceeded {timeout}s total transfer time")
                    chunks.append(chunk)
                return b"".join(chunks)
        except PSVValidationError:
            raise
        except Exception as exc:  # network errors, retryable statuses
            last = exc
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise DownloadError(f"{url}: failed after {retries + 1} attempts ({last})")


def _download_one(hospital: str, name: str, expected_size: int, dest_dir: Path,
                  base_url: str, retries: int, timeout: float) -> tuple[str, str]:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size == expected_size:
        return name, "skipped"
    raw = _get_with_retries(f"{base_url}{HOSPITALS[hospital]}/{name}", retries, timeout)
    if looks_like_html(raw[:512]):
        raise PSVValidationError(f"{name}: server returned HTML instead of PSV")
    if len(raw) != expected_size:
        raise DownloadError(f"{name}: got {len(raw)} bytes, listing says {expected_size}")
    parse_psv_bytes(raw, name)  # schema validation before the file is accepted
    tmp = dest.with_suffix(".psv.part")
    tmp.write_bytes(raw)
    tmp.replace(dest)  # atomic on the same volume
    return name, "downloaded"


def download_physionet2019(hospitals=("A", "B"), workers: int = 8, retries: int = 5,
                           timeout: float = 60, base_url: str = BASE_URL,
                           out_root: Path | None = None, limit: int | None = None) -> dict:
    """Download and verify; returns the manifest dict (also written to disk)."""
    out_root = Path(out_root) if out_root else raw_dir("physionet2019")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "dataset": "physionet2019",
        "dataset_version": "1.0.0",
        "source_url": base_url,
        "project_url": PROJECT_URL,
        "license": LICENSE,
        "citation": CITATION,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hospitals": {},
    }
    failures: list[str] = []
    for h in hospitals:
        listing = fetch_listing(h, base_url, timeout)
        names = sorted(listing)
        if limit is not None:
            names = names[:limit]
        dest_dir = out_root / HOSPITALS[h]
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("hospital %s: %d files listed", h, len(listing))
        counts = {"downloaded": 0, "skipped": 0}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, h, n, listing[n], dest_dir, base_url, retries, timeout): n
                    for n in names}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    _, status = fut.result()
                    counts[status] += 1
                except Exception as exc:
                    failures.append(f"{h}/{futs[fut]}: {exc}")
                if i % 2000 == 0:
                    log.info("hospital %s: %d/%d (%.0fs)", h, i, len(names), time.time() - t0)
        files = {}
        for n in names:
            p = dest_dir / n
            if p.exists():
                files[n] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        manifest["hospitals"][h] = {
            "directory": HOSPITALS[h],
            "listed_files": len(listing),
            "expected_files": EXPECTED_COUNTS[h],
            "present_files": len(files),
            "total_bytes": sum(f["bytes"] for f in files.values()),
            "transfer": counts,
            "files": files,
        }
    manifest["failures"] = failures
    write_manifest(manifest, out_root)
    if failures:
        raise DownloadError(f"{len(failures)} files failed; first: {failures[:3]}")
    if limit is None:
        for h in hospitals:
            got = manifest["hospitals"][h]["present_files"]
            if got != EXPECTED_COUNTS[h]:
                raise DownloadError(f"hospital {h}: {got} files, expected {EXPECTED_COUNTS[h]}")
    return manifest


def write_manifest(manifest: dict, out_root: Path) -> Path:
    path = Path(out_root) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path
