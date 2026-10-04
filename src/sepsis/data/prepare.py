"""Convert validated raw PSV files into the canonical Parquet research store.

Outputs (under ``data/processed/<dataset>/``):

* ``hourly.parquet``     stay_id, hospital, record, hour_index + 40 source inputs (no label)
* ``outcomes.parquet``   stay_id, target_id, hour_index, label, label_version
* ``encounters.parquet`` dataset, dataset_version, stay_id, hospital_key, n_hours, ...
* ``quality.json``       counts, schema, per-hospital missingness, invalid files

Labels are written to a separate table so feature/scoring code never sees them.
The challenge source is a wide hourly table; it is kept wide rather than
exploded to long-form observations (the PSV gives no per-measurement
availability time; the end of each hourly bin is the availability boundary).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.data.schema import (
    ADMINISTRATIVE,
    CLINICAL,
    COLUMNS,
    DEMOGRAPHICS,
    EXPECTED_COUNTS,
    HOSPITALS,
    LABEL,
    PSVValidationError,
    read_psv,
    sha256_file,
    stay_id,
)
from sepsis.paths import processed_dir, raw_dir

log = logging.getLogger(__name__)

TARGET_ID = "challenge2019_state"
LABEL_VERSION = "challenge2019_state@1"  # supplied SepsisLabel, unshifted
INPUT_COLUMNS = CLINICAL + DEMOGRAPHICS + ADMINISTRATIVE


def _load_one(args):
    hospital, path = args
    try:
        df = read_psv(Path(path))
    except (PSVValidationError, UnicodeDecodeError, OSError) as exc:
        return hospital, Path(path).stem, None, str(exc)
    return hospital, Path(path).stem, df, None


def load_raw(raw_root: Path, hospitals=("A", "B"), workers: int = 8):
    jobs = []
    for h in hospitals:
        d = Path(raw_root) / HOSPITALS[h]
        files = sorted(d.glob("p*.psv"))
        jobs += [(h, str(f)) for f in files]
    frames, errors = [], []
    if workers > 1 and len(jobs) > 200:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_load_one, jobs, chunksize=256))
    else:
        results = [_load_one(j) for j in jobs]
    for h, rec, df, err in results:
        if err:
            errors.append({"hospital": h, "record": rec, "error": err})
            continue
        df.insert(0, "hour_index", np.arange(len(df), dtype=np.int32))
        df.insert(0, "record", rec)
        df.insert(0, "hospital", h)
        df.insert(0, "stay_id", stay_id(h, rec))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return data, errors


def prepare(dataset: str = "physionet2019", raw_root: Path | None = None,
            out_dir: Path | None = None, hospitals=("A", "B"), workers: int = 8,
            require_expected_counts: bool = True) -> dict:
    raw_root = Path(raw_root) if raw_root else raw_dir(dataset)
    out_dir = Path(out_dir) if out_dir else processed_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, errors = load_raw(raw_root, hospitals, workers)
    if data.empty:
        raise RuntimeError(f"no valid PSV files found under {raw_root}")
    if data.duplicated(["stay_id", "hour_index"]).any():
        raise RuntimeError("duplicate (stay_id, hour_index) rows")

    hourly = data[["stay_id", "hospital", "record", "hour_index"] + INPUT_COLUMNS].copy()
    for c in INPUT_COLUMNS:
        hourly[c] = hourly[c].astype("float32")
    outcomes = pd.DataFrame({
        "stay_id": data["stay_id"],
        "target_id": TARGET_ID,
        "hour_index": data["hour_index"],
        "label": data[LABEL].astype("int8"),
        "label_version": LABEL_VERSION,
    })
    g = data.groupby("stay_id", sort=False)
    encounters = pd.DataFrame({
        "dataset": dataset,
        "dataset_version": "1.0.0",
        "stay_id": g.size().index,
        "patient_key_if_available": None,  # challenge files expose no person linkage
        "hospital_key": g["hospital"].first().to_numpy(),
        "record": g["record"].first().to_numpy(),
        "n_hours": g.size().to_numpy(),
        "admission_offset_h": g["HospAdmTime"].first().to_numpy(),
        "first_iculos": g["ICULOS"].first().to_numpy(),
        "ever_positive": g[LABEL].max().astype(bool).to_numpy(),
        "first_row_positive": g[LABEL].first().astype(bool).to_numpy(),
    })

    for name, frame in (("hourly", hourly), ("outcomes", outcomes), ("encounters", encounters)):
        tmp = out_dir / f"{name}.parquet.tmp"
        frame.to_parquet(tmp, index=False)
        tmp.replace(out_dir / f"{name}.parquet")

    quality = quality_report(hourly, outcomes, encounters, errors)
    manifest_path = Path(raw_root) / "manifest.json"
    quality["raw_manifest_sha256"] = sha256_file(manifest_path) if manifest_path.exists() else None
    quality["synthetic"] = bool(manifest_path.exists() and json.loads(manifest_path.read_text()).get("synthetic"))
    quality["outputs"] = {p: sha256_file(out_dir / p) for p in
                          ("hourly.parquet", "outcomes.parquet", "encounters.parquet")}
    (out_dir / "quality.json").write_text(json.dumps(quality, indent=1))

    if errors:
        raise RuntimeError(f"{len(errors)} invalid source files, e.g. {errors[:2]}")
    if require_expected_counts:
        for h in hospitals:
            got = quality["hospitals"][h]["stays"]
            if got != EXPECTED_COUNTS[h]:
                raise RuntimeError(f"hospital {h}: {got} records, expected {EXPECTED_COUNTS[h]}")
    return quality


def quality_report(hourly, outcomes, encounters, errors) -> dict:
    rep = {"schema": {"columns": COLUMNS, "n_columns": len(COLUMNS)},
           "invalid_files": errors, "hospitals": {}}
    lab = outcomes["label"].to_numpy()
    for h, sub in hourly.groupby("hospital"):
        mask = (hourly["hospital"] == h).to_numpy()
        enc = encounters[encounters["hospital_key"] == h]
        rep["hospitals"][h] = {
            "stays": int(len(enc)),
            "rows": int(len(sub)),
            "patient_days": float(len(sub) / 24.0),
            "positive_hours": int(lab[mask].sum()),
            "positive_hour_prevalence": float(lab[mask].mean()),
            "ever_positive_stays": int(enc["ever_positive"].sum()),
            "left_censored_positive_stays": int(enc["first_row_positive"].sum()),
            "median_hours": float(enc["n_hours"].median()),
            "max_hours": int(enc["n_hours"].max()),
            "missing_fraction": {c: float(sub[c].isna().mean()) for c in INPUT_COLUMNS},
        }
    return rep
