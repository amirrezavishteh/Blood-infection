"""PhysioNet/CinC 2019 source schema and PSV parsing/validation.

Column definitions follow the challenge data description [S1]. Source names
and units are retained as supplied; ``Gender`` is the source field (0/1) and
is not reinterpreted, and ``Age`` is used exactly as provided (top-coded).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = [
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
CLINICAL = VITALS + LABS  # 34 time-varying clinical measurements
DEMOGRAPHICS = ["Age", "Gender"]
ADMINISTRATIVE = ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
COLUMNS = CLINICAL + DEMOGRAPHICS + ADMINISTRATIVE + [LABEL]  # 41 source columns
assert len(COLUMNS) == 41

# Units as documented by the challenge (source units; no conversion applied).
UNITS = {
    "HR": "beats/min", "O2Sat": "%", "Temp": "deg C", "SBP": "mm Hg",
    "MAP": "mm Hg", "DBP": "mm Hg", "Resp": "breaths/min", "EtCO2": "mm Hg",
    "BaseExcess": "mmol/L", "HCO3": "mmol/L", "FiO2": "fraction", "pH": "pH",
    "PaCO2": "mm Hg", "SaO2": "%", "AST": "IU/L", "BUN": "mg/dL",
    "Alkalinephos": "IU/L", "Calcium": "mg/dL", "Chloride": "mmol/L",
    "Creatinine": "mg/dL", "Bilirubin_direct": "mg/dL", "Glucose": "mg/dL",
    "Lactate": "mg/dL", "Magnesium": "mmol/dL", "Phosphate": "mg/dL",
    "Potassium": "mmol/L", "Bilirubin_total": "mg/dL", "TroponinI": "ng/mL",
    "Hct": "%", "Hgb": "g/dL", "PTT": "seconds", "WBC": "count*10^3/uL",
    "Fibrinogen": "mg/dL", "Platelets": "count*10^3/uL", "Age": "years",
    "Gender": "source code (0/1)", "Unit1": "indicator", "Unit2": "indicator",
    "HospAdmTime": "hours", "ICULOS": "hours",
}

HOSPITALS = {"A": "training_setA", "B": "training_setB"}
EXPECTED_COUNTS = {"A": 20336, "B": 20000}
MAX_FILE_BYTES = 2_000_000  # largest real file is ~120 KB; reject anything absurd


class PSVValidationError(ValueError):
    pass


@dataclass
class FileCheck:
    record_id: str
    rows: int = 0
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    ever_positive: bool = False
    first_row_positive: bool = False


def looks_like_html(head: bytes) -> bool:
    h = head.lstrip()[:64].lower()
    return h.startswith(b"<!doctype") or h.startswith(b"<html") or b"<head" in h


def parse_psv_bytes(raw: bytes, record_id: str = "?") -> pd.DataFrame:
    """Parse and validate one challenge PSV file. Raises PSVValidationError."""
    if not raw:
        raise PSVValidationError(f"{record_id}: empty file")
    if looks_like_html(raw[:512]):
        raise PSVValidationError(f"{record_id}: HTML payload, not a PSV file")
    text = raw.decode("ascii", errors="strict")
    lines = text.strip().splitlines()
    header = lines[0].strip().split("|")
    if header != COLUMNS:
        raise PSVValidationError(f"{record_id}: unexpected header ({len(header)} columns)")
    if len(lines) < 2:
        raise PSVValidationError(f"{record_id}: no data rows")
    try:
        arr = np.array([ln.split("|") for ln in lines[1:]], dtype=np.float64)
    except ValueError as exc:  # ragged rows or non-numeric tokens
        raise PSVValidationError(f"{record_id}: malformed rows ({exc})") from exc
    if arr.ndim != 2 or arr.shape[1] != len(COLUMNS):
        raise PSVValidationError(f"{record_id}: malformed rows (expected {len(COLUMNS)} columns)")
    df = pd.DataFrame(arr, columns=COLUMNS)
    validate_frame(df, record_id)
    return df


def validate_frame(df: pd.DataFrame, record_id: str = "?") -> None:
    lab = df[LABEL]
    if lab.isna().any() or not lab.isin([0, 1]).all():
        raise PSVValidationError(f"{record_id}: SepsisLabel must be 0/1 with no missing values")
    iculos = df["ICULOS"].to_numpy()
    if np.isnan(iculos).any():
        raise PSVValidationError(f"{record_id}: missing ICULOS")
    if len(iculos) > 1 and not np.all(np.diff(iculos) == 1):
        raise PSVValidationError(f"{record_id}: ICULOS is not strictly hourly")
    # The challenge label, once positive, stays positive for the record.
    if lab.any():
        first = int(np.argmax(lab.to_numpy()))
        if not (lab.to_numpy()[first:] == 1).all():
            raise PSVValidationError(f"{record_id}: SepsisLabel reverts to 0 after onset")


def read_psv(path: Path) -> pd.DataFrame:
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PSVValidationError(f"{path.name}: file too large ({size} bytes)")
    return parse_psv_bytes(path.read_bytes(), path.stem)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stay_id(hospital: str, record: str) -> str:
    """Namespaced stay identifier, e.g. ``physionet2019:A:p000001``."""
    return f"physionet2019:{hospital}:{record}"
