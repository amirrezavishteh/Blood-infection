"""Synthetic PhysioNet-2019-format fixtures for tests, smoke runs and public demos.

The timelines are entirely simulated (no patient data). Septic records drift
toward tachycardia, fever, hypotension, tachypnoea and rising lactate/WBC
before a simulated onset; the challenge-style label turns on six hours before
that onset and stays on. Some records are positive from their first row
(left-censored), mirroring the real corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sepsis.data.schema import COLUMNS, HOSPITALS, LABS

_BASE = {"HR": 85, "O2Sat": 97, "Temp": 36.9, "SBP": 122, "MAP": 82, "DBP": 64, "Resp": 18}
_NOISE_SD = {"HR": 7, "O2Sat": 1.5, "Temp": 0.35, "SBP": 11, "MAP": 8, "DBP": 7, "Resp": 2.8}
_SEPSIS_DRIFT = {"HR": 28, "O2Sat": -4, "Temp": 1.6, "SBP": -25, "MAP": -18, "DBP": -12, "Resp": 9}
_LAB_BASE = {lab: (1.0, 0.1) for lab in LABS}
_LAB_BASE.update({
    "Lactate": (1.4, 0.4), "WBC": (9.0, 2.0), "Creatinine": (1.0, 0.25), "Platelets": (220, 50),
    "BUN": (18, 5), "Glucose": (125, 25), "pH": (7.39, 0.03), "HCO3": (24, 2), "FiO2": (0.4, 0.1),
    "Hgb": (11, 1.5), "Hct": (33, 4), "Potassium": (4.1, 0.4), "Chloride": (104, 3),
    "Bilirubin_total": (0.8, 0.3), "PaCO2": (40, 4), "BaseExcess": (0, 2), "SaO2": (96, 2),
})
_LAB_DRIFT = {"Lactate": 2.6, "WBC": 7.0, "Creatinine": 0.9, "Platelets": -80, "pH": -0.08,
              "HCO3": -5, "Bilirubin_total": 1.2}


def simulate_record(rng: np.random.Generator, septic: bool, left_censored: bool = False) -> np.ndarray:
    n = int(rng.integers(12, 90))
    if septic:
        onset = 0 if left_censored else int(rng.integers(8, n + 6))  # onset may be after discharge
        label_start = max(0, onset - 6)
    else:
        onset, label_start = None, n
    t = np.arange(n)
    out = np.full((n, len(COLUMNS)), np.nan)
    col = {c: i for i, c in enumerate(COLUMNS)}
    strength = float(rng.uniform(0.15, 1.0))  # some septic courses are physiologically subtle
    if septic:
        ramp = strength * (np.clip((t - (onset - 14)) / 14.0, 0, 1.3) if not left_censored else np.ones(n))
    else:
        # non-septic deterioration episodes create realistic false positives
        ramp = np.zeros(n)
        if rng.random() < 0.25:
            s0 = int(rng.integers(0, n))
            ramp[s0:] = np.clip((t[s0:] - s0) / 10.0, 0, 1) * float(rng.uniform(0.2, 0.7))
    for v, base in _BASE.items():
        series = base + rng.normal(0, _NOISE_SD[v], n) + ramp * _SEPSIS_DRIFT[v]
        miss = rng.random(n) < (0.12 if v != "Temp" else 0.6)
        series[miss] = np.nan
        out[:, col[v]] = np.round(series, 2)
    for lab, (mu, sd) in _LAB_BASE.items():
        if lab == "EtCO2":
            continue  # absent in this simulated unit, as it is for most real records
        period = int(rng.integers(8, 25))
        draw_at = np.arange(int(rng.integers(0, period)), n, period)
        vals = mu + rng.normal(0, sd, len(draw_at)) + ramp[draw_at] * _LAB_DRIFT.get(lab, 0.0)
        out[draw_at, col[lab]] = np.round(vals, 3)
    out[:, col["Age"]] = float(rng.integers(18, 90))
    out[:, col["Gender"]] = float(rng.integers(0, 2))
    unit = int(rng.integers(0, 3))
    out[:, col["Unit1"]] = np.nan if unit == 2 else float(unit == 0)
    out[:, col["Unit2"]] = np.nan if unit == 2 else float(unit == 1)
    out[:, col["HospAdmTime"]] = -round(float(rng.exponential(10)), 2)
    out[:, col["ICULOS"]] = t + 1
    lab = np.zeros(n)
    lab[label_start:] = 1
    out[:, col["SepsisLabel"]] = lab
    return out


def _fmt(x: float) -> str:
    if np.isnan(x):
        return "NaN"
    return f"{x:g}"


def write_psv(arr: np.ndarray, path: Path) -> None:
    lines = ["|".join(COLUMNS)] + ["|".join(_fmt(v) for v in row) for row in arr]
    Path(path).write_text("\n".join(lines) + "\n")


def make_fixture(out_root: Path, n_a: int = 120, n_b: int = 60, septic_rate: float = 0.25,
                 seed: int = 7) -> Path:
    """Write a small synthetic corpus laid out like the real raw directory."""
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    offset = {"A": 0, "B": 100000}
    for h, n in (("A", n_a), ("B", n_b)):
        d = out_root / HOSPITALS[h]
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            septic = rng.random() < septic_rate
            left = septic and rng.random() < 0.08
            write_psv(simulate_record(rng, septic, left), d / f"p{offset[h] + i:06d}.psv")
    (out_root / "manifest.json").write_text(json.dumps({
        "dataset": "physionet2019-format synthetic fixture", "synthetic": True, "seed": seed,
        "hospitals": {"A": {"present_files": n_a}, "B": {"present_files": n_b}},
        "note": "simulated timelines; not patient data; metrics on it say nothing about real performance"}, indent=1))
    return out_root
