"""Frozen, persisted split design.

Hospital A records are stratified by ever-positive status into
train/validation/calibration/test (60/15/10/15 by default) with a fixed
seed. All hospital B records form the external test. Each record (stay) is
assigned to exactly one partition; the challenge files expose no person
linkage, which is documented as a limitation in the split file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

A_PARTITIONS = ("a_train", "a_validation", "a_calibration", "a_test")
DEFAULT_FRACTIONS = {"a_train": 0.60, "a_validation": 0.15, "a_calibration": 0.10, "a_test": 0.15}
ALL_SPLITS = A_PARTITIONS + ("b_external",)
HELD_OUT_SPLITS = ("a_test", "b_external")


def make_splits(encounters: pd.DataFrame, seed: int = 2019,
                fractions: dict[str, float] | None = None) -> dict:
    fractions = fractions or DEFAULT_FRACTIONS
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("A fractions must sum to 1")
    rng = np.random.default_rng(seed)
    assign: dict[str, list[str]] = {k: [] for k in ALL_SPLITS}
    a = encounters[encounters["hospital_key"] == "A"]
    for _, stratum in a.groupby("ever_positive"):
        ids = np.sort(stratum["stay_id"].to_numpy())
        rng.shuffle(ids)
        n = len(ids)
        cuts = np.floor(np.cumsum([fractions[k] for k in A_PARTITIONS]) * n + 0.5).astype(int)
        start = 0
        for k, end in zip(A_PARTITIONS, cuts):
            assign[k] += ids[start:end].tolist()
            start = end
    assign["b_external"] = sorted(encounters.loc[encounters["hospital_key"] == "B", "stay_id"])
    for k in assign:
        assign[k] = sorted(assign[k])
    split = {
        "seed": seed,
        "fractions": fractions,
        "stratify_by": "ever_positive",
        "grouping": "record (stay); PhysioNet 2019 exposes no cross-admission person linkage",
        "partitions": assign,
        "counts": {k: len(v) for k, v in assign.items()},
    }
    check_disjoint(split)
    split["split_hash"] = split_hash(split)
    return split


def split_hash(split: dict) -> str:
    payload = json.dumps({k: split["partitions"][k] for k in sorted(split["partitions"])},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def check_disjoint(split: dict) -> None:
    seen: dict[str, str] = {}
    for k, ids in split["partitions"].items():
        for i in ids:
            if i in seen:
                raise AssertionError(f"{i} in both {seen[i]} and {k}")
            seen[i] = k


def save_splits(split: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(split, indent=1))


def load_splits(path: Path) -> dict:
    split = json.loads(Path(path).read_text())
    if split_hash(split) != split.get("split_hash"):
        raise ValueError(f"split file {path} does not match its recorded hash")
    return split


def partition_of(split: dict) -> dict[str, str]:
    return {i: k for k, ids in split["partitions"].items() for i in ids}
