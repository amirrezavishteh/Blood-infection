"""Project-relative paths with a single documented data-directory override.

``SEPSIS_DATA_DIR`` relocates ``data/`` (raw + processed research data).
``SEPSIS_ARTIFACT_DIR`` relocates ``artifacts/`` (model bundles, runs, reports).
Everything else resolves relative to the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("SEPSIS_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def artifact_dir() -> Path:
    return Path(os.environ.get("SEPSIS_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")).resolve()


def raw_dir(dataset: str) -> Path:
    return data_dir() / "raw" / dataset


def processed_dir(dataset: str) -> Path:
    return data_dir() / "processed" / dataset


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve ``path`` and refuse it unless it lies inside ``root``."""
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"{resolved} is outside the trusted directory {root}")
    return resolved
