"""Access to the processed research store (Parquet) with a feature cache.

Labels live in ``outcomes.parquet`` and are only joined for training and
evaluation; scoring paths read ``hourly.parquet`` alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.data.splits import load_splits, partition_of
from sepsis.features.causal import FeatureConfig, compute_features, schema_version
from sepsis.paths import processed_dir

log = logging.getLogger(__name__)
SCORING_DROP = ["hospital", "record"]


@dataclass
class Store:
    root: Path

    @classmethod
    def for_dataset(cls, dataset: str = "physionet2019", root: Path | None = None) -> "Store":
        return cls(Path(root) if root else processed_dir(dataset))

    @cached_property
    def hourly(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "hourly.parquet")

    @cached_property
    def outcomes(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "outcomes.parquet")

    @cached_property
    def encounters(self) -> pd.DataFrame:
        return pd.read_parquet(self.root / "encounters.parquet")

    @cached_property
    def quality(self) -> dict:
        return json.loads((self.root / "quality.json").read_text())

    @property
    def splits_path(self) -> Path:
        return self.root / "splits.json"

    @cached_property
    def splits(self) -> dict:
        return load_splits(self.splits_path)

    @cached_property
    def partition(self) -> dict[str, str]:
        return partition_of(self.splits)

    def stays(self, split: str) -> list[str]:
        return list(self.splits["partitions"][split])

    def features(self, cfg: FeatureConfig, chunk_stays: int = 3000) -> pd.DataFrame:
        """Causal features for every stay, cached per feature-schema version."""
        version = schema_version(cfg)
        path = self.root / "features" / f"{version}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        src = self.hourly.drop(columns=SCORING_DROP)
        ids = src["stay_id"].unique()
        parts = []
        for i in range(0, len(ids), chunk_stays):  # stays are independent: chunking is exact
            chunk = src[src["stay_id"].isin(set(ids[i:i + chunk_stays]))]
            parts.append(compute_features(chunk, cfg))
            log.info("features %s: %d/%d stays", version, min(i + chunk_stays, len(ids)), len(ids))
        feats = pd.concat(parts, ignore_index=True)
        tmp = path.with_suffix(".parquet.tmp")
        feats.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic: an interrupted build never leaves a truncated cache
        (path.with_suffix(".json")).write_text(json.dumps({"schema_version": version,
                                                            "config": cfg.to_dict()}, indent=1))
        return feats

    def matrix(self, cfg: FeatureConfig, split: str, names: list[str]):
        """(X, y, frame) for a split; frame holds stay_id/hour_index/label."""
        feats = self.features(cfg)
        ids = set(self.stays(split))
        f = feats[feats["stay_id"].isin(ids)]
        lab = self.outcomes[["stay_id", "hour_index", "label"]]
        f = f.merge(lab, on=["stay_id", "hour_index"], how="left", validate="one_to_one")
        if f["label"].isna().any():
            raise RuntimeError("feature rows without outcome labels")
        f = f.sort_values(["stay_id", "hour_index"], kind="stable").reset_index(drop=True)
        X = f[names].to_numpy(dtype=np.float32)
        y = f["label"].to_numpy(dtype=np.int8)
        return X, y, f[["stay_id", "hour_index", "label"]]
