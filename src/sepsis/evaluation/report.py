"""Experiment comparison report across runs (consumed by the dashboard)."""

from __future__ import annotations

import json
from pathlib import Path

from sepsis.models.bundle import bundle_root


def json_safe(o):
    """Replace NaN/inf with None recursively (strict JSON for the API/dashboard)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def collect_runs() -> list[dict]:
    root = bundle_root()
    out = []
    if not root.exists():
        return out
    for rd in sorted(root.iterdir()):
        bj = rd / "bundle.json"
        if not bj.exists():
            continue
        meta = json.loads(bj.read_text())
        entry = {
            "run_id": rd.name,
            "model_kind": meta["model_kind"],
            "target_id": meta["target_id"],
            "feature_set": meta["feature_config"]["feature_set"],
            "feature_schema_version": meta["feature_schema_version"],
            "calibrated": bool(meta.get("calibration")),
            "alert_policy": meta.get("alert_policy"),
            "benchmark_threshold": meta.get("benchmark_threshold"),
            "validation_auprc": max((t["val_auprc"] for t in meta.get("tuning_trials", [])), default=None),
            "provenance": {k: meta["provenance"].get(k) for k in ("split_hash", "code_revision", "created_at",
                                                                  "raw_manifest_sha256", "train_rows")},
            "evaluations": {},
        }
        for ev in sorted((rd / "eval").glob("*/metrics.json")) if (rd / "eval").exists() else []:
            entry["evaluations"][ev.parent.name] = json.loads(ev.read_text())
        ps = rd / "policy_search.json"
        if ps.exists():
            pol = json.loads(ps.read_text())
            entry["policy_search"] = {"by_budget": pol["by_budget"], "primary_budget": pol.get("primary_budget")}
        out.append(json_safe(entry))
    return out


def build_experiments_report() -> Path:
    runs = collect_runs()
    path = bundle_root().parent / "reports" / "experiments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": runs}, indent=1, default=str))
    return path
