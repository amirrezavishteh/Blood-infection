"""Policy selection and frozen evaluation for a model run.

Order of operations enforced here:
train (a_train, early stopping on a_validation) -> calibrate (a_calibration)
-> select-policy (a_validation) -> evaluate (a_test, then b_external).
Evaluation refuses uncalibrated runs and runs without a frozen policy and
benchmark threshold, and every evaluation of ``b_external`` is logged so
repeated looks at the external hospital are visible.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.alerts.matching import evaluate_alerts
from sepsis.alerts.policy import AlertPolicy
from sepsis.alerts.search import alerts_for, prepare_stays, search
from sepsis.data.store import Store
from sepsis.evaluation.metrics import coverage, hourly_metrics, subgroup_metrics, utility_with_ci
from sepsis.evaluation.official import run_official
from sepsis.evaluation.utility import best_threshold
from sepsis.features.causal import partial_qsofa, partial_sirs
from sepsis.models.bundle import load_bundle, write_meta

log = logging.getLogger(__name__)
FALLBACK_PROBABILITY = 0.0  # fixed before testing: unavailable hours -> probability 0, label 0
EVALUATION_SPLITS = ("a_validation", "a_test", "b_external")


def scored_frame(bundle, store: Store, split: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Returns (scores with labels, feature frame, batch scoring seconds)."""
    feats = store.features(bundle.feature_config)
    ids = set(store.stays(split))
    f = feats[feats["stay_id"].isin(ids)].sort_values(["stay_id", "hour_index"]).reset_index(drop=True)
    t0 = time.perf_counter()
    s = bundle.score_frame(f[["stay_id", "hour_index"] + bundle.feature_names])
    elapsed = time.perf_counter() - t0
    lab = store.outcomes[["stay_id", "hour_index", "label"]]
    s = s.merge(lab, on=["stay_id", "hour_index"], how="left", validate="one_to_one")
    return s, f, elapsed


def select_policy(run_dir: Path, store: Store | None = None, split: str = "a_validation",
                  budgets=(1, 2, 5, 10, 20), primary_budget: float = 10,
                  consecutive=(1, 2, 3), cooldowns=(0, 6, 12)) -> dict:
    if split != "a_validation":
        raise ValueError("policy selection is only permitted on a_validation")
    b = load_bundle(run_dir)
    if not b.calibration:
        raise RuntimeError("calibrate the run before selecting a policy")
    store = store or Store.for_dataset(b.meta["provenance"]["dataset"])
    fr, _, _ = scored_frame(b, store, split)
    res = search(fr, budgets=budgets, consecutive=consecutive, cooldowns=cooldowns)
    primary = res["by_budget"].get(str(primary_budget))
    if not primary or not primary["feasible"]:
        raise RuntimeError(f"primary budget {primary_budget} infeasible on validation")
    bench = fr.copy()
    thr, util, curve = best_threshold(bench["stay_id"].to_numpy(), bench["label"].to_numpy(),
                                      bench["score"].to_numpy())
    res["benchmark_threshold"] = {"threshold": thr, "a_validation_utility": util, "curve": curve,
                                  "rule": "max normalised utility on a_validation hourly scores"}
    res["primary_budget"] = primary_budget
    res["selected_policy"] = {k: primary[k] for k in ("threshold", "consecutive", "cooldown_h", "policy_version")}
    (b.path / "policy_search.json").write_text(json.dumps(res, indent=1, default=float))
    meta = dict(b.meta)
    meta["alert_policy"] = {**res["selected_policy"], "selected_on": split,
                            "budget_alerts_per_100_patient_days": primary_budget}
    meta["benchmark_threshold"] = thr
    meta["fallback_probability"] = FALLBACK_PROBABILITY
    write_meta(b.path, meta)
    return res


def _log_access(run_dir: Path, split: str) -> int:
    path = Path(run_dir) / "evaluation_log.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append({"split": split, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    path.write_text(json.dumps(entries, indent=1))
    return sum(1 for e in entries if e["split"] == split)


def evaluate(run_dir: Path, split: str, store: Store | None = None, n_boot: int = 200,
             official: bool = True) -> dict:
    if split not in EVALUATION_SPLITS:
        raise ValueError(f"evaluation split must be one of {EVALUATION_SPLITS}")
    b = load_bundle(run_dir)
    if not b.calibration or not b.policy or b.meta.get("benchmark_threshold") is None:
        raise RuntimeError("run must be calibrated and have a frozen policy before evaluation")
    store = store or Store.for_dataset(b.meta["provenance"]["dataset"])
    if b.meta["provenance"]["split_hash"] != store.splits["split_hash"]:
        raise RuntimeError("split mismatch")
    n_looks = _log_access(run_dir, split)
    if split == "b_external" and n_looks > 1:
        log.warning("b_external has now been evaluated %d times for this run", n_looks)
    out_dir = Path(run_dir) / "eval" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fr, feats, scoring_seconds = scored_frame(b, store, split)
    thr = b.meta["benchmark_threshold"]
    fr["probability"] = fr["score"].fillna(FALLBACK_PROBABILITY)
    fr["prediction"] = ((fr["status"] == "scored") & (fr["score"] >= thr)).astype(int)
    n_fallback = int((fr["status"] != "scored").sum())
    # The official scorer is dominated by writing/reading one file per record (I/O releases the
    # GIL), so it runs on a background thread while the in-process metrics are computed.
    pool = ThreadPoolExecutor(max_workers=1) if official else None
    official_future = pool.submit(run_official, fr[["stay_id", "hour_index", "label", "probability", "prediction"]].copy(),
                                  out_dir / "official") if official else None

    policy = AlertPolicy.from_dict(b.policy)
    stays = prepare_stays(fr)
    alert_stays = alerts_for(policy, stays)
    alerts = evaluate_alerts(alert_stays)

    demo = store.hourly.groupby("stay_id", sort=False)[["Age", "Gender"]].first().reset_index()
    baselines = {}
    for name, fn in (("partial_qSOFA", partial_qsofa), ("partial_SIRS", partial_sirs)):
        base = feats[["stay_id", "hour_index"]].copy()
        base["score"] = fn(feats)
        base = base.merge(fr[["stay_id", "hour_index", "label"]], on=["stay_id", "hour_index"])
        m = hourly_metrics(base, n_boot=0, probabilistic=False)
        baselines[name] = {"auroc": m["auroc"], "auprc": m["auprc"],
                           "note": "rule score from latest values; GCS/culture data unavailable"}

    res = {
        "run": b.model_version,
        "split": split,
        "target_id": b.target_id,
        "model_kind": b.meta["model_kind"],
        "feature_set": b.feature_config.feature_set,
        "feature_schema_version": b.feature_schema_version,
        "split_hash": store.splits["split_hash"],
        "evaluation_number_for_split": n_looks,
        "records": int(fr["stay_id"].nunique()),
        "hourly": hourly_metrics(fr, n_boot=n_boot),
        "benchmark": {**utility_with_ci(fr, n_boot=n_boot), "threshold": thr,
                      "fallback_rows": n_fallback, "fallback_probability": FALLBACK_PROBABILITY},
        "alert_policy": b.policy,
        "alerts": alerts,
        "coverage": coverage(fr),
        "subgroups": subgroup_metrics(fr, demo, n_boot=min(n_boot, 100)),
        "baselines": baselines,
        "operational": {"batch_scoring_seconds": scoring_seconds,
                        "rows": int(len(fr)),
                        "rows_per_second": len(fr) / max(scoring_seconds, 1e-9),
                        "evaluation_seconds": None},
    }
    if official:
        res["official"] = official_future.result()
        pool.shutdown()
    res["operational"]["evaluation_seconds"] = round(time.perf_counter() - t0, 1)
    fr.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, default=_json_default))
    (out_dir / "report.md").write_text(render_report(res))
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]" if x else "n/a"


def render_report(r: dict) -> str:
    h, bm, a, c = r["hourly"], r["benchmark"], r["alerts"], r["coverage"]
    lines = [
        f"# Evaluation report - {r['split']}",
        "",
        f"Run `{r['run']}` ({r['model_kind']}, feature set `{r['feature_set']}`), target "
        f"`{r['target_id']}`, split hash `{r['split_hash']}`. Look #{r['evaluation_number_for_split']} at this split.",
        "",
        "Research sepsis score for the PhysioNet 2019 benchmark state; not a clinically validated probability.",
        "",
        "## Hourly discrimination and calibration",
        "",
        f"| Metric | Value | 95% CI (record bootstrap) |\n|---|---|---|",
        f"| AUROC | {h['auroc']:.3f} | {_ci(h.get('auroc_ci'))} |",
        f"| AUPRC | {h['auprc']:.3f} | {_ci(h.get('auprc_ci'))} |",
        f"| Positive-hour prevalence | {h['positive_hour_prevalence']:.4f} | |",
        f"| Brier score | {h['brier']:.4f} | |",
        "",
        "## Official challenge utility",
        "",
        f"Normalised utility {bm['utility']:.3f} {_ci(bm.get('utility_ci'))} at threshold {bm['threshold']:.4f}; "
        f"{bm['fallback_rows']} unavailable hours scored with fixed fallback p={bm['fallback_probability']}.",
    ]
    if "official" in r:
        o = r["official"]
        lines += ["", f"Official evaluator (commit {o['evaluator_commit'][:7]}): AUROC {o['AUROC']:.3f}, "
                      f"AUPRC {o['AUPRC']:.3f}, Accuracy {o['Accuracy']:.3f}, F-measure {o['F-measure']:.3f}, "
                      f"Utility {o['Utility']:.3f}."]
    lines += [
        "", "## Alert policy (application metrics)", "",
        f"Policy `{r['alert_policy']['policy_version']}`: threshold {r['alert_policy']['threshold']:.4f}, "
        f"{r['alert_policy']['consecutive']} consecutive h, cooldown {r['alert_policy']['cooldown_h']} h.",
        "",
        "| Measure | Value |\n|---|---|",
        f"| Onset-evaluable septic records | {a['septic_onset_evaluable_stays']} (left-censored excluded: {a['left_censored_positive_stays']}) |",
        f"| Early event sensitivity [onset-12h, onset) | {a['early_sensitivity']:.3f} ({a['detected_early']} detected, {a['missed']} missed) |",
        f"| At-least-6h warning [onset-12h, onset-6h] | {a['six_hour_warning_rate']:.3f} |",
        f"| Lead time median (IQR), detected | {a['lead_time_median_h']:.1f} h ({a['lead_time_iqr_h']}) |",
        f"| Alert precision (one-to-one, [onset-12h, onset+3h]) | {a['alert_precision']:.3f} (pre-onset {a['matched_pre_onset']}, late {a['matched_late']}) |",
        f"| Alerts per 100 patient-days | {a['alerts_per_100_patient_days']:.2f} |",
        f"| Unmatched alerts per 100 patient-days | {a['unmatched_per_100_patient_days']:.2f} |",
        f"| False alerts in non-septic records per 100 non-septic patient-days | {a['false_alerts_nonseptic_per_100_patient_days']:.2f} |",
        "", "## Coverage", "",
        f"{c['scored_hours']}/{c['scheduled_hours']} hours scored ({c['coverage']:.3f}); unavailable: {c['unavailable_by_reason']}.",
        "", "## Subgroups", "", "| Subgroup | Records | Septic | AUROC | 95% CI | Note |\n|---|---|---|---|---|---|",
    ]
    for s in r["subgroups"]:
        lines.append(f"| {s['subgroup']} | {s['stays']} | {s['septic_stays']} | {s['auroc']:.3f} | "
                     f"{_ci(s.get('auroc_ci'))} | {'unstable (few events)' if s['unstable'] else ''} |")
    lines += ["", "## Rule-based references (same split)", "", "| Score | AUROC | AUPRC |\n|---|---|---|"]
    for k, v in r["baselines"].items():
        lines.append(f"| {k} | {v['auroc']:.3f} | {v['auprc']:.3f} |")
    op = r["operational"]
    lines += ["", "## Operational", "",
              f"Batch scoring {op['rows']} rows in {op['batch_scoring_seconds']:.2f} s "
              f"({op['rows_per_second']:.0f} rows/s); full evaluation {op['evaluation_seconds']} s."]
    return "\n".join(lines) + "\n"
