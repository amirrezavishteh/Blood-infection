"""Model card generated from a run's own artifacts (bundle, evaluations, explanation).

Nothing is typed by hand: every number comes from ``bundle.json``,
``eval/<split>/metrics.json`` and ``explain.json``, so the card cannot drift
from the run it describes.
"""

from __future__ import annotations

import json
from pathlib import Path

from sepsis.models.bundle import bundle_root, load_bundle
from sepsis.models.train import selection_auprc

SPLIT_NAMES = {"a_test": "Hospital A internal test", "b_external": "Hospital B (external)"}


def _ci(c) -> str:
    return f" ({c[0]:.3f}–{c[1]:.3f})" if c else ""


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def prior_looks_by_other_runs(run_dir: Path, split: str) -> int:
    """Evaluations of ``split`` by other runs logged before this run's first look at it."""
    def log(rd: Path) -> list:
        f = rd / "evaluation_log.json"
        return json.loads(f.read_text()) if f.exists() else []
    mine = [e["at"] for e in log(Path(run_dir)) if e["split"] == split]
    if not mine:
        return 0
    first = min(mine)
    n = 0
    for rd in bundle_root().iterdir():
        if rd.resolve() != Path(run_dir).resolve():
            n += sum(1 for e in log(rd) if e["split"] == split and e["at"] < first)
    return n


def build_model_card(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    b = load_bundle(run_dir)
    m = b.meta
    evals = {}
    for split in ("a_test", "b_external"):
        p = run_dir / "eval" / split / "metrics.json"
        if p.exists():
            evals[split] = json.loads(p.read_text())
    ex = json.loads((run_dir / "explain.json").read_text()) if (run_dir / "explain.json").exists() else None
    fc = b.feature_config
    pol = m.get("alert_policy") or {}
    L = [
        f"# Model card: `{b.model_version}`",
        "",
        "> **Research use only.** Retrospective research score for the PhysioNet/CinC 2019 benchmark "
        "state. Not a medical device, not a validated clinical probability, not for patient care.",
        "",
        "## Model",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Kind | {m['model_kind']}{' (' + str(m['bag']['n_seeds']) + ' seeds)' if m.get('bag') else ''} |",
        f"| Target | `{b.target_id}`: supplied SepsisLabel (positive from onset − 6 h onward), unshifted |",
        f"| Features | {len(b.feature_names)} causal features, set `{fc.feature_set}`, schema `{b.feature_schema_version}` |",
        f"| Selected parameters | `{json.dumps(m.get('selected_params'), sort_keys=True)}` |",
        f"| Validation AUPRC (selection metric) | {selection_auprc(m):.4f} |",
        f"| Calibration | {(m.get('calibration') or {}).get('method', 'none')} on `{(m.get('calibration') or {}).get('fitted_on', '-')}` |",
        f"| Alert policy | threshold {pol.get('threshold', float('nan')):.4f}, {pol.get('consecutive')} consecutive h, "
        f"cooldown {pol.get('cooldown_h')} h, budget {pol.get('budget_alerts_per_100_patient_days')} alerts / 100 patient-days (`{pol.get('policy_version')}`) |",
        "",
        "## Intended use",
        "",
        "- Retrospective research and teaching: replaying de-identified ICU records, comparing models and alert policies.",
        "- **Out of scope:** any clinical decision, live EHR feeds, populations other than adult ICU stays similar to the training hospitals.",
        "",
        "## Training data",
        "",
        f"- PhysioNet/CinC Challenge 2019 v1.0.0 (CC BY 4.0), hospital A training split: {m['provenance']['train_rows']:,} hourly rows, "
        f"positive-hour prevalence {_pct(m.get('train_positive_prevalence'))}.",
        f"- Split hash `{m['provenance']['split_hash']}`; raw manifest SHA-256 `{str(m['provenance'].get('raw_manifest_sha256'))[:16]}…`; "
        f"code `{str(m['provenance'].get('code_revision'))[:12]}`; created {m['provenance'].get('created_at')}.",
        "",
        "## Held-out performance",
        "",
    ]
    if not evals:
        L.append("Not evaluated yet.")
    else:
        L += ["| Split | Records | Look (this run) | Earlier looks (other runs) | Hourly AUROC (95% CI) | AUPRC | Official utility (95% CI) | Brier | Early detection | Alert precision | Alerts / 100 pd |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, e in evals.items():
            h, bm, a = e["hourly"], e["benchmark"], e["alerts"]
            L.append(f"| {SPLIT_NAMES[s]} | {e['records']:,} | {e.get('evaluation_number_for_split', 1)} | {prior_looks_by_other_runs(run_dir, s)} | "
                     f"{h['auroc']:.3f}{_ci(h.get('auroc_ci'))} | {h['auprc']:.3f} | {bm['utility']:.3f}{_ci(bm.get('utility_ci'))} | "
                     f"{h['brier']:.4f} | {_pct(a['early_sensitivity'])} | {_pct(a['alert_precision'])} | {a['alerts_per_100_patient_days']:.1f} |")
        L += ["", "Look counts come from each run's evaluation log. Any earlier look at the same records, by this run or "
              "another, means the result is not an untouched estimate and may be slightly optimistic; model choices "
              "were made on validation data only.", ""]
        for s, e in evals.items():
            L += [f"### Subgroups: {SPLIT_NAMES[s]}", "", "| Subgroup | Records | Septic | AUROC (95% CI) | Note |", "|---|---|---|---|---|"]
            for g in e["subgroups"]:
                auroc = "n/a" if g["auroc"] is None else f"{g['auroc']:.3f}"
                note = "unstable (few events)" if g["unstable"] else ""
                L.append(f"| {g['subgroup']} | {g['stays']:,} | {g['septic_stays']} | {auroc}{_ci(g.get('auroc_ci'))} | {note} |")
            L.append("")
    if ex:
        L += ["## What the model relies on", "",
              "Share of mean absolute contribution by feature family: " +
              ", ".join(f"{k} {100 * v:.0f}%" for k, v in ex["family_share"].items()) + ".", "",
              "| Feature | Family | Share | Shift at B (SMD) | Missing A → B |", "|---|---|---|---|---|"]
        for r in ex["top_features"][:10]:
            smd = r["smd_b_vs_a"]
            L.append(f"| `{r['feature']}` | {r['family']} | {100 * r['importance_share']:.1f}% | "
                     f"{'n/a' if smd is None else f'{smd:+.2f}'} | {100 * r['missing_a_train']:.0f}% → {100 * r['missing_b']:.0f}% |")
        L += ["", "Observation-pattern features (hours since a measurement, measurement counts) can encode local charting "
              "habits rather than physiology; large shifts at hospital B are a transfer risk.", ""]
    L += ["## Limitations", "",
          "- Two hospitals only; the label is a retrospective Sepsis-3 reconstruction; no GCS, cultures, antibiotics or vasopressors.",
          "- Hourly bins approximate result availability; live data with delayed results is untested.",
          "- Subgroup estimates with few septic records are unstable.",
          ""]
    out = run_dir / "MODEL_CARD.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
