"""Command-line interface: ``python -m sepsis <command>``.

download -> validate -> prepare -> train -> calibrate -> select-policy -> evaluate
(plus ``fixture`` for a synthetic smoke corpus, ``promote`` and ``report``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_run(run: str) -> Path:
    from sepsis.models.bundle import bundle_root

    root = bundle_root()
    if run == "latest":
        runs = sorted(p for p in root.iterdir() if (p / "bundle.json").exists())
        if not runs:
            raise SystemExit("no runs found")
        return runs[-1]
    p = root / run
    if not (p / "bundle.json").exists():
        raise SystemExit(f"run {run} not found under {root}")
    return p


def cmd_download(a):
    from sepsis.data.download import download_physionet2019

    if a.dataset != "physionet2019":
        raise SystemExit("only physionet2019 is downloadable without credentials")
    cfg = _load_yaml(a.config)["download"] if a.config else {}
    m = download_physionet2019(workers=a.workers or cfg.get("workers", 8), retries=cfg.get("retries", 5),
                               timeout=cfg.get("timeout_s", 60), limit=a.limit)
    for h, v in m["hospitals"].items():
        print(f"hospital {h}: {v['present_files']} files, {v['total_bytes']:,} bytes, {v['transfer']}")


def cmd_validate(a):
    from sepsis.data.schema import EXPECTED_COUNTS, HOSPITALS, read_psv, sha256_file
    from sepsis.paths import raw_dir

    cfg = _load_yaml(a.config)
    root = Path(a.raw_root) if a.raw_root else raw_dir(cfg.get("dataset", "physionet2019"))
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else None
    problems, counts = [], {}
    for h in cfg.get("hospitals", ["A", "B"]):
        files = sorted((root / HOSPITALS[h]).glob("p*.psv"))
        counts[h] = len(files)
        mf = (manifest or {}).get("hospitals", {}).get(h, {}).get("files", {})
        for f in files:
            try:
                read_psv(f)
            except Exception as exc:
                problems.append(f"{h}/{f.name}: {exc}")
                continue
            if mf and f.name in mf and a.checksums and sha256_file(f) != mf[f.name]["sha256"]:
                problems.append(f"{h}/{f.name}: checksum differs from manifest")
        exp = cfg.get("expected_counts", EXPECTED_COUNTS).get(h)
        if not a.allow_partial and exp is not None and counts[h] != exp:
            problems.append(f"hospital {h}: {counts[h]} files, expected {exp}")
    print(json.dumps({"counts": counts, "problems": problems[:50], "n_problems": len(problems)}, indent=1))
    if problems:
        raise SystemExit(1)


def cmd_prepare(a):
    from sepsis.data.prepare import prepare
    from sepsis.data.splits import make_splits, save_splits
    from sepsis.data.store import Store
    from sepsis.features.causal import FeatureConfig, schema_version

    tcfg = _load_yaml(a.config)
    dcfg = _load_yaml(a.dataset_config)
    q = prepare(tcfg.get("dataset", "physionet2019"), raw_root=a.raw_root, workers=a.workers,
                require_expected_counts=not a.allow_partial)
    store = Store.for_dataset(tcfg.get("dataset", "physionet2019"))
    sp = dcfg.get("split", {})
    split = make_splits(store.encounters, seed=sp.get("seed", 2019), fractions=sp.get("fractions"))
    save_splits(split, store.splits_path)
    print("hospitals:", {h: {k: v[k] for k in ("stays", "rows", "ever_positive_stays",
                                              "positive_hour_prevalence")} for h, v in q["hospitals"].items()})
    print("splits:", split["counts"], "hash", split["split_hash"])
    for fs in tcfg.get("feature_sets", [{}]):
        cfg = FeatureConfig.from_dict(fs)
        feats = store.features(cfg)
        print(f"features {schema_version(cfg)}: {feats.shape}")


def cmd_train(a):
    from sepsis.models.train import train

    run_dir = train(_load_yaml(a.config), run_id=a.run_id)
    print(run_dir.name)


def cmd_calibrate(a):
    from sepsis.models.train import calibrate

    print(json.dumps(calibrate(_resolve_run(a.run)), indent=1))


def cmd_select_policy(a):
    from sepsis.evaluation.pipeline import select_policy

    cfg = _load_yaml(a.config) if a.config else {}
    res = select_policy(_resolve_run(a.run), split=a.split, budgets=tuple(cfg.get("budgets", (1, 2, 5, 10, 20))),
                        primary_budget=cfg.get("primary_budget", 10),
                        consecutive=tuple(cfg.get("consecutive", (1, 2, 3))),
                        cooldowns=tuple(cfg.get("cooldown_h", (0, 6, 12))))
    for b, v in res["by_budget"].items():
        if v["feasible"]:
            print(f"budget {b:>4}/100pd: thr={v['threshold']:.4f} k={v['consecutive']} cd={v['cooldown_h']} "
                  f"rate={v['alerts_per_100_patient_days']:.2f} early_sens={v['early_sensitivity']:.3f} "
                  f"precision={v['alert_precision']:.3f}")
        else:
            print(f"budget {b:>4}/100pd: infeasible")
    print("selected:", res["selected_policy"], "benchmark threshold:", round(res["benchmark_threshold"]["threshold"], 4))


def cmd_evaluate(a):
    from sepsis.evaluation.pipeline import evaluate

    r = evaluate(_resolve_run(a.run), a.split, n_boot=a.n_boot, official=not a.no_official)
    print((_resolve_run(a.run) / "eval" / a.split / "report.md").read_text())
    return r


def cmd_promote(a):
    from sepsis.models.bundle import bundle_root
    from sepsis.models.train import promote

    out = bundle_root().parent / "promotion.json"
    print(json.dumps(promote([_resolve_run(r) for r in a.runs], out), indent=1))


def cmd_fixture(a):
    from sepsis.data.synthetic import make_fixture

    print(make_fixture(Path(a.out), n_a=a.n_a, n_b=a.n_b, seed=a.seed))


def cmd_report(a):
    from sepsis.evaluation.report import build_experiments_report

    path = build_experiments_report()
    print(path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sepsis", description=__doc__)
    p.add_argument("--data-dir", help="override data directory (SEPSIS_DATA_DIR)")
    p.add_argument("--artifact-dir", help="override artifact directory (SEPSIS_ARTIFACT_DIR)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download"); s.add_argument("--dataset", default="physionet2019")
    s.add_argument("--config", default="configs/physionet2019.yaml"); s.add_argument("--workers", type=int)
    s.add_argument("--limit", type=int, help="first N files per hospital (testing)"); s.set_defaults(fn=cmd_download)

    s = sub.add_parser("validate"); s.add_argument("--config", default="configs/physionet2019.yaml")
    s.add_argument("--raw-root"); s.add_argument("--checksums", action="store_true")
    s.add_argument("--allow-partial", action="store_true"); s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("prepare"); s.add_argument("--config", default="configs/challenge2019_state.yaml")
    s.add_argument("--dataset-config", default="configs/physionet2019.yaml"); s.add_argument("--raw-root")
    s.add_argument("--workers", type=int, default=8); s.add_argument("--allow-partial", action="store_true")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train"); s.add_argument("--config", required=True); s.add_argument("--run-id")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("calibrate"); s.add_argument("--run", required=True); s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("select-policy"); s.add_argument("--run", required=True)
    s.add_argument("--split", default="a_validation"); s.add_argument("--config", default="configs/alert_policy.yaml")
    s.set_defaults(fn=cmd_select_policy)

    s = sub.add_parser("evaluate"); s.add_argument("--run", required=True); s.add_argument("--split", required=True)
    s.add_argument("--n-boot", type=int, default=200); s.add_argument("--no-official", action="store_true")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("promote"); s.add_argument("--runs", nargs="+", required=True); s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("fixture"); s.add_argument("--out", required=True)
    s.add_argument("--n-a", type=int, default=120); s.add_argument("--n-b", type=int, default=60)
    s.add_argument("--seed", type=int, default=7); s.set_defaults(fn=cmd_fixture)

    s = sub.add_parser("report"); s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    if a.data_dir:
        os.environ["SEPSIS_DATA_DIR"] = str(Path(a.data_dir).resolve())
    if a.artifact_dir:
        os.environ["SEPSIS_ARTIFACT_DIR"] = str(Path(a.artifact_dir).resolve())
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
