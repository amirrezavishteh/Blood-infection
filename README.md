# Sepsis early-warning research simulator

A retrospective research application that replays de-identified ICU records hour by hour, computes a **research sepsis score**, shows its trajectory and contributing measurements, and evaluates an explicit, versioned alert policy. It is built on the open **PhysioNet/CinC Challenge 2019** dataset.

> **Not for patient care.** The score estimates the PhysioNet 2019 benchmark state (`challenge2019_state`). It is not a cleared diagnostic device, not a clinically validated probability, and not a prediction of *new* sepsis in the next six hours. Retrospective results do not establish clinical benefit.

## Results on the real PhysioNet 2019 data

All three models were trained on hospital A, calibrated and given a frozen alert policy on hospital A data, and then scored **once** on held-out records: the hospital A internal test (3,050 records) and all of hospital B (20,000 records), a different hospital never used for any choice. 95% intervals come from 200 record-level bootstrap resamples.

| Model | Split | Hourly AUROC | Hourly AUPRC | Official utility | Early detection* | Alert precision | Alerts / 100 patient-days |
|---|---|---|---|---|---|---|---|
| Logistic regression | A test | 0.766 (0.739–0.790) | 0.078 | 0.331 (0.284–0.385) | 20.7% | 13.4% | 10.4 |
| Logistic regression | Hospital B | 0.683 (0.669–0.698) | 0.047 | 0.131 (0.102–0.158) | 12.8% | 11.9% | 4.2 |
| LightGBM | A test | 0.786 (0.762–0.807) | 0.076 | 0.344 (0.292–0.394) | 20.3% | 14.1% | 10.6 |
| LightGBM | Hospital B | 0.742 (0.731–0.755) | 0.053 | 0.177 (0.150–0.202) | 11.6% | 14.6% | 3.6 |
| **LightGBM + measurement-pattern features** (promoted) | A test | **0.808 (0.786–0.830)** | **0.085** | **0.393 (0.346–0.446)** | 20.3% | 15.9% | 9.0 |
| **LightGBM + measurement-pattern features** (promoted) | Hospital B | **0.776 (0.764–0.788)** | **0.061** | **0.243 (0.216–0.268)** | 12.0% | 12.1% | 4.0 |

\*Septic records with an alert 0–12 h before the onset proxy, at the alert policy chosen on hospital A validation for 10 alerts per 100 patient-days.

- **Every model loses accuracy at the new hospital.** The official utility falls by a third to more than half from A to B; the simple logistic model drops most.
- **Rule references are much weaker.** Partial qSOFA and SIRS reach AUROC 0.58 and 0.64 on the A test (0.58 and 0.65 on B).
- **Alerts are the hard part.** At the chosen operating point the model warns early for about one septic patient in five on hospital A and one in eight on hospital B, with roughly one correct alert in every six to eight.
- **Measurement-pattern features helped and transferred best here**, but they can encode local ordering habits (in the dashboard, "hours since the last lactate" is often a top driver), so this result needs checking at more hospitals.
- Positive-hour prevalence is 2.2% (A) and 1.4% (B), so a random score has AUPRC around 0.02 and 0.014.

Full per-run reports (calibration, subgroups, lead times, official-scorer files) are written to `artifacts/runs/<run>/eval/<split>/`.


## What is in the box

| Path | Content |
|---|---|
| `src/sepsis/data/` | Verified downloader (resumable, checksummed manifest), PSV schema validation, Parquet store, frozen splits, synthetic fixtures |
| `src/sepsis/labels/` | Target A `challenge2019_state` (supplied label, never re-shifted), onset proxy, Target B `incident_sepsis_6h` horizon labels with censoring |
| `src/sepsis/features/` | Shared **causal** feature transformer, used identically by batch training and live replay |
| `src/sepsis/models/` | Train-only preprocessing, logistic baseline, LightGBM candidate, Platt calibration, versioned and checksummed model bundles, contributions |
| `src/sepsis/alerts/` | Alert policy state machine, budgeted policy search, alert/onset matching metrics |
| `src/sepsis/evaluation/` | Vectorised challenge utility, pinned official scorer adapter, record-bootstrap CIs, calibration, subgroup and coverage reports |
| `apps/api/` | FastAPI contracts, SQLAlchemy app state (PostgreSQL or SQLite), persistent job queue with leases |
| `apps/worker/` | Background worker: idempotent transactional hourly replay, playing-replay ticker, dataset validation jobs |
| `apps/web/` | React + TypeScript dashboard (dataset manager, replay workspace, alert review, experiment report) |
| `configs/` | Dataset, target, model and alert-policy YAML |
| `tests/` | Leakage, label-boundary, contamination, train/serve, idempotency, recovery, clock-access and PostgreSQL tests |
| `third_party/evaluation_2019/` | Official PhysioNet 2019 evaluator, pinned at commit `467c49b` (BSD-2) |
| `infra/` | Dockerfile, Compose (PostgreSQL + API + worker), systemd units, nginx config, Windows pipeline script, smoke script |
| `docs/DEPLOYMENT.md` | Deployment guide |

`data/` and `artifacts/` are git-ignored. They hold row-level data and derived artifacts governed by the source terms.

## Run locally

Works on Windows, macOS and Linux. You need **Python 3.11+**, **Node 20+** and **git**. The demo runs on any laptop; the real-data pipeline wants about 16 GB RAM and 30 GB of free disk.

### 1. Install

Windows (PowerShell):

```powershell
git clone https://github.com/amirrezavishteh/Blood-infection.git
cd Blood-infection
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
cd apps\web; npm ci; npm run build; cd ..\..
```

macOS / Linux:

```bash
git clone https://github.com/amirrezavishteh/Blood-infection.git
cd Blood-infection
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
(cd apps/web && npm ci && npm run build)
```

The examples below use `python`; on Windows that means `.venv\Scripts\python`, on macOS/Linux `.venv/bin/python` (or activate the venv first).

### 2a. Quick demo with synthetic data (about 2 minutes, no download)

```bash
python -m sepsis --data-dir data/demo/data --artifact-dir data/demo/artifacts demo
python -m sepsis --data-dir data/demo/data --artifact-dir data/demo/artifacts serve
```

Open <http://127.0.0.1:8000>. The demo is simulated data in PhysioNet format and is labelled *synthetic* in the app; its numbers say nothing about real performance.

### 2b. Real PhysioNet data (about 1.5 hours on 32 cores)

```bash
python -m sepsis -v download --dataset physionet2019
python -m sepsis validate --checksums
python -m sepsis -v prepare
python -m sepsis -v train --config configs/lightgbm_extended.yaml --run-id lgbm-ext
python -m sepsis calibrate --run lgbm-ext
python -m sepsis select-policy --run lgbm-ext --split a_validation
python -m sepsis evaluate --run lgbm-ext --split a_test
python -m sepsis evaluate --run lgbm-ext --split b_external
python -m sepsis report
python -m sepsis serve
```

- `python -m sepsis explain --run lgbm-ext` writes which feature families the model relies on and how its top features shift at hospital B; `python -m sepsis model-card --run lgbm-ext` writes `MODEL_CARD.md` from the run's own files.
- The download resumes if interrupted; rerun the same command.
- `infra/run_pipeline.ps1` trains and evaluates all three models (logistic, LightGBM, extended LightGBM) and promotes the best one on validation.
- `--run latest` refers to the newest run.

### 3. Use the dashboard

1. **Datasets** → **Import PhysioNet 2019** → wait until the status is *ready*.
2. **Replay** → choose *a_test* (hospital A) or *b_external* (hospital B) → **Create replay**.
3. **Step +1 h** or **Play**; click a record to see its score, drivers, vitals and labs.
4. **Alert review** to acknowledge simulated alerts; **Experiments** for the evaluation report.
5. After a replay finishes, **Show retrospective labels** reveals the true labels and onset proxy.

### Development mode (live reload)

```bash
python -m sepsis serve                 # terminal 1: API + worker on :8000
cd apps/web && npm run dev             # terminal 2: dashboard on http://127.0.0.1:5173 (proxies to :8000)
```

### Tests

```bash
python -m pytest                       # 82 tests: core, API/worker, embedded PostgreSQL
cd apps/web && npx vitest run          # dashboard tests
bash infra/smoke.sh                    # 2-minute end-to-end pipeline check on synthetic data
```

The same checks run on every push and pull request through GitHub Actions (`.github/workflows/ci.yml`).

### Configuration

| Setting | Default | Purpose |
|---|---|---|
| `--data-dir` / `SEPSIS_DATA_DIR` | `data/` | Raw and processed data |
| `--artifact-dir` / `SEPSIS_ARTIFACT_DIR` | `artifacts/` | Model runs and reports |
| `DATABASE_URL` | SQLite file in `data/app/` | e.g. `postgresql+psycopg://user:pass@host:5432/sepsis` |
| `SEPSIS_API_TOKEN` | unset | When set, every `/v1` request needs `Authorization: Bearer <token>`; the dashboard asks for it once |
| `serve --host / --port` | `127.0.0.1:8000` | Bind address; keep localhost unless behind a reverse proxy |

## Deploy to a server

Step-by-step instructions for **Docker Compose** and for a **Linux server with systemd, PostgreSQL and nginx/HTTPS**, plus updates, backups, a security checklist and troubleshooting, are in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Data

| Dataset | Status here | Role |
|---|---|---|
| [PhysioNet 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/) (CC BY 4.0) | Downloaded and verified: 20,336 A + 20,000 B files, SHA-256 manifest | Training (A) and untouched hospital-transfer test (B) |
| MIMIC-IV, eICU-CRD, HiRID, AmsterdamUMCdb | Not used; credentialed access required | Future Target B / multi-hospital studies |

The downloader reads each server directory listing (names and byte sizes) and downloads with modest concurrency and exponential backoff. It enforces a per-file deadline, rejects HTML or oversized payloads, validates the 41-column schema before accepting a file, writes atomically, and fails visibly unless 20,336 / 20,000 valid files are present. The bulk ZIP endpoint returns 404, so per-file endpoints are used.

Please cite: Reyna MA, et al. *Early Prediction of Sepsis From Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019.* Crit Care Med 2020;48(2):210–217; and Goldberger A, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 2000;101(23).

## Design decisions that matter

**Target.** `challenge2019_state` is the supplied `SepsisLabel`, unchanged. It turns positive 6 h before the challenge-defined onset and stays positive. The onset proxy (first positive hour + 6, the same convention as the official scorer's `t_sepsis`) is an analysis convention. Records positive from their first row are left-censored: they stay in the benchmark but are excluded from lead-time summaries. Target B (`incident_sepsis_6h`) is implemented and boundary-tested for richer datasets, but is not trained here.

**Splits.** Hospital A is split 60/15/10/15 into train, validation, calibration and test, stratified by ever-positive status with seed 2019; the split is persisted with a hash and checked for tampering. All of hospital B is the external test, used only for final evaluation, and every look at B is logged. PhysioNet 2019 exposes no person linkage, so records (stays) are the grouping and bootstrap unit. MIMIC-IV is not treated as independent of hospital A, since both come from Beth Israel Deaconess.

**Causal features.** The feature row for hour *t* uses only that record's rows ≤ *t*. Each row is the hourly bin that has just ended; the PSV files give no result-availability times, so this is an approximation. The feature families are:
- **Vitals:** latest value within a 6 h expiry, and trailing 3/6/12 h mean, min, max, std (≥2 observations) and slope (≥2 observation times).
- **Labs:** latest value within a 48 h expiry, and the change since the previous observation.
- **Demographics:** age and the source `Gender` code.
- **Derived:** shock index and pulse pressure.

Nothing is back-filled, interpolated or normalised per stay. Labels, IDs, hospital identity and unit indicators are excluded. The `extended` variant adds observation-pattern features (observed-now indicators, hours since last measurement, measurement counts), which can encode local ordering workflow; transfer is reported for both variants. Expiry settings were fixed in advance, not tuned on test data.

**Models.** Regularised logistic regression (train-only clipping, median imputation with missingness indicators, scaling) and CPU LightGBM with a small fixed grid, early-stopped on A validation only, compared on the same split. Class weighting affects training only; evaluation uses natural prevalence. Selection uses A-validation AUPRC. A sigmoid calibrator is fitted on A calibration. Bundles carry the model, calibrator, feature order and dtypes, schema hash, target, provenance (manifest hash, split hash, code revision), benchmark threshold and frozen alert policy. They are loaded only from the trusted artifact directory after a checksum check.

**Two outputs.** (1) A complete hourly score series and a benchmark threshold (maximising utility on A validation), scored with the pinned official evaluator. (2) An emitted-alert stream from the policy state machine. Hours without at least one valid vital sign are `data_unavailable`: they are never displayed as low risk, and the benchmark scores them with a documented fixed fallback (p = 0), whose count is reported.

**Alert policy.** A versioned state machine with states `monitoring`, `alerted`, `acknowledged`, `cooldown` and `data_unavailable`, separate from the model. It is parameterised by threshold, consecutive hours and cooldown, and searched on A validation under simulated budgets of 1, 2, 5, 10 and 20 alerts per 100 patient-days (primary: 10). Budgets are research comparison points, not clinical limits. Acknowledgement records simulated review; it is not an outcome label and never retrains the model.

**Replay.** Each replay pins dataset, records, model, target, policy and schema. Each simulated hour runs as one database transaction: visible history, causal features, score, policy step, then prediction, alert, audit event and checkpoint. Unique constraints on `(replay_id, stay_id, hour_index, model_version, policy_version)` and on alert triggers, together with the checkpoint, make retries and duplicate executions no-ops. A crash mid-step rolls back completely. Jobs are claimed with a conditional `UPDATE … RETURNING`, carry leases and heartbeats, and are reclaimed after a worker dies. Active-replay endpoints never return data beyond the replay clock (403) and never return labels; the retrospective view (labels, onset proxy) unlocks only after a replay finishes. A reset creates a new replay ID and preserves the original run.

## Tests

```bash
.venv/Scripts/python -m pytest          # Python: core, API/worker, PostgreSQL (embedded via pgserver)
cd apps/web && npx vitest run           # dashboard
```

| Risk | Test |
|---|---|
| Look-ahead leakage | Perturbing or removing future rows leaves every earlier feature unchanged; online prefix features equal batch features |
| Shifted-label error | Stored outcomes equal the raw `SepsisLabel`; no extra shift |
| Incident-label boundaries | Onset at *t*, at *t*+6, at *t*+7, discharge before horizon end, prevalent exclusion |
| Train/test contamination | Split ID intersections are empty, stratification holds, tampered split files are rejected, imputation medians equal train-only statistics |
| Train/serve drift | Replay scores in the database equal batch inference for every stay-hour |
| Duplicate processing | Duplicate step requests reuse the job; re-executed steps are no-ops; a mid-step crash leaves nothing behind; an expired lease is reclaimed and late completion by the dead worker is ignored |
| False reassurance | Missing vitals give an explicit `data_unavailable`; schema mismatch, checksum tampering and untrusted paths are refused |
| Future-data UI leakage | Requests beyond the replay clock return 403; responses contain no labels; retrospective access is locked until the replay finishes |
| Official scorer | The vectorised utility equals the pinned official evaluator on random cases and end to end |
| PostgreSQL | A 6-thread claim race never double-assigns a job; replay steps run idempotently |

## Limits

- PhysioNet 2019 has no per-measurement availability times, no person linkage across admissions, no GCS, and no culture or antibiotic events. The rule-based qSOFA/SIRS references are partial.
- Hospital B is a single external site; transfer to other institutions, live feeds or delayed lab results is untested.
- The Compose stack and the systemd/nginx files were written but not run in the development environment (no Docker or Linux server there). The same production configuration (PostgreSQL, 2 uvicorn workers, separate worker, bearer token) was tested natively end to end.
- A future EHR adapter needs event and receipt-time handling, terminology mapping, site-specific monitoring and prospective silent evaluation before any care-facing use.
