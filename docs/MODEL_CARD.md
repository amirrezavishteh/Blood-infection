# Model card: `lgbm-v2-extended`

> **Research use only.** Retrospective research score for the PhysioNet/CinC 2019 benchmark state. Not a medical device, not a validated clinical probability, not for patient care.

## Model

| Item | Value |
|---|---|
| Kind | lightgbm_bag (3 seeds) |
| Target | `challenge2019_state`: supplied SepsisLabel (positive from onset − 6 h onward), unshifted |
| Features | 319 causal features, set `extended`, schema `causal-v1-fcc69c40e34d` |
| Selected parameters | `{"bagging_fraction": 1.0, "class_weight": null, "early_stopping": 150, "feature_fraction": 0.3, "lambda_l2": 50.0, "learning_rate": 0.02, "max_rounds": 4000, "min_child_samples": 200, "num_leaves": 15}` |
| Validation AUPRC (selection metric) | 0.0956 |
| Calibration | sigmoid on `a_calibration` |
| Alert policy | threshold 0.0953, 1 consecutive h, cooldown 12 h, budget 10 alerts / 100 patient-days (`policy-81128dd780`) |

## Intended use

- Retrospective research and teaching: replaying de-identified ICU records, comparing models and alert policies.
- **Out of scope:** any clinical decision, live EHR feeds, populations other than adult ICU stays similar to the training hospitals.

## Training data

- PhysioNet/CinC Challenge 2019 v1.0.0 (CC BY 4.0), hospital A training split: 474,168 hourly rows, positive-hour prevalence 2.2%.
- Split hash `0355517360dd0672`; raw manifest SHA-256 `5eba1747d3949eca…`; code `d4866546ae24`; created 2026-10-03T21:22:50+00:00.

## Held-out performance

| Split | Records | Look (this run) | Earlier looks (other runs) | Hourly AUROC (95% CI) | AUPRC | Official utility (95% CI) | Brier | Early detection | Alert precision | Alerts / 100 pd |
|---|---|---|---|---|---|---|---|---|---|---|
| Hospital A internal test | 3,050 | 1 | 3 | 0.805 (0.783–0.828) | 0.091 | 0.369 (0.317–0.425) | 0.0206 | 17.7% | 12.8% | 9.8 |
| Hospital B (external) | 20,000 | 1 | 3 | 0.786 (0.775–0.798) | 0.063 | 0.249 (0.218–0.273) | 0.0137 | 11.4% | 16.1% | 2.9 |

Look counts come from each run's evaluation log. Any earlier look at the same records, by this run or another, means the result is not an untouched estimate and may be slightly optimistic; model choices were made on validation data only.

### Subgroups: Hospital A internal test

| Subgroup | Records | Septic | AUROC (95% CI) | Note |
|---|---|---|---|---|
| age <50 | 688 | 66 | 0.763 (0.720–0.797) |  |
| age 50-64 | 888 | 74 | 0.830 (0.801–0.866) |  |
| age 65-79 | 1,050 | 93 | 0.808 (0.772–0.841) |  |
| age >=80 | 423 | 35 | 0.823 (0.756–0.878) |  |
| Gender=0 (source code) | 1,275 | 102 | 0.791 (0.756–0.819) |  |
| Gender=1 (source code) | 1,774 | 166 | 0.814 (0.790–0.834) |  |

### Subgroups: Hospital B (external)

| Subgroup | Records | Septic | AUROC (95% CI) | Note |
|---|---|---|---|---|
| age <50 | 4,817 | 276 | 0.772 (0.744–0.796) |  |
| age 50-64 | 6,277 | 351 | 0.788 (0.767–0.809) |  |
| age 65-79 | 6,582 | 386 | 0.790 (0.769–0.807) |  |
| age >=80 | 2,322 | 129 | 0.804 (0.781–0.838) |  |
| Gender=0 (source code) | 9,268 | 495 | 0.797 (0.780–0.814) |  |
| Gender=1 (source code) | 10,730 | 647 | 0.777 (0.764–0.793) |  |

## What the model relies on

Share of mean absolute contribution by feature family: lab 42%, vital sign 33%, observation pattern 22%, derived score 3%, demographic 1%.

| Feature | Family | Share | Shift at B (SMD) | Missing A → B |
|---|---|---|---|---|
| `FiO2__hours_since` | observation pattern | 9.1% | +0.60 | 42% → 71% |
| `Resp__min_24h` | vital sign | 3.4% | +0.19 | 2% → 4% |
| `BUN__last` | lab | 3.2% | -0.04 | 16% → 25% |
| `Lactate__hours_since` | observation pattern | 3.2% | +0.16 | 64% → 76% |
| `Calcium__delta` | lab | 3.2% | +0.23 | 66% → 55% |
| `Phosphate__delta` | lab | 3.0% | +0.03 | 65% → 81% |
| `PTT__last` | lab | 2.3% | +0.11 | 34% → 80% |
| `Bilirubin_total__hours_since` | observation pattern | 2.0% | +0.00 | 76% → 63% |
| `WBC__from_first` | lab | 1.8% | +0.02 | 18% → 27% |
| `Creatinine__last` | lab | 1.7% | +0.15 | 18% → 25% |

Observation-pattern features (hours since a measurement, measurement counts) can encode local charting habits rather than physiology; large shifts at hospital B are a transfer risk.

## Limitations

- Two hospitals only; the label is a retrospective Sepsis-3 reconstruction; no GCS, cultures, antibiotics or vasopressors.
- Hourly bins approximate result availability; live data with delayed results is untested.
- Subgroup estimates with few septic records are unstable.
