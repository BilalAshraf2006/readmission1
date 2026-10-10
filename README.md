# 30-day readmission after diabetic hospital stays: a generalization and fairness study

A leakage-free, mortality-excluded 30-day readmission model trained on the real
**UCI Diabetes 130-US Hospitals** data (1999–2008) and evaluated for

1. **cross-population generalization**: trained on one population and tested on another
   (an external **Synthea** cohort restricted to diabetic patients, plus the reverse
   direction and a within-Synthea reference);
2. **subgroup discrimination, calibration and error rates** by race, sex, age band and
   insurance, with equal-opportunity and equalized-odds gaps.

All numeric results are written by the code to `results/`, and all figures to `figures/`.

| Deliverable | Location |
|---|---|
| Figures | `figures/*.png` (see the list below) |
| All numeric results | `results/*.csv`, `results/*.json` |
| Harmonized external cohort (for re-analysis without re-running Synthea) | `data/external/synthea_cohort.csv.gz` |

## Headline results

The numbers below are copied from `results/`.

| | Logistic regression | Gradient boosting |
|---|---|---|
| Internal test AUROC (diabetes-130; 19,583 encounters) | 0.644 (0.629–0.658) | 0.649 (0.633–0.663) |
| Internal calibration slope | 1.02 (0.92–1.10) | 1.00 (0.91–1.08) |
| External AUROC (Synthea; 6,332 stays) | 0.544 (0.509–0.579) | 0.542 (0.506–0.574) |
| External observed/expected | 0.69 (0.62–0.76) | 0.78 (0.71–0.87) |
| External AUROC, unplanned-readmission outcome | 0.478 (0.444–0.510) | 0.470 (0.432–0.503) |
| Synthea-trained model on the Synthea test split | 0.676 (0.615–0.737) | 0.759 (0.699–0.813) |
| External AUROC if planned-care stays are kept (un-amended) | 0.751 (0.734–0.768) | 0.781 (0.765–0.796) |

* The model is modest but well calibrated internally. It does **not** transfer to the
  synthetic population, and whatever external signal remains comes from recognizing
  patients due for scheduled (planned) treatment.
* The failure is one of transfer: models trained on Synthea predict well within
  Synthea but fail on diabetes-130. A domain classifier separates the two cohorts almost
  perfectly (AUROC 1.000).
* Fairness: internally, false-positive rates rise steeply with age while calibration
  holds. Externally, the subgroup pattern changes: a sex gap in sensitivity appears,
  and over-prediction concentrates in African American and Asian patients.

## Quick start

```bash
pip install -r requirements.txt     # Python 3.11; Java 17+ is needed only to regenerate Synthea
python run_all.py                   # full pipeline (Synthea generation takes ~5 h on 4 cores)
python run_all.py --use-committed-synthea   # skip Synthea; use the committed harmonized cohort
python run_all.py --from train      # re-run the analysis from existing cohorts
python -m pytest tests              # unit tests (metrics, ICD-9/SNOMED mapping, grouped split)
```

Fitted models are cached in `models/` under a hash of their inputs, so re-running
`train.py` with unchanged data reloads them. `python src/train.py --only-d130` fits only
the diabetes-130 models (no Synthea data needed).

Individual steps (run from `src/`):

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `download_data.py` | Downloads diabetes-130 (UCI first, verified mirror as fallback) and the Synthea jar | `data/raw/`, `results/download_manifest.json` |
| 2 | `generate_synthea.py` | Runs Synthea for 8 states (2 per census region) with a diabetes keep-module | `data/raw/synthea/run_*/csv/` |
| 3a | `build_diabetes130.py` | Label, exclusions, missing values, harmonized features | `data/processed/d130.parquet`, `results/cohort_flow_d130.json` |
| 3b | `build_synthea.py` | Stays, label from dates, exclusions, harmonized features | `data/processed/synthea.parquet`, `results/cohort_flow_synthea.json` |
| 4 | `train.py` | Grouped split, grouped-CV tuning, final fit, predictions | `results/models.json`, `results/predictions.parquet` |
| 5 | `evaluate.py` | Metrics with patient-level bootstrap CIs, DCA, fairness, sensitivity analyses | `results/metrics_*.csv`, `results/fairness_*.csv`, `results/dca.csv` |
| 5b | `shift.py` | Table 1, standardized mean differences, domain classifier | `results/table1.csv`, `results/domain_classifier.json` |
| 6 | `figures.py` | All figures | `figures/` |

`src/config.py` holds every path, seed and constant, `src/harmonize.py` holds the
shared feature definitions and the ICD-9 ↔ SNOMED mapping, and `src/metrics.py` holds
the metric and bootstrap code.

## Key design decisions

* **Label**: `readmitted == "<30"` → 1; `"NO"` and `">30"` → 0. For Synthea: the next
  inpatient stay starts ≤ 30 days after discharge. Overlapping or abutting inpatient
  encounters (e.g. ICU transfers) are first merged into one stay, so a transfer is never a
  readmission.
* **Exclusions**: discharge disposition 11, 13, 14, 19, 20, 21 (died or hospice), and
  3 rows with sex "Unknown/Invalid". In Synthea, in-hospital death or discharge to
  hospice. The exact counts are in `results/cohort_flow_*.json` and Figure 1.
* **Amendment 1 (planned care)**: data QA of the first Synthea state, done before any
  external evaluation, showed that 67% of its index stays were planned care (scheduled
  inpatient chemotherapy cycles, rehab/detox and transplant admissions), whose
  "readmission" is simply the next scheduled cycle. Planned-care index admissions are
  therefore excluded in both cohorts (diabetes-130: primary diagnosis V57/V58), following
  the CMS "always planned" categories. The outcome definition is unchanged. The
  un-amended cohort and an unplanned-only outcome are reported as sensitivity analyses.
* **No leakage**: an 80/20 split grouped by patient (StratifiedGroupKFold), with
  hyper-parameters tuned by grouped 5-fold CV inside the training split. All
  pre-processing sits inside sklearn Pipelines, so it is re-fitted per fold. Bootstrap
  CIs resample patients, not encounters.
* **Shared features** (identical definitions in both cohorts): age (10-year band
  midpoint), sex, length of stay, prior-year inpatient / emergency / outpatient visits,
  number of diagnoses (capped at 9, the diabetes-130 recording limit), number of
  medications, and 18 comorbidity flags (ICD-9 3-digit categories ↔ SNOMED-CT concepts;
  see `results/comorbidity_mapping.csv` and the audit trail
  `results/synthea_snomed_flag_audit.csv`). **Race and insurance are never model
  inputs.** They are used only for the fairness audit.
* **Missing values**: `?` means missing. The literal `"None"` in `A1Cresult` /
  `max_glu_serum` means "not measured" and is kept as a category. `weight` (97% missing)
  is dropped. `payer_code` (40% missing) is not a feature but becomes the insurance
  audit attribute with an explicit "Unknown" group. `medical_specialty` (49% missing)
  has no Synthea analogue, so it appears (with a "Missing" level) only in the internal
  full-feature sensitivity model. Missing race is "Unknown". A missing diagnosis code
  sets no comorbidity flag.
* **Models**: L2 logistic regression and `HistGradientBoostingClassifier`, with no
  resampling or class weights (this preserves calibration). The decision threshold
  maximizes Youden's J on training out-of-fold predictions and is reused unchanged on
  every test set.
