# ASD Eye-Tracking Classification — Paper-Ready Pipeline

**TabICLv2 vs XGBoost · Leak-Free LOOCV · Severity Scoring · Explainability**

This notebook implements the full machine learning pipeline for the paper *"ASD Detection via Eye-Tracking Biomarkers using Tabular Foundation Models"*. It classifies children as ASD or typically-developing (TD) from oculomotor features derived from webcam-quality eye-tracking recordings, using a rigorously leak-free evaluation framework.

---

## Overview

The pipeline compares two classifiers under identical conditions:

- **TabICLv2** (primary) — a tabular in-context learning foundation model (ICML 2026) that requires no gradient-based training on the target dataset
- **XGBoost** (baseline) — a gradient-boosted tree ensemble trained from scratch within each fold

Both models operate on the same 26 oculomotor features (fixation, saccade, blink, pupil, and gaze position metrics) extracted from eye-tracking trials, using the same cross-validation folds and the same within-fold feature selection procedure.

---

## Data

The pipeline expects a single CSV file: `trial_features_tabpfn.csv`

Each row is one trial for one participant. Required columns:

| Column | Description |
|---|---|
| `participant_id` | Unique participant identifier |
| `trial` | Trial identifier |
| `stimulus_type` | Stimulus condition |
| `label` | Ground truth (1 = ASD, 0 = TD) |
| *(feature columns)* | 26 oculomotor feature values |

The data was degraded to simulate ~25 Hz webcam-quality eye-tracking (median inter-sample interval ≈ 39.8 ms) prior to feature extraction. The pipeline works on this degraded signal directly.

If running on Google Colab without the CSV already present, the notebook will prompt you to upload a zip file containing it.

---

## Setup

### Requirements

```bash
pip install tabicl[finetune] skrub scikit-learn statsmodels pandas numpy matplotlib shap joblib scipy tqdm xgboost
```

PyTorch is also required (installed automatically with `tabicl`). GPU is optional — the notebook detects CUDA automatically. Fine-tuning (`FinetunedTabICLClassifier`) is used only for the final model after LOOCV, not inside any fold.

### Running

Open `ASD_TabICLv2_PaperReady.ipynb` and run cells top to bottom. Each step is self-contained and saves its outputs to `results_paper/` before proceeding.

---

## Pipeline Steps

### Participant Split (fixed before everything else)
A stratified 80/20 participant-level split (seed = 42) is created once and locked to disk as `results_paper/split_indices.json`. All downstream steps load this file — the split is never re-created unless the file is deleted.

- Training cohort: n = 46 (ASD = 22, TD = 24)
- Independent holdout: n = 11 (ASD = 5, TD = 6)

### Step 0 — Global FDR (reporting only)
Benjamini-Hochberg FDR correction is run on participant-means of the full training set to produce the paper's feature-statistics table (`feature_statistics.csv`). This result is **not** used for model training — it exists purely for descriptive reporting.

### Step 1 — Leave-One-Out Cross-Validation (primary result)
The core evaluation. For each of the 46 training participants:

1. That participant is held out entirely
2. FDR feature selection runs on the remaining 45 participants' means only
3. Both TabICLv2 and XGBoost are trained on all trials from those 45 participants
4. Both models predict on all trials of the held-out participant
5. Trial-level probabilities are soft-voted (mean) into one participant-level P(ASD) score

Performance is measured at the participant level. Primary threshold is fixed at 0.5; the Youden-J threshold is computed after all folds complete and reported as a secondary metric only.

Bootstrap 95% confidence intervals (2,000 iterations) are computed for AUC, sensitivity, specificity, F1, accuracy, and Brier score. Cohen's d on P(ASD) scores quantifies effect size.

### Step 2 — McNemar's Test
Tests whether TabICLv2 and XGBoost make significantly different errors across participants. Uses the exact McNemar test (two-tailed).

### Step 3 — Severity Scoring
Each participant's soft-voted P(ASD) is mapped to a three-tier clinical risk band:

| Band | P(ASD) |
|---|---|
| Low risk | < 0.40 |
| Borderline | 0.40 – 0.69 |
| High risk | ≥ 0.70 |

A numeric severity score (0–100) is also recorded. Distribution plots are saved for both groups.

### Step 4 — Global Feature Importance
Final models are trained on the full training set (n = 46) using FDR-selected features.

- **TabICLv2**: permutation importance — AUC drop when each feature is shuffled (5 repetitions per feature)
- **XGBoost**: mean absolute SHAP values via `shap.TreeExplainer`

Top-10 features are plotted side-by-side. No holdout data is used in this step.

### Step 5 — Per-Participant Local Explanations
For every participant, the fold's trained model (the exact model that produced that participant's prediction) is used to compute local feature importance via permutation. The top 3 driving features are recorded in plain language. XGBoost local SHAP values are also computed.

All 46 participants' explanations are saved to `participant_explanations.csv`, and predicted-ASD cases are printed with plain-language summaries.

### Step 6 — Permutation Test
N = 1,000 full-pipeline re-runs with shuffled participant labels to generate a null AUC distribution. The empirical p-value is the proportion of null AUCs ≥ the observed AUC.

### Step 7 — ROC Curves & Probability Distributions
ROC curves for both models with bootstrap CI-annotated AUC values, plus a histogram of P(ASD) scores by true diagnosis group.

### Step 8 — Holdout Evaluation (one-time gate)
The independent holdout set is evaluated exactly once. A flag file (`results_paper/.holdout_used`) is written on completion. If the flag exists, this cell skips automatically — preventing p-hacking by re-running on holdout until results look good.

---

## Outputs

All outputs are saved to `results_paper/`:

| File | Description |
|---|---|
| `split_indices.json` | Fixed train/holdout participant IDs |
| `feature_statistics.csv` | Global FDR table with Cohen's d and BH-corrected p-values |
| `loocv_fold_results.csv` | Per-fold predictions for both models |
| `loocv_tabicl_participants.csv` | Participant-level results, TabICLv2 |
| `loocv_xgb_participants.csv` | Participant-level results, XGBoost |
| `metrics_tabicl.csv` | Bootstrap CI metrics, TabICLv2 |
| `metrics_xgb.csv` | Bootstrap CI metrics, XGBoost |
| `mcnemar_test.csv` | McNemar test contingency table and p-value |
| `permutation_test.csv` | Null distribution summary and empirical p-value |
| `permutation_test.png` | Null AUC histogram with observed AUC marked |
| `importance_tabicl.csv` | TabICLv2 permutation importance scores |
| `importance_xgb_shap.csv` | XGBoost mean SHAP values |
| `importance_global.png` | Side-by-side global importance bar charts |
| `participant_explanations.csv` | Per-participant top features and severity scores |
| `severity_distribution.png` | Strip plot and stacked severity bar chart |
| `roc_and_distributions.png` | ROC curves and P(ASD) histograms |
| `holdout_tabicl.csv` | Holdout predictions, TabICLv2 (written once) |
| `holdout_xgb.csv` | Holdout predictions, XGBoost (written once) |

---

## Key Design Decisions

**No leakage.** Feature selection (FDR) runs exclusively on participant-means of the training fold at each of the 46 LOOCV iterations. The held-out participant is invisible to both feature selection and model fitting. This addresses the most common methodological flaw in prior ASD eye-tracking ML studies.

**Participant-level evaluation.** Models are trained on trials but evaluated on participants. Trial-level probabilities are aggregated by soft voting before any metric is computed — consistent with clinical deployment where a diagnosis is per-person, not per-trial.

**Fixed primary threshold.** The 0.5 threshold is fixed before LOOCV begins. The Youden-J threshold is computed after all folds complete and reported as a secondary metric only, ensuring the primary results are not threshold-optimised.

**One-touch holdout.** The holdout set is evaluated once and gated by a flag file. This is enforced in code, not just convention.

**Zero-shot foundation model.** TabICLv2 uses no gradient-based training on the target dataset. Inside each LOOCV fold, it receives the 45 training participants as its in-context prompt and classifies the held-out participant in a single forward pass.

---

## Configuration

Key parameters at the top of the config cell:

| Parameter | Default | Description |
|---|---|---|
| `FEATURE_FILE` | `trial_features_tabpfn.csv` | Path to input feature CSV |
| `RANDOM_STATE` | `42` | Global random seed |
| `N_ESTIMATORS` | `64` | TabICLv2 ensemble size (use 128 for camera-ready) |
| `N_PERM` | `1000` | Permutation test iterations |
| `N_BOOT` | `2000` | Bootstrap CI iterations |
| `HOLDOUT_RATIO` | `0.20` | Fraction of participants held out |
| `SEV_LOW` | `0.40` | Low/Borderline severity threshold |
| `SEV_HIGH` | `0.70` | Borderline/High severity threshold |
| `OUTPUT_DIR` | `results_paper` | Output directory |

---

## Citation

If you use this pipeline, please cite:

> *[Paper citation — to be added on publication]*

TabICLv2: Qu et al., ICML 2026. arXiv:2602.11139  
XGBoost: Chen & Guestrin, KDD 2016.  
SHAP: Lundberg & Lee, NeurIPS 2017.
