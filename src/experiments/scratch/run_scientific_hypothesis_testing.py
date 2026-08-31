"""
run_scientific_hypothesis_testing.py — Scientific Hypothesis Testing & v3 Candidate Design

Tasks:
  1. Bias vs Variance Diagnosis:
     - Extrapolates 2025 control success rate trend (2019-2024 linear & 3-year rolling).
     - Analyzes systematic bias error (p_mean vs expected r_2025).
     - Tests recency-weighted mean shift calibration.
  2. Variance Hypothesis Stress Test:
     - Evaluates ultra-complex models (num_leaves=128, 256, min_child=5, 10) alongside baseline & strong reg.
     - Maps prediction variance vs CV Brier vs AUC to test non-linearity.
  3. Single-Variable Conservative v3 Candidate Design:
     - Compares single-variable candidates against 1st submission baseline.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import linregress, iqr
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor


def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


print("Loading train.csv and test.csv ...")
df_train = pd.read_csv(config.TRAIN_PATH)
df_test = pd.read_csv(config.TEST_PATH)
print(f"Loaded train: {len(df_train):,} rows, test: {len(df_test):,} rows.\n")

folds = get_cv_folds(df_train, strategy="time")

# ==============================================================================
# TASK 1: Base-Rate Extrapolation & Bias vs Variance Analysis
# ==============================================================================
print("======================================================================")
print("TASK 1: Base-Rate Extrapolation & Bias Analysis")
print("======================================================================")

seasons = np.array([2019, 2020, 2021, 2022, 2023, 2024])
rates = np.array([df_train[df_train['season'] == s][config.TARGET_COL].mean() for s in seasons])

slope, intercept, r_val, p_val, std_err = linregress(seasons, rates)
exp_r_2025_linear = slope * 2025 + intercept
exp_r_2025_recent3 = np.mean(rates[-3:])  # Mean of 2022, 2023, 2024
exp_r_2025_recent2 = np.mean(rates[-2:])  # Mean of 2023, 2024

print(f"Historical Rates (2019-2024): {[round(r, 4) for r in rates]}")
print(f"Linear Trend Slope: {slope:+.5f} / year (r_val={r_val:.4f}, p_val={p_val:.4f})")
print(f"Extrapolated 2025 Rate (Linear): {exp_r_2025_linear:.4f}")
print(f"Extrapolated 2025 Rate (Recent 3-yr Mean 2022-24): {exp_r_2025_recent3:.4f}")
print(f"Extrapolated 2025 Rate (Recent 2-yr Mean 2023-24): {exp_r_2025_recent2:.4f}")

# Analyze model prediction mean (0.5238) bias error against 2025 expected rates
pred_mean = 0.52376
bias_linear = pred_mean - exp_r_2025_linear
bias_recent2 = pred_mean - exp_r_2025_recent2

print(f"\nModel Pred Mean: {pred_mean:.4f}")
print(f"Bias vs Linear Extrapolated 2025 ({exp_r_2025_linear:.4f}): {bias_linear:+.4f}")
print(f"Bias vs Recent 2-yr Mean ({exp_r_2025_recent2:.4f}): {bias_recent2:+.4f}")

t1_df = pd.DataFrame([{
    "historical_rates": str([round(r, 4) for r in rates]),
    "slope": slope, "intercept": intercept,
    "exp_r_2025_linear": exp_r_2025_linear,
    "exp_r_2025_recent3": exp_r_2025_recent3,
    "exp_r_2025_recent2": exp_r_2025_recent2,
    "pred_mean": pred_mean,
    "bias_linear": bias_linear,
    "bias_recent2": bias_recent2
}])
t1_df.to_csv("~/LG_data/outputs/41_bias_vs_variance_raw.csv", index=False)


# ==============================================================================
# TASK 2: Variance Hypothesis Stress Test (Ultra Complex to Strong Reg Spectrum)
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Variance Hypothesis Stress Test (Over-complex to Over-regularized)")
print("======================================================================")

SPECTRUM = {
    "1. Ultra Complex 256 (leaves=256, min_child=5)": {
        "leaves": 256, "lr": 0.05, "min_child": 5, "colsample": 0.9, "subsample": 0.9, "n_est": 300
    },
    "2. Complex 128 (leaves=128, min_child=10)": {
        "leaves": 128, "lr": 0.05, "min_child": 10, "colsample": 0.85, "subsample": 0.85, "n_est": 300
    },
    "3. Baseline (leaves=63, min_child=20)": {
        "leaves": 63, "lr": 0.05, "min_child": 20, "colsample": 0.8, "subsample": 0.8, "n_est": 300
    },
    "4. Moderate Reg 31 (leaves=31, min_child=50)": {
        "leaves": 31, "lr": 0.04, "min_child": 50, "colsample": 0.75, "subsample": 0.75, "n_est": 350
    },
    "5. Shallow 31 (leaves=31, min_child=100)": {
        "leaves": 31, "lr": 0.03, "min_child": 100, "colsample": 0.7, "subsample": 0.7, "n_est": 400
    },
    "6. Strong Reg 15 (leaves=15, min_child=500)": {
        "leaves": 15, "lr": 0.02, "min_child": 500, "colsample": 0.6, "subsample": 0.6, "n_est": 400
    }
}

prep_full = PitchPreprocessor()
prep_full.fit(df_train, is_final=True)
X_tr_full = prep_full.transform(df_train)
y_tr_full = df_train[config.TARGET_COL].values
cat_cols_full = [c for c in X_tr_full.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
cat_idx_full = [X_tr_full.columns.get_loc(c) for c in cat_cols_full if c in X_tr_full.columns]

spectrum_results = []

for sname, hp in SPECTRUM.items():
    fold_scores = []
    fold_aucs = []

    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=hp["n_est"],
            num_leaves=hp["leaves"],
            learning_rate=hp["lr"],
            min_child_samples=hp["min_child"],
            colsample_bytree=hp["colsample"],
            subsample=hp["subsample"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=[X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns])
        preds = model.predict_proba(X_va)[:, 1]

        r = float(np.mean(y_va))
        brier = float(np.mean((preds - y_va) ** 2))
        base_brier = float(r * (1.0 - r))
        unclipped = 100000.0 * (1.0 - (brier / base_brier)) if base_brier > 0 else 0.0
        auc = roc_auc_score(y_va, preds)

        fold_scores.append(unclipped)
        fold_aucs.append(auc)

    # Full train prediction for variance
    m_full = lgb.LGBMClassifier(
        n_estimators=hp["n_est"],
        num_leaves=hp["leaves"],
        learning_rate=hp["lr"],
        min_child_samples=hp["min_child"],
        colsample_bytree=hp["colsample"],
        subsample=hp["subsample"],
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    m_full.fit(X_tr_full.values, y_tr_full, categorical_feature=cat_idx_full)
    preds_full = m_full.predict_proba(X_tr_full.values)[:, 1]

    pred_var = float(np.var(preds_full))
    pred_std = float(np.std(preds_full))

    spectrum_results.append({
        "model_name": sname,
        "leaves": hp["leaves"],
        "min_child": hp["min_child"],
        "mean_3fold_unclipped": np.mean(fold_scores),
        "mean_auc": np.mean(fold_aucs),
        "pred_variance": pred_var,
        "pred_std": pred_std
    })

spec_df = pd.DataFrame(spectrum_results)
print(spec_df[["model_name", "mean_3fold_unclipped", "mean_auc", "pred_variance", "pred_std"]].to_string(index=False))
spec_df.to_csv("~/LG_data/outputs/42_variance_stress_test_raw.csv", index=False)


# ==============================================================================
# TASK 3: Single-Variable Controlled v3 Candidate Evaluation
# ==============================================================================
print("\n======================================================================")
print("TASK 3: Single-Variable Controlled v3 Candidates Evaluation")
print("======================================================================")

# Controlled Experiment Candidates relative to Baseline (leaves=63, min_child=20, lr=0.05):
# Candidate V3-A (Single Param Change): min_child=40 ONLY (keeps leaves=63, lr=0.05, colsample=0.8, subsample=0.8)
# Candidate V3-B (Single Param Change): leaves=45 ONLY (keeps min_child=20, lr=0.05, colsample=0.8, subsample=0.8)
# Candidate V3-C (Recency Shift ONLY): Baseline model + post-hoc shift to match 2023-2024 mean (r=0.4930)

V3_CANDIDATES = {
    "Baseline (1차 제출 동일)": {
        "leaves": 63, "min_child": 20, "lr": 0.05, "colsample": 0.8, "subsample": 0.8, "n_est": 300, "shift": 0.0
    },
    "V3-A (Single Change: min_child=40)": {
        "leaves": 63, "min_child": 40, "lr": 0.05, "colsample": 0.8, "subsample": 0.8, "n_est": 300, "shift": 0.0
    },
    "V3-B (Single Change: leaves=45)": {
        "leaves": 45, "min_child": 20, "lr": 0.05, "colsample": 0.8, "subsample": 0.8, "n_est": 300, "shift": 0.0
    },
    "V3-C (Baseline + Recency Base-Rate Shift -0.030)": {
        "leaves": 63, "min_child": 20, "lr": 0.05, "colsample": 0.8, "subsample": 0.8, "n_est": 300, "shift": -0.0307  # shifts 0.5238 -> 0.4931
    }
}

v3_eval_rows = []

for vname, hp in V3_CANDIDATES.items():
    fold_scores = []
    fold_aucs = []

    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=hp["n_est"],
            num_leaves=hp["leaves"],
            learning_rate=hp["lr"],
            min_child_samples=hp["min_child"],
            colsample_bytree=hp["colsample"],
            subsample=hp["subsample"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=[X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns])
        preds = model.predict_proba(X_va)[:, 1] + hp["shift"]
        preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

        r = float(np.mean(y_va))
        brier = float(np.mean((preds - y_va) ** 2))
        base_brier = float(r * (1.0 - r))
        unclipped = 100000.0 * (1.0 - (brier / base_brier)) if base_brier > 0 else 0.0
        auc = roc_auc_score(y_va, preds)

        fold_scores.append(unclipped)
        fold_aucs.append(auc)

    v3_eval_rows.append({
        "candidate": vname,
        "mean_3fold_unclipped": np.mean(fold_scores),
        "f0_skill": fold_scores[0],
        "f1_skill": fold_scores[1],
        "f2_skill": fold_scores[2],
        "mean_auc": np.mean(fold_aucs),
    })

v3_df = pd.DataFrame(v3_eval_rows)
print(v3_df.to_string(index=False))
v3_df.to_csv("~/LG_data/outputs/43_v3_candidate_raw.csv", index=False)

print("\n======================================================================")
print("ALL SCIENTIFIC HYPOTHESIS TESTING COMPLETE!")
print("======================================================================")
