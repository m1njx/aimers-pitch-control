"""
run_failure_diagnosis_and_rollback.py — Diagnosis of 2nd Submission Drop (714.78 -> 684.98)

Tasks:
  1. Prediction Distribution Audit:
     Predicts full train.csv (1.47M rows) & test.csv with Baseline (leaves=63, min_child=20) vs Strong Reg 15 (leaves=15, min_child=500).
     Computes mean, std, min, max, IQR, variance.
  2. Seasonal Representativeness Analysis:
     Calculates season-by-season stats (control_success rate, feature means) for 2019-2024.
  3. Regularization Rollback & Candidate Re-evaluation:
     Evaluates 5 candidate configurations from Baseline to Strong Reg on CV score, AUC, and prediction variance ratio.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import iqr
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
model_features_69 = config.MODEL_FEATURE_COLS

# ==============================================================================
# TASK 1: Prediction Distribution Audit (Baseline vs Strong Reg 15)
# ==============================================================================
print("======================================================================")
print("TASK 1: Prediction Distribution Audit on Train (1.47M rows)")
print("======================================================================")

# Fit PitchPreprocessor on full train
prep_final = PitchPreprocessor()
prep_final.fit(df_train, is_final=True)
X_train_full = prep_final.transform(df_train)
y_train_full = df_train[config.TARGET_COL].values

cat_cols_full = [c for c in X_train_full.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
cat_indices = [X_train_full.columns.get_loc(c) for c in cat_cols_full if c in X_train_full.columns]

# Train Baseline (leaves=63, lr=0.05, min_child=20, n_est=300)
print("Training Baseline (leaves=63, min_child=20)...")
m_base = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, learning_rate=0.05,
    min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
    random_state=42, verbosity=-1, n_jobs=-1
)
m_base.fit(X_train_full.values, y_train_full, categorical_feature=cat_indices)
preds_base_train = m_base.predict_proba(X_train_full.values)[:, 1]

# Train Strong Reg 15 (leaves=15, lr=0.02, min_child=500, n_est=400)
print("Training Strong Reg 15 (leaves=15, min_child=500)...")
m_strong = lgb.LGBMClassifier(
    n_estimators=400, num_leaves=15, learning_rate=0.02,
    min_child_samples=500, colsample_bytree=0.6, subsample=0.6,
    random_state=42, verbosity=-1, n_jobs=-1
)
m_strong.fit(X_train_full.values, y_train_full, categorical_feature=cat_indices)
preds_strong_train = m_strong.predict_proba(X_train_full.values)[:, 1]

def get_dist_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "iqr": float(iqr(arr))
    }

dist_base = get_dist_stats(preds_base_train)
dist_strong = get_dist_stats(preds_strong_train)

print("\n--- Train Prediction Distribution Comparison ---")
print(f"Baseline   (leaves=63): Mean={dist_base['mean']:.4f}, Std={dist_base['std']:.4f}, Var={dist_base['variance']:.6f}, Min={dist_base['min']:.4f}, Max={dist_base['max']:.4f}, IQR={dist_base['iqr']:.4f}")
print(f"Strong Reg (leaves=15): Mean={dist_strong['mean']:.4f}, Std={dist_strong['std']:.4f}, Var={dist_strong['variance']:.6f}, Min={dist_strong['min']:.4f}, Max={dist_strong['max']:.4f}, IQR={dist_strong['iqr']:.4f}")
print(f"Variance Shrinkage Ratio (Strong / Base): {dist_strong['variance'] / dist_base['variance']:.4f} (Strong Reg lost {(1.0 - dist_strong['variance'] / dist_base['variance'])*100:.1f}% of variance!)")

t1_dist_df = pd.DataFrame([
    {"model": "Baseline (leaves=63, min_child=20)", **dist_base},
    {"model": "Strong Reg 15 (leaves=15, min_child=500)", **dist_strong}
])
t1_dist_df.to_csv("~/LG_data/outputs/38_prediction_dist_raw.csv", index=False)


# ==============================================================================
# TASK 2: Seasonal Representativeness Analysis (2019-2024)
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Season-by-Season Statistics & Fluctuation Analysis")
print("======================================================================")

season_stats = []
for s in sorted(df_train['season'].unique()):
    df_s = df_train[df_train['season'] == s]
    r_s = float(df_s[config.TARGET_COL].mean())
    n_s = len(df_s)
    p_succ_mean = float(df_s['asof_pitcher_success_rate'].mean()) if 'asof_pitcher_success_rate' in df_s else np.nan
    b_succ_mean = float(df_s['asof_batter_success_rate'].mean()) if 'asof_batter_success_rate' in df_s else np.nan
    exh_ratio = float((df_s['game_type'] == 'F').mean()) if 'game_type' in df_s else np.nan
    season_stats.append({
        "season": int(s),
        "count": n_s,
        "control_success_rate": r_s,
        "base_brier": r_s * (1.0 - r_s),
        "pitcher_success_rate_mean": p_succ_mean,
        "batter_success_rate_mean": b_succ_mean,
        "exhibition_ratio": exh_ratio
    })

season_df = pd.DataFrame(season_stats)
print(season_df.to_string(index=False))
season_df.to_csv("~/LG_data/outputs/39_season_representativeness_raw.csv", index=False)


# ==============================================================================
# TASK 3: Regularization Rollback & Candidate Re-evaluation
# ==============================================================================
print("\n======================================================================")
print("TASK 3: Regularization Rollback & Candidate Spectrum Evaluation")
print("======================================================================")

CANDIDATES = {
    "1. Baseline (leaves=63, lr=0.05, min_child=20)": {
        "leaves": 63, "lr": 0.05, "min_child": 20, "colsample": 0.8, "subsample": 0.8, "n_est": 300
    },
    "2. Moderate Reg 31 (leaves=31, lr=0.04, min_child=50)": {
        "leaves": 31, "lr": 0.04, "min_child": 50, "colsample": 0.75, "subsample": 0.75, "n_est": 350
    },
    "3. Shallow 31 (leaves=31, lr=0.03, min_child=100)": {
        "leaves": 31, "lr": 0.03, "min_child": 100, "colsample": 0.7, "subsample": 0.7, "n_est": 400
    },
    "4. Shallow 15 (leaves=15, lr=0.03, min_child=200)": {
        "leaves": 15, "lr": 0.03, "min_child": 200, "colsample": 0.7, "subsample": 0.7, "n_est": 400
    },
    "5. Strong Reg 15 (leaves=15, lr=0.02, min_child=500)": {
        "leaves": 15, "lr": 0.02, "min_child": 500, "colsample": 0.6, "subsample": 0.6, "n_est": 400
    }
}

cand_eval_rows = []

for cname, hp in CANDIDATES.items():
    fold_scores = []
    fold_aucs = []
    val_preds_list = []

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
        unclipped_skill = 100000.0 * (1.0 - (brier / base_brier)) if base_brier > 0 else 0.0
        auc = roc_auc_score(y_va, preds)

        fold_scores.append(unclipped_skill)
        fold_aucs.append(auc)
        val_preds_list.append(preds)

    # Full train prediction to measure variance retention
    model_full = lgb.LGBMClassifier(
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
    model_full.fit(X_train_full.values, y_train_full, categorical_feature=cat_indices)
    preds_full = model_full.predict_proba(X_train_full.values)[:, 1]

    pred_var = float(np.var(preds_full))
    pred_std = float(np.std(preds_full))
    pred_iqr = float(iqr(preds_full))

    cand_eval_rows.append({
        "candidate": cname,
        "mean_3fold_unclipped": float(np.mean(fold_scores)),
        "f0_skill": fold_scores[0],
        "f1_skill": fold_scores[1],
        "f2_skill": fold_scores[2],
        "mean_auc": float(np.mean(fold_aucs)),
        "pred_var": pred_var,
        "pred_std": pred_std,
        "pred_iqr": pred_iqr,
        "var_retention_ratio": pred_var / dist_base["variance"]
    })

cand_eval_df = pd.DataFrame(cand_eval_rows)
print("\n--- Rollback Candidate Evaluation Summary ---")
print(cand_eval_df[["candidate", "mean_3fold_unclipped", "mean_auc", "pred_var", "var_retention_ratio"]].to_string(index=False))
cand_eval_df.to_csv("~/LG_data/outputs/40_rollback_check_raw.csv", index=False)

print("\n======================================================================")
print("DIAGNOSIS SCRIPT COMPLETE!")
print("======================================================================")
