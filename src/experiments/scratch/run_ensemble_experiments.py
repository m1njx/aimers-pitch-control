"""
run_ensemble_experiments.py — LightGBM + CatBoost Ensemble Correlation, Weight Search, Variance Audit & Packaging

Tasks:
  1. Calculate Pearson correlation between LightGBM and CatBoost fold predictions.
  2. Weight search w_LGBM in [0.3, 0.4, 0.5, 0.6, 0.7] with variance safety check (std >= 0.0560)
     and nested selection on Inner Folds (2022, 2023).
  3. Outer Fold 2 (2024) held-out verification.
  4. Final ensemble packaging (submit_v4.zip) & dummy rehearsal.
"""

import sys, os, time, shutil, zipfile, subprocess, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

import config
import model_config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from submission_checklist import run_checklist


def calc_raw_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


print("======================================================================")
print("Loading train.csv for LightGBM + CatBoost Ensemble Experiments ...")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

lgb_fold_preds = []
cb_fold_preds = []
fold_y_trues = []

for fi, fold in enumerate(folds):
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values
    fold_y_trues.append(y_va)

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

    # --- 1. Train LightGBM Candidate ---
    m_lgb = lgb.LGBMClassifier(
        n_estimators=model_config.LIGHTGBM_CONFIG["params"]["n_estimators"],
        num_leaves=model_config.LIGHTGBM_CONFIG["params"]["num_leaves"],
        learning_rate=model_config.LIGHTGBM_CONFIG["params"]["learning_rate"],
        min_child_samples=model_config.LIGHTGBM_CONFIG["params"]["min_child_samples"],
        colsample_bytree=model_config.LIGHTGBM_CONFIG["params"]["colsample_bytree"],
        subsample=model_config.LIGHTGBM_CONFIG["params"]["subsample"],
        random_state=42, verbosity=-1, n_jobs=-1
    )
    m_lgb.fit(X_tr, y_tr, categorical_feature=[X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns])
    raw_p_lgb = m_lgb.predict_proba(X_va)[:, 1]
    p_lgb = np.clip(raw_p_lgb + model_config.LIGHTGBM_CONFIG["shift"], 1e-6, 1.0 - 1e-6)
    lgb_fold_preds.append(p_lgb)

    # --- 2. Train CatBoost Candidate ---
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].fillna("MISSING").astype(str)
        X_va_cb[c] = X_va_cb[c].fillna("MISSING").astype(str)

    num_cols = [c for c in X_va_cb.columns if c not in cat_cols]
    for c in num_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_va_cb[c] = X_va_cb[c].astype(np.float32)

    m_cb = CatBoostClassifier(
        iterations=model_config.CATBOOST_CONFIG["params"]["iterations"],
        depth=model_config.CATBOOST_CONFIG["params"]["depth"],
        learning_rate=model_config.CATBOOST_CONFIG["params"]["learning_rate"],
        l2_leaf_reg=model_config.CATBOOST_CONFIG["params"]["l2_leaf_reg"],
        random_seed=42, verbose=0, cat_features=cat_cols
    )
    m_cb.fit(X_tr_cb, y_tr)
    raw_p_cb = m_cb.predict_proba(X_va_cb)[:, 1]
    p_cb = np.clip(raw_p_cb + model_config.CATBOOST_CONFIG["shift"], 1e-6, 1.0 - 1e-6)
    cb_fold_preds.append(p_cb)

    print(f"Fold {fi} ({fold.val_season}) Models Fitted successfully.")

# ------------------------------------------------------------------------------
# TASK 1: Correlation Analysis
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 1: Model Prediction Correlation Analysis")
print("======================================================================")

corr_results = []
for fi in range(len(folds)):
    r, pval = pearsonr(lgb_fold_preds[fi], cb_fold_preds[fi])
    corr_results.append({"fold": fi, "val_season": folds[fi].val_season, "pearson_r": float(r), "p_value": float(pval)})
    print(f"Fold {fi} ({folds[fi].val_season}): Pearson r = {r:.4f}")

all_lgb = np.concatenate(lgb_fold_preds)
all_cb = np.concatenate(cb_fold_preds)
overall_r, _ = pearsonr(all_lgb, all_cb)
print(f"Overall Combined Pearson r = {overall_r:.4f}")

# ------------------------------------------------------------------------------
# TASK 2: Ensemble Weight Search & Variance Audit
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 2: Ensemble Weight Search & Variance Safety Audit")
print("======================================================================")

WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]
ensemble_rows = []

for w_lgb in WEIGHTS:
    w_cb = round(1.0 - w_lgb, 2)
    f_briers = []
    f_skills = []
    f_aucs = []

    ens_preds_concat = []

    for fi in range(len(folds)):
        p_ens = np.clip(w_lgb * lgb_fold_preds[fi] + w_cb * cb_fold_preds[fi], 1e-6, 1.0 - 1e-6)
        y_va = fold_y_trues[fi]

        brier = calc_raw_brier(y_va, p_ens)
        skill, _, _, _ = calc_brier_skill_score(y_va, p_ens)
        auc = roc_auc_score(y_va, p_ens)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)
        ens_preds_concat.extend(p_ens)

    ens_preds_concat = np.array(ens_preds_concat)
    mean_raw_brier = float(np.mean(f_briers))
    mean_skill = float(np.mean(f_skills))
    mean_auc = float(np.mean(f_aucs))
    inner_raw_brier = float(np.mean(f_briers[:2]))

    mean_pred = float(np.mean(ens_preds_concat))
    std_pred = float(np.std(ens_preds_concat))
    var_safe = std_pred >= 0.0560

    ensemble_rows.append({
        "w_lgb": w_lgb,
        "w_cb": w_cb,
        "f0_brier": f_briers[0],
        "f1_brier": f_briers[1],
        "f2_brier": f_briers[2],
        "inner_raw_brier": inner_raw_brier,
        "mean_raw_brier": mean_raw_brier,
        "mean_skill": mean_skill,
        "mean_auc": mean_auc,
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "var_safe": var_safe
    })

ens_df = pd.DataFrame(ensemble_rows)
print("=== Ensemble Weight Search Spectrum ===")
print(ens_df[["w_lgb", "w_cb", "inner_raw_brier", "f2_brier", "mean_raw_brier", "mean_skill", "std_pred", "var_safe"]].to_string(index=False))

# Select Best Ensemble Weight on Inner Folds ONLY (Nested Selection)
# Filter candidates where var_safe is True
valid_cand_df = ens_df[ens_df["var_safe"]]
if len(valid_cand_df) > 0:
    best_idx = valid_cand_df["inner_raw_brier"].idxmin()
else:
    best_idx = ens_df["inner_raw_brier"].idxmin()

best_ens_row = ensemble_rows[best_idx]
print(f"\nNested Selection on Inner Folds -> Optimal Weight: LGBM={best_ens_row['w_lgb']:.2f}, CatBoost={best_ens_row['w_cb']:.2f}")
print(f"Optimal Ensemble Performance -> Mean Raw Brier: {best_ens_row['mean_raw_brier']:.6f}, Skill: {best_ens_row['mean_skill']:.2f}, Std Pred: {best_ens_row['std_pred']:.4f}")

# Save json summary for Task 1 and 2
ens_summary = {
    "overall_pearson_r": overall_r,
    "fold_correlations": corr_results,
    "weight_search": ensemble_rows,
    "best_ens_row": best_ens_row
}

with open("~/LG_data/outputs/ensemble_exp_summary.json", "w") as f:
    json.dump(ens_summary, f, indent=2)

print("\nTASK 1 & 2 ENSEMBLE SCRIPT COMPLETE!")
