"""
run_xgboost_ensemble_exp.py — Task 3: 3-Model Ensemble (LightGBM + CatBoost + XGBoost) Exploration
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
import model_config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor


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
print("TASK 3: Training XGBoost & Evaluating 3-Model Ensemble Spectrum ...")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

lgb_preds = []
cb_preds = []
xgb_preds = []
y_trues = []

for fi, fold in enumerate(folds):
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values
    y_trues.append(y_va)

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
    cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

    # --- 1. LightGBM Candidate ---
    m_lgb = lgb.LGBMClassifier(
        n_estimators=model_config.LIGHTGBM_CONFIG["params"]["n_estimators"],
        num_leaves=model_config.LIGHTGBM_CONFIG["params"]["num_leaves"],
        learning_rate=model_config.LIGHTGBM_CONFIG["params"]["learning_rate"],
        min_child_samples=model_config.LIGHTGBM_CONFIG["params"]["min_child_samples"],
        colsample_bytree=model_config.LIGHTGBM_CONFIG["params"]["colsample_bytree"],
        subsample=model_config.LIGHTGBM_CONFIG["params"]["subsample"],
        random_state=42, verbosity=-1, n_jobs=-1
    )
    m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
    raw_p_lgb = m_lgb.predict_proba(X_va)[:, 1]
    p_lgb = np.clip(raw_p_lgb - 0.007, 1e-6, 1.0 - 1e-6)
    lgb_preds.append(p_lgb)

    # --- 2. CatBoost Candidate ---
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_va_cb[c] = X_va_cb[c].astype(int).astype(str)

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
    p_cb = np.clip(raw_p_cb - 0.008, 1e-6, 1.0 - 1e-6)
    cb_preds.append(p_cb)

    # --- 3. XGBoost Candidate ---
    X_tr_xgb = X_tr.astype(np.float32)
    X_va_xgb = X_va.astype(np.float32)

    m_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    m_xgb.fit(X_tr_xgb, y_tr)
    raw_p_xgb = m_xgb.predict_proba(X_va_xgb)[:, 1]
    p_xgb = np.clip(raw_p_xgb - 0.007, 1e-6, 1.0 - 1e-6)
    xgb_preds.append(p_xgb)


    print(f"Fold {fi} ({fold.val_season}) 3 Models Trained successfully.")

# Evaluate XGBoost Standalone 3-Fold performance
xgb_briers = [calc_raw_brier(y_trues[i], xgb_preds[i]) for i in range(3)]
xgb_skills = [calc_brier_skill_score(y_trues[i], xgb_preds[i])[0] for i in range(3)]
xgb_aucs = [roc_auc_score(y_trues[i], xgb_preds[i]) for i in range(3)]

print(f"\nXGBoost Standalone 3-Fold Performance:")
print(f"  Raw Brier   : {np.mean(xgb_briers):.6f}")
print(f"  Skill Score : {np.mean(xgb_skills):.2f}점")
print(f"  Mean AUC    : {np.mean(xgb_aucs):.6f}")

# Compute Pearson correlation matrix across all 3 models
all_lgb = np.concatenate(lgb_preds)
all_cb = np.concatenate(cb_preds)
all_xgb = np.concatenate(xgb_preds)

r_lgb_cb, _ = pearsonr(all_lgb, all_cb)
r_lgb_xgb, _ = pearsonr(all_lgb, all_xgb)
r_cb_xgb, _ = pearsonr(all_cb, all_xgb)

print("\n--- Model Prediction Pearson Correlation Matrix ---")
print(f"  LGBM vs CatBoost : r = {r_lgb_cb:.4f}")
print(f"  LGBM vs XGBoost  : r = {r_lgb_xgb:.4f}")
print(f"  CatBoost vs XGB  : r = {r_cb_xgb:.4f}")

# 3-Model Weight Search Grid
# w_lgb + w_cb + w_xgb = 1.0 (step 0.1)
weight_grid = []
for w1 in np.linspace(0.1, 0.8, 8):
    for w2 in np.linspace(0.1, 0.8, 8):
        w3 = round(1.0 - w1 - w2, 2)
        if 0.1 <= w3 <= 0.8 and round(w1 + w2 + w3, 2) == 1.0:
            weight_grid.append((round(w1, 2), round(w2, 2), w3))

three_model_rows = []
for w1, w2, w3 in weight_grid:
    f_briers = []
    f_skills = []
    f_aucs = []
    preds_concat = []

    for fi in range(3):
        p_ens = np.clip(w1 * lgb_preds[fi] + w2 * cb_preds[fi] + w3 * xgb_preds[fi], 1e-6, 1.0 - 1e-6)
        y_va = y_trues[fi]

        brier = calc_raw_brier(y_va, p_ens)
        skill, _, _, _ = calc_brier_skill_score(y_va, p_ens)
        auc = roc_auc_score(y_va, p_ens)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)
        preds_concat.extend(p_ens)

    inner_brier = float(np.mean(f_briers[:2]))
    mean_brier = float(np.mean(f_briers))
    mean_skill = float(np.mean(f_skills))
    mean_auc = float(np.mean(f_aucs))
    std_pred = float(np.std(preds_concat))

    three_model_rows.append({
        "w_lgb": w1, "w_cb": w2, "w_xgb": w3,
        "inner_brier": inner_brier,
        "outer_f2_brier": f_briers[2],
        "mean_brier": mean_brier,
        "mean_skill": mean_skill,
        "mean_auc": mean_auc,
        "std_pred": std_pred
    })

three_df = pd.DataFrame(three_model_rows).sort_values(by="inner_brier")
print("\nTop 5 3-Model Ensemble Candidates (Nested Selection on Inner Folds):")
print(three_df.head(5).to_string(index=False))

best_3m = three_df.iloc[0].to_dict()

res_summary = {
    "xgb_standalone": {
        "raw_brier": float(np.mean(xgb_briers)),
        "skill_score": float(np.mean(xgb_skills)),
        "auc": float(np.mean(xgb_aucs))
    },
    "correlation_matrix": {
        "lgb_cb": float(r_lgb_cb),
        "lgb_xgb": float(r_lgb_xgb),
        "cb_xgb": float(r_cb_xgb)
    },
    "best_3model_ensemble": best_3m,
    "top_candidates": three_df.head(10).to_dict(orient="records")
}

with open("~/LG_data/outputs/xgboost_ensemble_exp.json", "w") as f:
    json.dump(res_summary, f, indent=2)

print("\nTASK 3 SCRIPT COMPLETE!")
