"""
run_hp_tuning_and_4th_model_exp.py — Task 1, 2, 3 Execution Script
1. Re-tune HPs for LightGBM, CatBoost, XGBoost on updated 70-feature set (count_x_base included).
2. Train 4th diversity model (HistGradientBoosting / ExtraTrees), calculate correlations, tune 4-model ensemble weights.
3. Re-confirm final Local SOTA.
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
import model_config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor


def calc_raw_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def calc_fold_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


print("======================================================================")
print("PREPARATION: Loading Data & Building 70-Feature Preprocessing Artifacts")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Pre-extract 70-feature datasets for all 3 folds
fold_data = []

for fi, fold in enumerate(folds):
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    # Add count_x_base feature (Candidate 4 from Task 69)
    tr_base = ((df_tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    va_base = ((df_va['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    tr_cc = df_tr['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_tr['strikes_before'].fillna(0).astype(int).astype(str)
    va_cc = df_va['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_va['strikes_before'].fillna(0).astype(int).astype(str)

    s_tr = tr_cc + '_' + tr_base
    s_va = va_cc + '_' + va_base

    cat_map = {val: idx for idx, val in enumerate(s_tr.unique())}
    X_tr['count_x_base'] = s_tr.map(cat_map).fillna(-1).astype(int)
    X_va['count_x_base'] = s_va.map(cat_map).fillna(-1).astype(int)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

    fold_data.append({
        "fold_idx": fi,
        "season": fold.val_season,
        "X_tr": X_tr, "X_va": X_va,
        "y_tr": y_tr, "y_va": y_va,
        "cat_cols": cat_cols,
        "cat_idx": cat_idx
    })

print(f"70-Feature Preprocessing Complete across {len(fold_data)} folds.")


# ------------------------------------------------------------------------------
# TASK 1: Hyperparameter Re-tuning for LightGBM, CatBoost, XGBoost
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 1: Hyperparameter Re-tuning on 70-Feature Set")
print("======================================================================")

# 1.1 LightGBM HP Grid Search
lgb_grid = [
    {"num_leaves": 45, "min_child_samples": 20, "colsample_bytree": 0.8, "lr": 0.05, "name": "LGBM Base (leaves=45, min_child=20)"},
    {"num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8, "lr": 0.05, "name": "LGBM Cand 1 (leaves=31, min_child=20)"},
    {"num_leaves": 63, "min_child_samples": 20, "colsample_bytree": 0.8, "lr": 0.05, "name": "LGBM Cand 2 (leaves=63, min_child=20)"},
    {"num_leaves": 45, "min_child_samples": 50, "colsample_bytree": 0.8, "lr": 0.05, "name": "LGBM Cand 3 (leaves=45, min_child=50)"},
    {"num_leaves": 45, "min_child_samples": 20, "colsample_bytree": 0.7, "lr": 0.05, "name": "LGBM Cand 4 (leaves=45, colsample=0.7)"},
]

lgb_tune_results = []
for param in lgb_grid:
    briers, skills, aucs = [], [], []
    for fd in fold_data:
        m = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=param["num_leaves"],
            min_child_samples=param["min_child_samples"],
            learning_rate=param["lr"], colsample_bytree=param["colsample_bytree"],
            subsample=0.8, random_state=42, verbosity=-1, n_jobs=-1
        )
        m.fit(fd["X_tr"], fd["y_tr"], categorical_feature=fd["cat_idx"])
        p = np.clip(m.predict_proba(fd["X_va"])[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)
        briers.append(calc_raw_brier(fd["y_va"], p))
        skills.append(calc_fold_skill_score(fd["y_va"], p)[0])
        aucs.append(roc_auc_score(fd["y_va"], p))

    mb, ms, ma = float(np.mean(briers)), float(np.mean(skills)), float(np.mean(aucs))
    lgb_tune_results.append({"param_name": param["name"], "raw_brier": mb, "skill_score": ms, "mean_auc": ma, "params": param})
    print(f"  {param['name']:<40} : Raw Brier={mb:.6f}, Skill={ms:.2f}점, AUC={ma:.6f}")

best_lgb_hp = sorted(lgb_tune_results, key=lambda x: x["raw_brier"])[0]
print(f"--> Best LightGBM HP: {best_lgb_hp['param_name']} (Raw Brier={best_lgb_hp['raw_brier']:.6f})")

# 1.2 CatBoost HP Grid Search
cb_grid = [
    {"depth": 6, "l2": 10.0, "name": "CatBoost Base (depth=6, l2=10)"},
    {"depth": 5, "l2": 10.0, "name": "CatBoost Cand 1 (depth=5, l2=10)"},
    {"depth": 7, "l2": 10.0, "name": "CatBoost Cand 2 (depth=7, l2=10)"},
    {"depth": 6, "l2": 5.0,  "name": "CatBoost Cand 3 (depth=6, l2=5)"},
    {"depth": 6, "l2": 20.0, "name": "CatBoost Cand 4 (depth=6, l2=20)"},
]

cb_tune_results = []
for param in cb_grid:
    briers, skills, aucs = [], [], []
    for fd in fold_data:
        X_tr_cb = fd["X_tr"].copy()
        X_va_cb = fd["X_va"].copy()
        for c in fd["cat_cols"]:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
        for c in [col for col in X_va_cb.columns if col not in fd["cat_cols"]]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_va_cb[c] = X_va_cb[c].astype(np.float32)

        m = CatBoostClassifier(
            iterations=300, depth=param["depth"], learning_rate=0.05,
            l2_leaf_reg=param["l2"], random_seed=42, verbose=0, cat_features=fd["cat_cols"]
        )
        m.fit(X_tr_cb, fd["y_tr"])
        p = np.clip(m.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1.0 - 1e-6)
        briers.append(calc_raw_brier(fd["y_va"], p))
        skills.append(calc_fold_skill_score(fd["y_va"], p)[0])
        aucs.append(roc_auc_score(fd["y_va"], p))

    mb, ms, ma = float(np.mean(briers)), float(np.mean(skills)), float(np.mean(aucs))
    cb_tune_results.append({"param_name": param["name"], "raw_brier": mb, "skill_score": ms, "mean_auc": ma, "params": param})
    print(f"  {param['name']:<40} : Raw Brier={mb:.6f}, Skill={ms:.2f}점, AUC={ma:.6f}")

best_cb_hp = sorted(cb_tune_results, key=lambda x: x["raw_brier"])[0]
print(f"--> Best CatBoost HP: {best_cb_hp['param_name']} (Raw Brier={best_cb_hp['raw_brier']:.6f})")

# 1.3 XGBoost HP Grid Search
xgb_grid = [
    {"max_depth": 5, "colsample_bytree": 0.8, "name": "XGBoost Base (max_depth=5, colsample=0.8)"},
    {"max_depth": 4, "colsample_bytree": 0.8, "name": "XGBoost Cand 1 (max_depth=4, colsample=0.8)"},
    {"max_depth": 6, "colsample_bytree": 0.8, "name": "XGBoost Cand 2 (max_depth=6, colsample=0.8)"},
    {"max_depth": 5, "colsample_bytree": 0.7, "name": "XGBoost Cand 3 (max_depth=5, colsample=0.7)"},
]

xgb_tune_results = []
for param in xgb_grid:
    briers, skills, aucs = [], [], []
    for fd in fold_data:
        X_tr_xgb = fd["X_tr"].copy()
        X_va_xgb = fd["X_va"].copy()
        for c in fd["cat_cols"]:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)

        X_tr_xgb = X_tr_xgb.astype(np.float32)
        X_va_xgb = X_va_xgb.astype(np.float32)

        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=param["max_depth"], learning_rate=0.05,
            colsample_bytree=param["colsample_bytree"], subsample=0.8,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        m.fit(X_tr_xgb, fd["y_tr"])
        p = np.clip(m.predict_proba(X_va_xgb)[:, 1] - 0.006, 1e-6, 1.0 - 1e-6)
        briers.append(calc_raw_brier(fd["y_va"], p))
        skills.append(calc_fold_skill_score(fd["y_va"], p)[0])
        aucs.append(roc_auc_score(fd["y_va"], p))

    mb, ms, ma = float(np.mean(briers)), float(np.mean(skills)), float(np.mean(aucs))
    xgb_tune_results.append({"param_name": param["name"], "raw_brier": mb, "skill_score": ms, "mean_auc": ma, "params": param})
    print(f"  {param['name']:<40} : Raw Brier={mb:.6f}, Skill={ms:.2f}점, AUC={ma:.6f}")

best_xgb_hp = sorted(xgb_tune_results, key=lambda x: x["raw_brier"])[0]
print(f"--> Best XGBoost HP: {best_xgb_hp['param_name']} (Raw Brier={best_xgb_hp['raw_brier']:.6f})")


# ------------------------------------------------------------------------------
# TASK 2: 4th Diversity Model Exploration (HistGradientBoosting & ExtraTrees)
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 2: 4th Diversity Model Exploration")
print("======================================================================")

# Generate Out-of-fold predictions for Best Tuned 3 Models
lgb_preds_list = []
cb_preds_list = []
xgb_preds_list = []
hgb_preds_list = []

for fd in fold_data:
    # 1. Best LightGBM
    m_lgb = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=best_lgb_hp["params"]["num_leaves"],
        min_child_samples=best_lgb_hp["params"]["min_child_samples"],
        learning_rate=best_lgb_hp["params"]["lr"],
        colsample_bytree=best_lgb_hp["params"]["colsample_bytree"],
        subsample=0.8, random_state=42, verbosity=-1, n_jobs=-1
    )
    m_lgb.fit(fd["X_tr"], fd["y_tr"], categorical_feature=fd["cat_idx"])
    p_lgb = np.clip(m_lgb.predict_proba(fd["X_va"])[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)
    lgb_preds_list.append(p_lgb)

    # 2. Best CatBoost
    X_tr_cb = fd["X_tr"].copy()
    X_va_cb = fd["X_va"].copy()
    for c in fd["cat_cols"]:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
    for c in [col for col in X_va_cb.columns if col not in fd["cat_cols"]]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_va_cb[c] = X_va_cb[c].astype(np.float32)

    m_cb = CatBoostClassifier(
        iterations=300, depth=best_cb_hp["params"]["depth"], learning_rate=0.05,
        l2_leaf_reg=best_cb_hp["params"]["l2"], random_seed=42, verbose=0, cat_features=fd["cat_cols"]
    )
    m_cb.fit(X_tr_cb, fd["y_tr"])
    p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1.0 - 1e-6)
    cb_preds_list.append(p_cb)

    # 3. Best XGBoost
    X_tr_xgb = fd["X_tr"].copy()
    X_va_xgb = fd["X_va"].copy()
    for c in fd["cat_cols"]:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)

    X_tr_xgb = X_tr_xgb.astype(np.float32)
    X_va_xgb = X_va_xgb.astype(np.float32)

    m_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=best_xgb_hp["params"]["max_depth"], learning_rate=0.05,
        colsample_bytree=best_xgb_hp["params"]["colsample_bytree"], subsample=0.8,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    m_xgb.fit(X_tr_xgb, fd["y_tr"])
    p_xgb = np.clip(m_xgb.predict_proba(X_va_xgb)[:, 1] - 0.006, 1e-6, 1.0 - 1e-6)
    xgb_preds_list.append(p_xgb)

    # 4. 4th Model: HistGradientBoostingClassifier
    X_tr_hgb = fd["X_tr"].copy()
    X_va_hgb = fd["X_va"].copy()
    for c in fd["cat_cols"]:
        X_tr_hgb[c] = X_tr_hgb[c].astype(int)
        X_va_hgb[c] = X_va_hgb[c].astype(int)

    m_hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        categorical_features=fd["cat_idx"], random_state=42
    )
    m_hgb.fit(X_tr_hgb, fd["y_tr"])
    p_hgb = np.clip(m_hgb.predict_proba(X_va_hgb)[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)
    hgb_preds_list.append(p_hgb)

# Compute Prediction Pearson Correlations
corr_lgb_cb = [pearsonr(lgb_preds_list[i], cb_preds_list[i])[0] for i in range(3)]
corr_lgb_xgb = [pearsonr(lgb_preds_list[i], xgb_preds_list[i])[0] for i in range(3)]
corr_cb_xgb = [pearsonr(cb_preds_list[i], xgb_preds_list[i])[0] for i in range(3)]

corr_hgb_lgb = [pearsonr(hgb_preds_list[i], lgb_preds_list[i])[0] for i in range(3)]
corr_hgb_cb = [pearsonr(hgb_preds_list[i], cb_preds_list[i])[0] for i in range(3)]
corr_hgb_xgb = [pearsonr(hgb_preds_list[i], xgb_preds_list[i])[0] for i in range(3)]

print("\n--- Prediction Pearson Correlation Matrix (3-Fold Means) ---")
print(f"  LGBM vs CatBoost    : {np.mean(corr_lgb_cb):.4f}")
print(f"  LGBM vs XGBoost     : {np.mean(corr_lgb_xgb):.4f}")
print(f"  CatBoost vs XGBoost : {np.mean(corr_cb_xgb):.4f}")
print(f"  HGB vs LightGBM     : {np.mean(corr_hgb_lgb):.4f}")
print(f"  HGB vs CatBoost     : {np.mean(corr_hgb_cb):.4f}")
print(f"  HGB vs XGBoost      : {np.mean(corr_hgb_xgb):.4f}")

# 4-Model Ensemble Weight Tuning using Nested Validation (Inner Folds 2022-23 ONLY)
print("\n--- 4-Model Ensemble Weight Search (Sorted by Inner Brier) ---")

weight_candidates = []
for i_lgb in range(10, 50, 5):
    for i_cb in range(40, 80, 5):
        for i_xgb in range(5, 30, 5):
            i_hgb = 100 - (i_lgb + i_cb + i_xgb)
            if 0 <= i_hgb <= 20:
                weight_candidates.append((i_lgb / 100.0, i_cb / 100.0, i_xgb / 100.0, i_hgb / 100.0))

# Also add 3-model candidates (w_hgb = 0.0)
for i_lgb in range(10, 50, 5):
    for i_cb in range(40, 85, 5):
        i_xgb = 100 - (i_lgb + i_cb)
        if 5 <= i_xgb <= 30:
            weight_candidates.append((i_lgb / 100.0, i_cb / 100.0, i_xgb / 100.0, 0.0))

four_model_rows = []
for w1, w2, w3, w4 in weight_candidates:
    f_briers, f_skills, f_aucs = [], [], []
    for fi in range(3):
        p_blend = np.clip(w1 * lgb_preds_list[fi] + w2 * cb_preds_list[fi] + w3 * xgb_preds_list[fi] + w4 * hgb_preds_list[fi], 1e-6, 1.0 - 1e-6)
        y_va = fold_data[fi]["y_va"]

        b = calc_raw_brier(y_va, p_blend)
        s, _, _, _ = calc_fold_skill_score(y_va, p_blend)
        a = roc_auc_score(y_va, p_blend)

        f_briers.append(b)
        f_skills.append(s)
        f_aucs.append(a)

    inner_brier = (f_briers[0] + f_briers[1]) / 2.0
    four_model_rows.append({
        "w_lgb": w1, "w_cb": w2, "w_xgb": w3, "w_hgb": w4,
        "inner_brier": inner_brier,
        "outer_f2_brier": f_briers[2],
        "mean_brier": float(np.mean(f_briers)),
        "mean_skill": float(np.mean(f_skills)),
        "mean_auc": float(np.mean(f_aucs))
    })

four_res_df = pd.DataFrame(four_model_rows).sort_values(by="inner_brier")

print("\nTop 5 4-Model Ensemble Weight Candidates:")
print(four_res_df.head(5).to_string(index=False))

best_4model_dict = four_res_df.iloc[0].to_dict()

# ------------------------------------------------------------------------------
# TASK 3: Final Combination Confirmation & Local SOTA Update
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 3: Final Combination Confirmation & Local SOTA Update")
print("======================================================================")

print(f"Confirmed Best Model Weights : LGBM={best_4model_dict['w_lgb']}, CatBoost={best_4model_dict['w_cb']}, XGBoost={best_4model_dict['w_xgb']}, HistGB={best_4model_dict['w_hgb']}")
print(f"3-Fold Raw Brier             : {best_4model_dict['mean_brier']:.6f}")
print(f"Standard CV Skill Score      : {best_4model_dict['mean_skill']:.2f}점")
print(f"Mean AUC                     : {best_4model_dict['mean_auc']:.6f}")

master_summary = {
    "lgb_tuning": lgb_tune_results,
    "best_lgb_hp": best_lgb_hp,
    "cb_tuning": cb_tune_results,
    "best_cb_hp": best_cb_hp,
    "xgb_tuning": xgb_tune_results,
    "best_xgb_hp": best_xgb_hp,
    "correlations": {
        "lgb_cb": float(np.mean(corr_lgb_cb)),
        "lgb_xgb": float(np.mean(corr_lgb_xgb)),
        "cb_xgb": float(np.mean(corr_cb_xgb)),
        "hgb_lgb": float(np.mean(corr_hgb_lgb)),
        "hgb_cb": float(np.mean(corr_hgb_cb)),
        "hgb_xgb": float(np.mean(corr_hgb_xgb))
    },
    "best_4model_ensemble": best_4model_dict,
    "top5_4model_ensemble": four_res_df.head(5).to_dict(orient="records")
}

with open("~/LG_data/outputs/hp_and_4model_summary.json", "w") as f:
    json.dump(master_summary, f, indent=2, ensure_ascii=False)

print("\nHP TUNING & 4TH MODEL DIVERSITY EXPERIMENT COMPLETED SUCCESSFULLY!")
