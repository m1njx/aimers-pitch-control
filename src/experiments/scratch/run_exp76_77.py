import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

sys.path.insert(0, os.path.expanduser('~/LG_data'))
import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
import submission_checklist

warnings.filterwarnings('ignore')

def calc_raw_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def calc_fold_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier

print("Loading dataset...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# =========================================================================
# WORK 1: Scoring Position Detailed Interaction Features (76번)
# =========================================================================
print("\n" + "="*70)
print("[Task 1] 득점권 세분화 교차 피처 실험")
print("="*70)

# 1. Baseline 70-feature predictions (LGBM 20% : CB 70% : XGB 10%)
# Evaluate interaction feature candidates:
# Base 70 features: count_x_base included
# Cand 1: + count_x_base_x_outs (3-way interaction: count_code + base_state + outs_before)
# Cand 2: + base_x_outs (2-way: base_state + outs_before)
# Cand 3: + scoring_x_count_x_outs (is_scoring_pos + count_code + outs_before)

def evaluate_feature_candidate(cand_name, feature_builder_func):
    lgb_preds, cb_preds, xgb_preds, y_vals = [], [], [], []
    
    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values
        y_vals.append(y_va)

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        # Always add count_x_base
        for df_src, X_dst in [(df_tr, X_tr), (df_va, X_va)]:
            base = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                    (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                    (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
            cc = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                  df_src['strikes_before'].fillna(0).astype(int).astype(str))
            X_dst['count_x_base'] = (cc + '_' + base)
            
            if feature_builder_func is not None:
                feature_builder_func(df_src, X_dst, cc, base)

        # Encode categorical cols
        added_cat_cols = ['count_x_base']
        if feature_builder_func is not None and 'new_feature' in X_tr.columns:
            added_cat_cols.append('new_feature')

        for col in added_cat_cols:
            cat_map = {v: i for i, v in enumerate(X_tr[col].unique())}
            X_tr[col] = X_tr[col].map(cat_map).fillna(-1).astype(int)
            X_va[col] = X_va[col].map(cat_map).fillna(-1).astype(int)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c in added_cat_cols]
        cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

        # 1. LGBM (leaves=45)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=45, min_child_samples=20,
            learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
        p_lgb = np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1-1e-6)
        lgb_preds.append(p_lgb)

        # 2. CatBoost (depth=6, l2=10)
        X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
        for c in [col for col in X_va_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_va_cb[c] = X_va_cb[c].astype(np.float32)

        m_cb = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
            random_seed=42, verbose=0, cat_features=cat_cols
        )
        m_cb.fit(X_tr_cb, y_tr)
        p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6)
        cb_preds.append(p_cb)

        # 3. XGBoost (max_depth=5, colsample=0.8)
        X_tr_xgb, X_va_xgb = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_tr_xgb = X_tr_xgb.astype(np.float32)
        X_va_xgb = X_va_xgb.astype(np.float32)

        m_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            colsample_bytree=0.8, subsample=0.8,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        m_xgb.fit(X_tr_xgb, y_tr)
        p_xgb = np.clip(m_xgb.predict_proba(X_va_xgb)[:, 1] - 0.006, 1e-6, 1-1e-6)
        xgb_preds.append(p_xgb)

    # 3-Model Ensemble (0.20 : 0.70 : 0.10)
    ens_briers, ens_skills, ens_aucs = [], [], []
    for fi in range(3):
        p_ens = np.clip(0.20*lgb_preds[fi] + 0.70*cb_preds[fi] + 0.10*xgb_preds[fi], 1e-6, 1-1e-6)
        sk, br, _ = calc_fold_skill_score(y_vals[fi], p_ens)
        auc = roc_auc_score(y_vals[fi], p_ens)
        ens_briers.append(br)
        ens_skills.append(sk)
        ens_aucs.append(auc)

    mean_brier = float(np.mean(ens_briers))
    mean_skill = float(np.mean(ens_skills))
    mean_auc = float(np.mean(ens_aucs))
    inner_brier = float((ens_briers[0] + ens_briers[1]) / 2.0)

    print(f"[{cand_name}] Inner Brier={inner_brier:.6f} | 3-Fold Brier={mean_brier:.6f} | Skill={mean_skill:.2f}점 | AUC={mean_auc:.6f}")

    return {
        "cand_name": cand_name,
        "inner_brier": inner_brier,
        "mean_brier": mean_brier,
        "mean_skill": mean_skill,
        "mean_auc": mean_auc,
        "fold_briers": ens_briers,
        "fold_skills": ens_skills
    }

# Builders
def build_count_x_base_x_outs(df_src, X_dst, cc, base):
    outs = df_src['outs_before'].fillna(0).astype(int).astype(str)
    X_dst['new_feature'] = (cc + '_' + base + '_' + outs)

def build_base_x_outs(df_src, X_dst, cc, base):
    outs = df_src['outs_before'].fillna(0).astype(int).astype(str)
    X_dst['new_feature'] = (base + '_' + outs)

def build_scoring_x_count_x_outs(df_src, X_dst, cc, base):
    is_scoring = ((df_src['runner_on_2b'].fillna(0) > 0) | (df_src['runner_on_3b'].fillna(0) > 0)).astype(int).astype(str)
    outs = df_src['outs_before'].fillna(0).astype(int).astype(str)
    X_dst['new_feature'] = (is_scoring + '_' + cc + '_' + outs)

task1_results = []
task1_results.append(evaluate_feature_candidate("Baseline (count_x_base 70피처)", None))
task1_results.append(evaluate_feature_candidate("Cand 1 (+ count_x_base_x_outs 3중교차)", build_count_x_base_x_outs))
task1_results.append(evaluate_feature_candidate("Cand 2 (+ base_x_outs 2중교차)", build_base_x_outs))
task1_results.append(evaluate_feature_candidate("Cand 3 (+ scoring_x_count_x_outs)", build_scoring_x_count_x_outs))

# Use safeguard to select best
best_task1 = submission_checklist.safe_select_best_candidate(task1_results, sort_key="inner_brier", exp_name="Task 1 Interaction Features")

# Save JSON summary for Task 1
with open(RAW_DIR / 'task1_scoring_position_summary.json', 'w', encoding='utf-8') as f:
    json.dump(task1_results, f, indent=2, ensure_ascii=False)

# =========================================================================
# WORK 2: CatBoost Alternative Boosting & Parameters Exploration (77번)
# =========================================================================
print("\n" + "="*70)
print("[Task 2] CatBoost 대안 설정 (Ordered Boosting 등) 탐색")
print("="*70)

# We evaluate CatBoost alternative parameters on Fold 0, 1, 2:
# Option 1: Default (Plain, depth=6, l2=10)
# Option 2: Ordered Boosting (boosting_type='Ordered', depth=6, l2=10)
# Option 3: Deep Plain (Plain, depth=7, l2=15)
# Option 4: Shallow High-L2 (Plain, depth=5, l2=20)

cb_configs = [
    {"name": "CB Default (Plain, d=6, l2=10)", "boosting_type": "Plain", "depth": 6, "l2": 10.0},
    {"name": "CB Ordered (Ordered, d=6, l2=10)", "boosting_type": "Ordered", "depth": 6, "l2": 10.0},
    {"name": "CB Deep (Plain, d=7, l2=15)", "boosting_type": "Plain", "depth": 7, "l2": 15.0},
    {"name": "CB Shallow (Plain, d=5, l2=20)", "boosting_type": "Plain", "depth": 5, "l2": 20.0},
]

task2_cb_results = []

for cfg in cb_configs:
    cb_preds, y_vals = [], []
    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values
        y_vals.append(y_va)

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        # Add count_x_base
        for df_src, X_dst in [(df_tr, X_tr), (df_va, X_va)]:
            base = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                    (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                    (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
            cc = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                  df_src['strikes_before'].fillna(0).astype(int).astype(str))
            X_dst['count_x_base'] = (cc + '_' + base)

        cat_map = {v: i for i, v in enumerate(X_tr['count_x_base'].unique())}
        X_tr['count_x_base'] = X_tr['count_x_base'].map(cat_map).fillna(-1).astype(int)
        X_va['count_x_base'] = X_va['count_x_base'].map(cat_map).fillna(-1).astype(int)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']

        X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
        for c in [col for col in X_va_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_va_cb[c] = X_va_cb[c].astype(np.float32)

        m_cb = CatBoostClassifier(
            iterations=300, depth=cfg['depth'], learning_rate=0.05, l2_leaf_reg=cfg['l2'],
            boosting_type=cfg['boosting_type'], random_seed=42, verbose=0, cat_features=cat_cols
        )
        m_cb.fit(X_tr_cb, y_tr)
        p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6)
        cb_preds.append(p_cb)

    briers = [calc_raw_brier(y_vals[fi], cb_preds[fi]) for fi in range(3)]
    skills = [calc_fold_skill_score(y_vals[fi], cb_preds[fi])[0] for fi in range(3)]
    aucs = [roc_auc_score(y_vals[fi], cb_preds[fi]) for fi in range(3)]
    inner_br = (briers[0] + briers[1]) / 2.0

    print(f"[{cfg['name']}] Inner Brier={inner_br:.6f} | 3-Fold Brier={np.mean(briers):.6f} | Skill={np.mean(skills):.2f}점 | AUC={np.mean(aucs):.6f}")

    task2_cb_results.append({
        "name": cfg['name'],
        "inner_brier": inner_br,
        "mean_brier": float(np.mean(briers)),
        "mean_skill": float(np.mean(skills)),
        "mean_auc": float(np.mean(aucs)),
        "preds": cb_preds
    })

# Select best CatBoost config using safeguard
best_cb_cfg = submission_checklist.safe_select_best_candidate(task2_cb_results, sort_key="inner_brier", exp_name="Task 2 CatBoost Exploration")

with open(RAW_DIR / 'task2_catboost_alt_summary.json', 'w', encoding='utf-8') as f:
    json.dump([{k: v for k, v in d.items() if k != 'preds'} for d in task2_cb_results], f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("Experiment 76 & 77 Execution Completed Successfully!")
print("="*70)
