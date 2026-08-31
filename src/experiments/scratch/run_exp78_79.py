import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
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

print("Loading dataset for Stacking and Ceiling Analysis...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Generate OOF predictions for 3 Base Models (LGBM, CatBoost, XGBoost)
oof_lgb, oof_cb, oof_xgb, y_folds = [], [], [], []

for fi, fold in enumerate(folds):
    print(f"--- Generating Base Model OOF for Fold {fi} (val season={fold.val_season}) ---")
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values
    y_folds.append(y_va)

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    # count_x_base
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
    cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20, colsample_bytree=0.8, subsample=0.8, random_state=42, verbosity=-1, n_jobs=-1)
    m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
    oof_lgb.append(np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1-1e-6))

    # CatBoost
    X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
    for c in [col for col in X_va_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_va_cb[c] = X_va_cb[c].astype(np.float32)

    m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
    m_cb.fit(X_tr_cb, y_tr)
    oof_cb.append(np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6))

    # XGBoost
    X_tr_xgb, X_va_xgb = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)
    m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1, eval_metric="logloss")
    m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr)
    oof_xgb.append(np.clip(m_xgb.predict_proba(X_va_xgb.astype(np.float32))[:, 1] - 0.006, 1e-6, 1-1e-6))

# =========================================================================
# WORK 1: Stacking Ensemble Evaluation (78번)
# =========================================================================
print("\n" + "="*70)
print("[Task 1] Stacking Ensemble (Meta Models) Evaluation")
print("="*70)

# Build meta features matrix per fold: Meta X = [oof_lgb, oof_cb, oof_xgb]
meta_X_folds = []
for fi in range(3):
    meta_X_folds.append(np.column_stack([oof_lgb[fi], oof_cb[fi], oof_xgb[fi]]))

# Inner Folds (Fold 0, Fold 1) meta train data
meta_X_inner = np.vstack([meta_X_folds[0], meta_X_folds[1]])
meta_y_inner = np.concatenate([y_folds[0], y_folds[1]])

# 1. Baseline Weighted Blending (0.20 : 0.70 : 0.10)
blending_briers, blending_skills = [], []
for fi in range(3):
    p_blend = np.clip(0.20*oof_lgb[fi] + 0.70*oof_cb[fi] + 0.10*oof_xgb[fi], 1e-6, 1-1e-6)
    sk, br, _ = calc_fold_skill_score(y_folds[fi], p_blend)
    blending_briers.append(br)
    blending_skills.append(sk)

blending_inner_br = (blending_briers[0] + blending_briers[1]) / 2.0
blending_mean_br = np.mean(blending_briers)
blending_mean_sk = np.mean(blending_skills)

stacking_candidates = [
    {
        "name": "Weighted Blending Baseline (20:70:10)",
        "inner_brier": blending_inner_br,
        "mean_brier": blending_mean_br,
        "mean_skill": blending_mean_sk,
        "fold_briers": blending_briers
    }
]

# 2. Meta Model 1: Logistic Regression Meta Learner
lr_meta = LogisticRegression(C=1.0, max_iter=500, random_state=42)
lr_meta.fit(meta_X_inner, meta_y_inner)

lr_briers, lr_skills = [], []
for fi in range(3):
    p_lr = np.clip(lr_meta.predict_proba(meta_X_folds[fi])[:, 1], 1e-6, 1-1e-6)
    sk, br, _ = calc_fold_skill_score(y_folds[fi], p_lr)
    lr_briers.append(br)
    lr_skills.append(sk)

stacking_candidates.append({
    "name": "Stacking (LogisticRegression Meta-Learner C=1.0)",
    "inner_brier": (lr_briers[0] + lr_briers[1]) / 2.0,
    "mean_brier": float(np.mean(lr_briers)),
    "mean_skill": float(np.mean(lr_skills)),
    "fold_briers": lr_briers
})

# 3. Meta Model 2: Ridge Regression Meta Learner
ridge_meta = Ridge(alpha=10.0, random_state=42)
ridge_meta.fit(meta_X_inner, meta_y_inner)

ridge_briers, ridge_skills = [], []
for fi in range(3):
    p_ridge = np.clip(ridge_meta.predict(meta_X_folds[fi]), 1e-6, 1-1e-6)
    sk, br, _ = calc_fold_skill_score(y_folds[fi], p_ridge)
    ridge_briers.append(br)
    ridge_skills.append(sk)

stacking_candidates.append({
    "name": "Stacking (Ridge Linear Meta-Learner alpha=10.0)",
    "inner_brier": (ridge_briers[0] + ridge_briers[1]) / 2.0,
    "mean_brier": float(np.mean(ridge_briers)),
    "mean_skill": float(np.mean(ridge_skills)),
    "fold_briers": ridge_briers
})

# 4. Meta Model 3: Shallow LightGBM Meta Learner (depth=2)
m_lgb_meta = lgb.LGBMClassifier(n_estimators=50, max_depth=2, num_leaves=4, learning_rate=0.03, random_state=42, verbosity=-1)
m_lgb_meta.fit(meta_X_inner, meta_y_inner)

lgb_meta_briers, lgb_meta_skills = [], []
for fi in range(3):
    p_meta = np.clip(m_lgb_meta.predict_proba(meta_X_folds[fi])[:, 1], 1e-6, 1-1e-6)
    sk, br, _ = calc_fold_skill_score(y_folds[fi], p_meta)
    lgb_meta_briers.append(br)
    lgb_meta_skills.append(sk)

stacking_candidates.append({
    "name": "Stacking (Shallow LightGBM Meta-Learner depth=2)",
    "inner_brier": (lgb_meta_briers[0] + lgb_meta_skills[1]) / 2.0,
    "mean_brier": float(np.mean(lgb_meta_briers)),
    "mean_skill": float(np.mean(lgb_meta_skills)),
    "fold_briers": lgb_meta_briers
})

for cand in stacking_candidates:
    print(f"[{cand['name']}] Inner Brier={cand['inner_brier']:.6f} | 3-Fold Brier={cand['mean_brier']:.6f} | Skill={cand['mean_skill']:.2f}점")

best_stacking = submission_checklist.safe_select_best_candidate(stacking_candidates, sort_key="inner_brier", exp_name="Task 1 Stacking Ensemble")

raw_dir = config.OUTPUTS_DIR / 'raw'
with open(raw_dir / 'task1_stacking_summary.json', 'w', encoding='utf-8') as f:
    json.dump(stacking_candidates, f, indent=2, ensure_ascii=False)

# =========================================================================
# WORK 2: Predictability Ceiling & Variance Decomposition Analysis (79번)
# =========================================================================
print("\n" + "="*70)
print("[Task 2] 타겟(control_success) 이론적 예측 한계(Predictability Ceiling) 분석")
print("="*70)

# Calculate empirical Bayes optimal error by grouping similar situations
# Grouping factors: [pitcher_id, batter_id, count_code, base_state]
df_all = df_train.copy()
df_all['base_state_str'] = ((df_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                            (df_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                            (df_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
df_all['count_code_str'] = (df_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                            df_all['strikes_before'].fillna(0).astype(int).astype(str))

# Group 1: Pitcher x Count x Base State
grp1 = df_all.groupby(['pitcher_id', 'count_code_str', 'base_state_str'])[config.TARGET_COL].agg(['count', 'mean']).reset_index()
grp1_valid = grp1[grp1['count'] >= 5]  # at least 5 instances
weighted_bayes_brier_grp1 = float(np.sum(grp1_valid['count'] * grp1_valid['mean'] * (1.0 - grp1_valid['mean'])) / np.sum(grp1_valid['count']))

# Group 2: Count x Base State x Outs (Global situation ceiling)
grp2 = df_all.groupby(['count_code_str', 'base_state_str', 'outs_before'])[config.TARGET_COL].agg(['count', 'mean']).reset_index()
weighted_bayes_brier_grp2 = float(np.sum(grp2['count'] * grp2['mean'] * (1.0 - grp2['mean'])) / np.sum(grp2['count']))

# Global baseline brier
global_r = float(df_all[config.TARGET_COL].mean())
global_baseline_brier = float(global_r * (1.0 - global_r))

# Skill ceiling estimates
skill_ceiling_grp1 = float(100000.0 * (1.0 - (weighted_bayes_brier_grp1 / global_baseline_brier)))
skill_ceiling_grp2 = float(100000.0 * (1.0 - (weighted_bayes_brier_grp2 / global_baseline_brier)))

current_sota_brier = 0.247513
current_sota_skill = 859.63
gap_to_brier_limit = current_sota_brier - weighted_bayes_brier_grp1

print(f"Global Base Rate r = {global_r:.6f} | Baseline Brier = {global_baseline_brier:.6f}")
print(f"Group 1 (Pitcher x Count x Base, N>=5) Bayes Brier Ceiling = {weighted_bayes_brier_grp1:.6f} | Skill Ceiling = {skill_ceiling_grp1:.2f}점")
print(f"Group 2 (Count x Base x Outs) Bayes Brier Ceiling        = {weighted_bayes_brier_grp2:.6f} | Skill Ceiling = {skill_ceiling_grp2:.2f}점")
print(f"Current Local SOTA Brier = {current_sota_brier:.6f} | Skill = {current_sota_skill:.2f}점")
print(f"Remaining Distance to Bayes Optimal Ceiling: {gap_to_brier_limit:.6f} (Only {gap_to_brier_limit/global_baseline_brier*100:.3f}% of total variance remaining!)")

ceiling_summary = {
    "global_r": global_r,
    "global_baseline_brier": global_baseline_brier,
    "bayes_brier_pitcher_situation": weighted_bayes_brier_grp1,
    "bayes_skill_ceiling_pitcher_situation": skill_ceiling_grp1,
    "bayes_brier_situation_global": weighted_bayes_brier_grp2,
    "bayes_skill_ceiling_situation_global": skill_ceiling_grp2,
    "current_sota_brier": current_sota_brier,
    "current_sota_skill": current_sota_skill,
    "remaining_brier_gap": gap_to_brier_limit,
    "explained_variance_ratio": float(1.0 - (current_sota_brier - weighted_bayes_brier_grp1) / (global_baseline_brier - weighted_bayes_brier_grp1))
}

with open(raw_dir / 'task2_predictability_ceiling_summary.json', 'w', encoding='utf-8') as f:
    json.dump(ceiling_summary, f, indent=2, ensure_ascii=False)

print("\nStacking and Predictability Ceiling Evaluation Finished Successfully!")
