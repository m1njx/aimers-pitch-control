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

# Task 1: Scoring Position Detailed Interaction Features
print("\n" + "="*70)
print("[Task 1] 득점권 세분화 교차 피처 빠른 평가")
print("="*70)

# Pre-transform 3 folds once to save time
prep_folds = []
for fi, fold in enumerate(folds):
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    # Base count_x_base
    for df_src, X_dst in [(df_tr, X_tr), (df_va, X_va)]:
        base = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        cc = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_src['strikes_before'].fillna(0).astype(int).astype(str))
        X_dst['count_x_base'] = (cc + '_' + base)
        
        outs = df_src['outs_before'].fillna(0).astype(int).astype(str)
        is_scoring = ((df_src['runner_on_2b'].fillna(0) > 0) | (df_src['runner_on_3b'].fillna(0) > 0)).astype(int).astype(str)
        
        X_dst['count_x_base_x_outs'] = (cc + '_' + base + '_' + outs)
        X_dst['base_x_outs'] = (base + '_' + outs)
        X_dst['scoring_x_count_x_outs'] = (is_scoring + '_' + cc + '_' + outs)

    prep_folds.append({'X_tr': X_tr, 'X_va': X_va, 'y_tr': y_tr, 'y_va': y_va})

def eval_cand_feature(feature_name):
    lgb_preds, cb_preds, xgb_preds, y_vals = [], [], [], []

    for fi in range(3):
        X_tr = prep_folds[fi]['X_tr'].copy()
        X_va = prep_folds[fi]['X_va'].copy()
        y_tr = prep_folds[fi]['y_tr']
        y_va = prep_folds[fi]['y_va']
        y_vals.append(y_va)

        cols_to_use = [c for c in X_tr.columns if c not in ['count_x_base_x_outs', 'base_x_outs', 'scoring_x_count_x_outs']]
        if feature_name is not None:
            cols_to_use.append(feature_name)

        X_tr = X_tr[cols_to_use]
        X_va = X_va[cols_to_use]

        added_cat = ['count_x_base']
        if feature_name is not None: added_cat.append(feature_name)

        for col in added_cat:
            cat_map = {v: i for i, v in enumerate(X_tr[col].unique())}
            X_tr[col] = X_tr[col].map(cat_map).fillna(-1).astype(int)
            X_va[col] = X_va[col].map(cat_map).fillna(-1).astype(int)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c in added_cat]
        cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(n_estimators=200, num_leaves=45, learning_rate=0.05, min_child_samples=20, colsample_bytree=0.8, subsample=0.8, random_state=42, verbosity=-1, n_jobs=-1)
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
        lgb_preds.append(np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1-1e-6))

        # CatBoost
        X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
        for c in [col for col in X_va_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_va_cb[c] = X_va_cb[c].astype(np.float32)

        m_cb = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
        m_cb.fit(X_tr_cb, y_tr)
        cb_preds.append(np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6))

        # XGBoost
        X_tr_xgb, X_va_xgb = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)
        m_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1, eval_metric="logloss")
        m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr)
        xgb_preds.append(np.clip(m_xgb.predict_proba(X_va_xgb.astype(np.float32))[:, 1] - 0.006, 1e-6, 1-1e-6))

    ens_briers, ens_skills, ens_aucs = [], [], []
    for fi in range(3):
        p_ens = np.clip(0.20*lgb_preds[fi] + 0.70*cb_preds[fi] + 0.10*xgb_preds[fi], 1e-6, 1-1e-6)
        sk, br, _ = calc_fold_skill_score(y_vals[fi], p_ens)
        auc = roc_auc_score(y_vals[fi], p_ens)
        ens_briers.append(br)
        ens_skills.append(sk)
        ens_aucs.append(auc)

    inner_brier = float((ens_briers[0] + ens_briers[1]) / 2.0)
    mean_brier = float(np.mean(ens_briers))
    mean_skill = float(np.mean(ens_skills))
    mean_auc = float(np.mean(ens_aucs))

    cname = f"Cand: {feature_name}" if feature_name else "Baseline (count_x_base 70피처)"
    print(f"[{cname}] Inner Brier={inner_brier:.6f} | 3-Fold Brier={mean_brier:.6f} | Skill={mean_skill:.2f}점 | AUC={mean_auc:.6f}")
    return {
        "cand_name": cname,
        "inner_brier": inner_brier,
        "mean_brier": mean_brier,
        "mean_skill": mean_skill,
        "mean_auc": mean_auc,
        "fold_briers": ens_briers,
        "fold_skills": ens_skills
    }

task1_res = []
task1_res.append(eval_cand_feature(None))
task1_res.append(eval_cand_feature("count_x_base_x_outs"))
task1_res.append(eval_cand_feature("base_x_outs"))
task1_res.append(eval_cand_feature("scoring_x_count_x_outs"))

best_t1 = submission_checklist.safe_select_best_candidate(task1_res, sort_key="inner_brier", exp_name="Task 1 Interaction Features")

raw_dir = config.OUTPUTS_DIR / 'raw'
with open(raw_dir / 'task1_scoring_position_summary.json', 'w', encoding='utf-8') as f:
    json.dump(task1_res, f, indent=2, ensure_ascii=False)

# Task 2: CatBoost Alt Configurations
print("\n" + "="*70)
print("[Task 2] CatBoost 대안 파라미터 빠른 평가")
print("="*70)

cb_cfgs = [
    {"name": "CB Default (Plain, d=6, l2=10)", "depth": 6, "l2": 10.0},
    {"name": "CB Deep (Plain, d=7, l2=15)", "depth": 7, "l2": 15.0},
    {"name": "CB Shallow (Plain, d=5, l2=20)", "depth": 5, "l2": 20.0},
    {"name": "CB Low-L2 (Plain, d=6, l2=5)", "depth": 6, "l2": 5.0},
]

task2_res = []
for cfg in cb_cfgs:
    cb_preds, y_vals = [], []
    for fi in range(3):
        X_tr = prep_folds[fi]['X_tr'].copy()
        X_va = prep_folds[fi]['X_va'].copy()
        y_tr = prep_folds[fi]['y_tr']
        y_va = prep_folds[fi]['y_va']
        y_vals.append(y_va)

        cols_to_use = [c for c in X_tr.columns if c not in ['count_x_base_x_outs', 'base_x_outs', 'scoring_x_count_x_outs']]
        X_tr = X_tr[cols_to_use]
        X_va = X_va[cols_to_use]

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

        m_cb = CatBoostClassifier(iterations=200, depth=cfg['depth'], learning_rate=0.05, l2_leaf_reg=cfg['l2'], random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
        m_cb.fit(X_tr_cb, y_tr)
        cb_preds.append(np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6))

    briers = [calc_raw_brier(y_vals[fi], cb_preds[fi]) for fi in range(3)]
    skills = [calc_fold_skill_score(y_vals[fi], cb_preds[fi])[0] for fi in range(3)]
    aucs = [roc_auc_score(y_vals[fi], cb_preds[fi]) for fi in range(3)]
    inner_br = (briers[0] + briers[1]) / 2.0

    print(f"[{cfg['name']}] Inner Brier={inner_br:.6f} | 3-Fold Brier={np.mean(briers):.6f} | Skill={np.mean(skills):.2f}점 | AUC={np.mean(aucs):.6f}")

    task2_res.append({
        "name": cfg['name'],
        "inner_brier": inner_br,
        "mean_brier": float(np.mean(briers)),
        "mean_skill": float(np.mean(skills)),
        "mean_auc": float(np.mean(aucs))
    })

best_t2 = submission_checklist.safe_select_best_candidate(task2_res, sort_key="inner_brier", exp_name="Task 2 CatBoost Alt Exploration")

with open(raw_dir / 'task2_catboost_alt_summary.json', 'w', encoding='utf-8') as f:
    json.dump(task2_res, f, indent=2, ensure_ascii=False)

print("\nAll Tasks Completed Successfully!")
