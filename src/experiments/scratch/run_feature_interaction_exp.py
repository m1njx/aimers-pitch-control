"""
run_feature_interaction_exp.py — Tasks 1, 2, 3 Execution Script
1. Test Candidate Interaction Features (count_x_scoring_pos, count_x_outs, base_x_outs, count_x_base).
2. Test Pitcher As-Of Full-Count Historical Success Rate feature with Bayesian smoothing.
3. Re-confirm final Local SOTA model.
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


def calc_fold_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


def evaluate_feature_set(extra_cat_cols=None, extra_num_df_fn=None, feature_set_name="Default"):
    """Evaluate 3-Fold CV performance of 3-model ensemble (20% LGBM + 70% CatBoost + 10% XGBoost)."""
    df_train = pd.read_csv(config.TRAIN_PATH)
    folds = get_cv_folds(df_train, strategy="time")

    ens_briers = []
    ens_skills = []
    ens_aucs = []

    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        # Add extra categorical features if specified
        if extra_cat_cols:
            for cname in extra_cat_cols:
                if cname == 'count_x_scoring_pos':
                    tr_sp = ((df_tr['runner_on_2b'].fillna(0) > 0) | (df_tr['runner_on_3b'].fillna(0) > 0)).astype(int).astype(str)
                    va_sp = ((df_va['runner_on_2b'].fillna(0) > 0) | (df_va['runner_on_3b'].fillna(0) > 0)).astype(int).astype(str)
                    tr_cc = df_tr['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_tr['strikes_before'].fillna(0).astype(int).astype(str)
                    va_cc = df_va['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_va['strikes_before'].fillna(0).astype(int).astype(str)
                    s_tr = tr_cc + '_' + tr_sp
                    s_va = va_cc + '_' + va_sp

                elif cname == 'count_x_outs':
                    tr_outs = df_tr['outs_before'].fillna(0).astype(int).astype(str)
                    va_outs = df_va['outs_before'].fillna(0).astype(int).astype(str)
                    tr_cc = df_tr['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_tr['strikes_before'].fillna(0).astype(int).astype(str)
                    va_cc = df_va['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_va['strikes_before'].fillna(0).astype(int).astype(str)
                    s_tr = tr_cc + '_' + tr_outs
                    s_va = va_cc + '_' + va_outs

                elif cname == 'base_x_outs':
                    tr_outs = df_tr['outs_before'].fillna(0).astype(int).astype(str)
                    va_outs = df_va['outs_before'].fillna(0).astype(int).astype(str)
                    tr_base = ((df_tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
                    va_base = ((df_va['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
                    s_tr = tr_base + '_' + tr_outs
                    s_va = va_base + '_' + va_outs

                elif cname == 'count_x_base':
                    tr_base = ((df_tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
                    va_base = ((df_va['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' + (df_va['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
                    tr_cc = df_tr['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_tr['strikes_before'].fillna(0).astype(int).astype(str)
                    va_cc = df_va['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_va['strikes_before'].fillna(0).astype(int).astype(str)
                    s_tr = tr_cc + '_' + tr_base
                    s_va = va_cc + '_' + va_base

                # Label Encode to Integer Categories
                cat_map = {val: idx for idx, val in enumerate(s_tr.unique())}
                X_tr[cname] = s_tr.map(cat_map).fillna(-1).astype(int)
                X_va[cname] = s_va.map(cat_map).fillna(-1).astype(int)


        if extra_num_df_fn:
            X_tr, X_va = extra_num_df_fn(df_tr, df_va, fold.fold_max_season, X_tr, X_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL or (extra_cat_cols and c in extra_cat_cols)]
        cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

        # 1. LightGBM (shift = -0.007)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=45, min_child_samples=20,
            learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
        p_lgb = np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)

        # 2. CatBoost (shift = -0.008)
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str) if X_tr_cb[c].dtype != object else X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(int).astype(str) if X_va_cb[c].dtype != object else X_va_cb[c].astype(str)

        for c in [col for col in X_va_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_va_cb[c] = X_va_cb[c].astype(np.float32)

        m_cb = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
            random_seed=42, verbose=0, cat_features=cat_cols
        )
        m_cb.fit(X_tr_cb, y_tr)
        p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1.0 - 1e-6)

        # 3. XGBoost (shift = -0.006)
        X_tr_xgb = X_tr.copy()
        X_va_xgb = X_va.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category')
            X_va_xgb[c] = X_va_xgb[c].astype('category')

        # Convert categories to integer codes for XGBoost stability
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].cat.codes.astype(np.float32)
            X_va_xgb[c] = X_va_xgb[c].cat.codes.astype(np.float32)

        X_tr_xgb = X_tr_xgb.astype(np.float32)
        X_va_xgb = X_va_xgb.astype(np.float32)

        m_xgb = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            colsample_bytree=0.8, subsample=0.8, random_state=42,
            n_jobs=-1, eval_metric="logloss"
        )
        m_xgb.fit(X_tr_xgb, y_tr)
        p_xgb = np.clip(m_xgb.predict_proba(X_va_xgb)[:, 1] - 0.006, 1e-6, 1.0 - 1e-6)

        # 1위 앙상블 조합 (LGBM 20% + CatBoost 70% + XGBoost 10%)
        p_ens = np.clip(0.20 * p_lgb + 0.70 * p_cb + 0.10 * p_xgb, 1e-6, 1.0 - 1e-6)

        b = calc_raw_brier(y_va, p_ens)
        s, _, _, _ = calc_fold_skill_score(y_va, p_ens)
        a = roc_auc_score(y_va, p_ens)

        ens_briers.append(b)
        ens_skills.append(s)
        ens_aucs.append(a)

    mean_b = float(np.mean(ens_briers))
    mean_s = float(np.mean(ens_skills))
    mean_a = float(np.mean(ens_aucs))

    return {
        "feature_set": feature_set_name,
        "fold_briers": ens_briers,
        "mean_brier": mean_b,
        "mean_skill": mean_s,
        "mean_auc": mean_a
    }


print("======================================================================")
print("TASK 1: Interaction Feature Ablation Study")
print("======================================================================")

# 0. Baseline (No extra interaction features)
base_res = evaluate_feature_set(feature_set_name="Baseline (No Extra Interaction)")
print(f"\nBaseline Performance: Raw Brier={base_res['mean_brier']:.6f}, Skill={base_res['mean_skill']:.2f}점, AUC={base_res['mean_auc']:.6f}")

# 1. Candidate 1: count_x_scoring_pos
c1_res = evaluate_feature_set(extra_cat_cols=['count_x_scoring_pos'], feature_set_name="Candidate 1: count_x_scoring_pos")
print(f"Candidate 1 (count_x_scoring_pos): Raw Brier={c1_res['mean_brier']:.6f}, Skill={c1_res['mean_skill']:.2f}점, Diff={c1_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={c1_res['mean_auc']:.6f}")

# 2. Candidate 2: count_x_outs
c2_res = evaluate_feature_set(extra_cat_cols=['count_x_outs'], feature_set_name="Candidate 2: count_x_outs")
print(f"Candidate 2 (count_x_outs)       : Raw Brier={c2_res['mean_brier']:.6f}, Skill={c2_res['mean_skill']:.2f}점, Diff={c2_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={c2_res['mean_auc']:.6f}")

# 3. Candidate 3: base_x_outs
c3_res = evaluate_feature_set(extra_cat_cols=['base_x_outs'], feature_set_name="Candidate 3: base_x_outs")
print(f"Candidate 3 (base_x_outs)        : Raw Brier={c3_res['mean_brier']:.6f}, Skill={c3_res['mean_skill']:.2f}점, Diff={c3_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={c3_res['mean_auc']:.6f}")

# 4. Candidate 4: count_x_base
c4_res = evaluate_feature_set(extra_cat_cols=['count_x_base'], feature_set_name="Candidate 4: count_x_base")
print(f"Candidate 4 (count_x_base)        : Raw Brier={c4_res['mean_brier']:.6f}, Skill={c4_res['mean_skill']:.2f}점, Diff={c4_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={c4_res['mean_auc']:.6f}")


# ------------------------------------------------------------------------------
# TASK 2: Pitcher As-Of Full-Count Historical Success Rate Feature
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 2: Pitcher As-Of Full-Count Historical Feature")
print("======================================================================")

def add_pitcher_fc_history(df_tr, df_va, fold_max_season, X_tr, X_va):
    # Filter training history prior to fold_max_season
    tm_fc = df_tr[(df_tr['balls_before'] == 3) & (df_tr['strikes_before'] == 2)]

    # Global full-count success rate
    global_fc_rate = float(tm_fc[config.TARGET_COL].mean()) if len(tm_fc) > 0 else 0.50

    # Group by pitcher
    p_fc = tm_fc.groupby('pitcher_id')[config.TARGET_COL].agg(['count', 'sum']).reset_index()
    p_fc.columns = ['pitcher_id', 'fc_count', 'fc_sum']

    # Empirical Bayes Smoothing (M = 20)
    M = 20.0
    p_fc['asof_pitcher_fullcount_success_rate'] = (p_fc['fc_sum'] + M * global_fc_rate) / (p_fc['fc_count'] + M)

    tr_merged = df_tr[['pitcher_id']].merge(p_fc[['pitcher_id', 'asof_pitcher_fullcount_success_rate']], on='pitcher_id', how='left')
    va_merged = df_va[['pitcher_id']].merge(p_fc[['pitcher_id', 'asof_pitcher_fullcount_success_rate']], on='pitcher_id', how='left')

    X_tr['asof_pitcher_fullcount_success_rate'] = tr_merged['asof_pitcher_fullcount_success_rate'].fillna(global_fc_rate).astype(np.float32)
    X_va['asof_pitcher_fullcount_success_rate'] = va_merged['asof_pitcher_fullcount_success_rate'].fillna(global_fc_rate).astype(np.float32)

    return X_tr, X_va


fc_res = evaluate_feature_set(extra_num_df_fn=add_pitcher_fc_history, feature_set_name="Pitcher Full-Count History Feature")
print(f"Pitcher Full-Count History Feature: Raw Brier={fc_res['mean_brier']:.6f}, Skill={fc_res['mean_skill']:.2f}점, Diff={fc_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={fc_res['mean_auc']:.6f}")

# Task 3: Best Combo Evaluation
accepted_cats = []
if c1_res['mean_brier'] < base_res['mean_brier']: accepted_cats.append('count_x_scoring_pos')
if c2_res['mean_brier'] < base_res['mean_brier']: accepted_cats.append('count_x_outs')
if c3_res['mean_brier'] < base_res['mean_brier']: accepted_cats.append('base_x_outs')
if c4_res['mean_brier'] < base_res['mean_brier']: accepted_cats.append('count_x_base')

use_fc_fn = add_pitcher_fc_history if fc_res['mean_brier'] < base_res['mean_brier'] else None

print("\n======================================================================")
print(f"TASK 3: Final Accepted Combo Evaluation (Accepted Cats: {accepted_cats}, Use FC: {use_fc_fn is not None})")
print("======================================================================")

if accepted_cats or use_fc_fn:
    final_combo_res = evaluate_feature_set(extra_cat_cols=accepted_cats if accepted_cats else None, extra_num_df_fn=use_fc_fn, feature_set_name="Final Accepted Feature Combo")
    print(f"Final Accepted Feature Combo: Raw Brier={final_combo_res['mean_brier']:.6f}, Skill={final_combo_res['mean_skill']:.2f}점, Diff={final_combo_res['mean_brier'] - base_res['mean_brier']:+.6f}, AUC={final_combo_res['mean_auc']:.6f}")
else:
    final_combo_res = base_res
    print("No candidate feature improved Raw Brier. Baseline 69 features remain optimal!")

summary_dict = {
    "baseline": base_res,
    "candidate1_count_x_scoring_pos": c1_res,
    "candidate2_count_x_outs": c2_res,
    "candidate3_base_x_outs": c3_res,
    "candidate4_count_x_base": c4_res,
    "pitcher_fullcount_history": fc_res,
    "accepted_cat_features": accepted_cats,
    "is_fc_history_accepted": bool(fc_res['mean_brier'] < base_res['mean_brier']),
    "final_combo": final_combo_res
}

with open("~/LG_data/outputs/interaction_exp_summary.json", "w") as f:
    json.dump(summary_dict, f, indent=2, ensure_ascii=False)

print("\nINTERACTION EXPERIMENT SCRIPT COMPLETED SUCCESSFULLY!")
