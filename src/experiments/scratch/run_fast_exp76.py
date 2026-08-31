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

print("Loading dataset for Fast Evaluation...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Evaluate Task 1 Interaction Candidates fast using LightGBM + CatBoost + XGBoost
def evaluate_interaction_candidates():
    results = []
    
    cands = [
        ("Baseline (count_x_base 70피처)", None),
        ("Cand 1 (+ count_x_base_x_outs 3중교차)", lambda d, X, cc, b: X.assign(new_feature=cc + '_' + b + '_' + d['outs_before'].fillna(0).astype(int).astype(str))),
        ("Cand 2 (+ base_x_outs 2중교차)", lambda d, X, cc, b: X.assign(new_feature=b + '_' + d['outs_before'].fillna(0).astype(int).astype(str))),
        ("Cand 3 (+ scoring_x_count_x_outs)", lambda d, X, cc, b: X.assign(new_feature=((d['runner_on_2b'].fillna(0)>0)|(d['runner_on_3b'].fillna(0)>0)).astype(int).astype(str) + '_' + cc + '_' + d['outs_before'].fillna(0).astype(int).astype(str)))
    ]

    for cand_name, func in cands:
        lgb_preds, cb_preds, xgb_preds, y_vals = [], [], [], []
        for fi, fold in enumerate(folds):
            df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
            df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
            y_tr = df_tr[config.TARGET_COL].values
            y_va = df_va[config.TARGET_COL].values
            if len(y_vals) < 3: y_vals.append(y_va)

            prep = PitchPreprocessor()
            prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
            X_tr = prep.transform(df_tr)
            X_va = prep.transform(df_va)

            for df_src, X_dst in [(df_tr, X_tr), (df_va, X_va)]:
                base = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                        (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                        (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
                cc = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                      df_src['strikes_before'].fillna(0).astype(int).astype(str))
                X_dst['count_x_base'] = (cc + '_' + base)
                if func is not None:
                    res_df = func(df_src, X_dst, cc, base)
                    X_dst['new_feature'] = res_df['new_feature']

            added_cat = ['count_x_base']
            if func is not None: added_cat.append('new_feature')

            for col in added_cat:
                cat_map = {v: i for i, v in enumerate(X_tr[col].unique())}
                X_tr[col] = X_tr[col].map(cat_map).fillna(-1).astype(int)
                X_va[col] = X_va[col].map(cat_map).fillna(-1).astype(int)

            cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                        or c == config.TRACKMAN_MATCH_FLAG_COL or c in added_cat]
            cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

            # Fast LightGBM
            m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20, colsample_bytree=0.8, subsample=0.8, random_state=42, verbosity=-1, n_jobs=-1)
            m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
            lgb_preds.append(np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1-1e-6))

            # Fast CatBoost
            X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
                X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
            for c in [col for col in X_va_cb.columns if col not in cat_cols]:
                X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
                X_va_cb[c] = X_va_cb[c].astype(np.float32)

            m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=42, verbose=0, cat_features=cat_cols)
            m_cb.fit(X_tr_cb, y_tr)
            cb_preds.append(np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6))

            # Fast XGBoost
            X_tr_xgb, X_va_xgb = X_tr.copy(), X_va.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)
            m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1, eval_metric="logloss")
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

        print(f"[{cand_name}] Inner Brier={inner_brier:.6f} | 3-Fold Brier={mean_brier:.6f} | Skill={mean_skill:.2f}점 | AUC={mean_auc:.6f}")
        results.append({
            "cand_name": cand_name,
            "inner_brier": inner_brier,
            "mean_brier": mean_brier,
            "mean_skill": mean_skill,
            "mean_auc": mean_auc,
            "fold_briers": ens_briers,
            "fold_skills": ens_skills
        })

    return results

res_t1 = evaluate_interaction_candidates()

# Select best via safeguard
best_t1 = submission_checklist.safe_select_best_candidate(res_t1, sort_key="inner_brier", exp_name="Fast Task 1 Interaction Features")

raw_dir = config.OUTPUTS_DIR / 'raw'
with open(raw_dir / 'task1_scoring_position_summary.json', 'w', encoding='utf-8') as f:
    json.dump(res_t1, f, indent=2, ensure_ascii=False)

print("\nTask 1 Fast Evaluation Finished!")
