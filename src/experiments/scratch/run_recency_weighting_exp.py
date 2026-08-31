"""
run_recency_weighting_exp.py — Task 4: Recency Sample Weighting vs Post-hoc Shift Experiment
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

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
print("TASK 4: Recency Sample Weighting vs Post-hoc Shift Experiment ...")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

weight_schemes = {
    "Uniform (Standard 1.0)": {2019: 1.0, 2020: 1.0, 2021: 1.0, 2022: 1.0, 2023: 1.0},
    "Linear Decay (0.2->1.0)": {2019: 0.2, 2020: 0.4, 2021: 0.6, 2022: 0.8, 2023: 1.0},
    "Exponential Decay (0.1->1.0)": {2019: 0.1, 2020: 0.2, 2021: 0.4, 2022: 0.7, 2023: 1.0},
    "Recent 3-Season Heavy (0.1->1.0)": {2019: 0.1, 2020: 0.1, 2021: 0.5, 2022: 1.0, 2023: 1.0}
}

scheme_results = []

for sname, sweights in weight_schemes.items():
    print(f"\nTesting Scheme: {sname} ...")
    f_briers_noshift = []
    f_briers_shifted = []
    f_skills_shifted = []
    f_aucs = []

    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        # Compute sample weights based on season
        sample_w_tr = df_tr["season"].map(sweights).fillna(0.5).values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

        m_lgb = lgb.LGBMClassifier(
            n_estimators=model_config.LIGHTGBM_CONFIG["params"]["n_estimators"],
            num_leaves=model_config.LIGHTGBM_CONFIG["params"]["num_leaves"],
            learning_rate=model_config.LIGHTGBM_CONFIG["params"]["learning_rate"],
            min_child_samples=model_config.LIGHTGBM_CONFIG["params"]["min_child_samples"],
            colsample_bytree=model_config.LIGHTGBM_CONFIG["params"]["colsample_bytree"],
            subsample=model_config.LIGHTGBM_CONFIG["params"]["subsample"],
            random_state=42, verbosity=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr, y_tr, sample_weight=sample_w_tr, categorical_feature=cat_idx)

        raw_p = m_lgb.predict_proba(X_va)[:, 1]
        p_noshift = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        p_shifted = np.clip(raw_p - 0.007, 1e-6, 1.0 - 1e-6)

        b_noshift = calc_raw_brier(y_va, p_noshift)
        b_shifted = calc_raw_brier(y_va, p_shifted)
        s_shifted, _, _, _ = calc_brier_skill_score(y_va, p_shifted)
        auc = roc_auc_score(y_va, p_shifted)

        f_briers_noshift.append(b_noshift)
        f_briers_shifted.append(b_shifted)
        f_skills_shifted.append(s_shifted)
        f_aucs.append(auc)

    inner_brier_noshift = float(np.mean(f_briers_noshift[:2]))
    inner_brier_shifted = float(np.mean(f_briers_shifted[:2]))
    outer_f2_brier_shifted = f_briers_shifted[2]

    mean_brier_noshift = float(np.mean(f_briers_noshift))
    mean_brier_shifted = float(np.mean(f_briers_shifted))
    mean_skill_shifted = float(np.mean(f_skills_shifted))
    mean_auc = float(np.mean(f_aucs))

    scheme_results.append({
        "scheme": sname,
        "inner_brier_noshift": inner_brier_noshift,
        "inner_brier_shifted": inner_brier_shifted,
        "outer_f2_brier_shifted": outer_f2_brier_shifted,
        "mean_brier_noshift": mean_brier_noshift,
        "mean_brier_shifted": mean_brier_shifted,
        "mean_skill_shifted": mean_skill_shifted,
        "mean_auc": mean_auc
    })

res_df = pd.DataFrame(scheme_results)
print("\n=== Recency Sample Weighting vs Post-hoc Shift Comparison ===")
print(res_df[["scheme", "inner_brier_shifted", "outer_f2_brier_shifted", "mean_brier_shifted", "mean_skill_shifted", "mean_auc"]].to_string(index=False))

with open("~/LG_data/outputs/recency_weighting_exp.json", "w") as f:
    json.dump(scheme_results, f, indent=2)

print("\nTASK 4 SCRIPT COMPLETE!")
