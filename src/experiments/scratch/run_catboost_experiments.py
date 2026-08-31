"""
run_catboost_experiments.py — CatBoost Baseline, Regularization Search, Recency Shift & Final Selection

Tasks:
  1. CatBoost Baseline 3-Fold Temporal Validation (iterations=300, depth=6, lr=0.05).
  2. CatBoost Regularization Spectrum (depth=4, 6, 8; l2_leaf_reg=3, 5, 10) & Variance Check.
  3. CatBoost Recency Shift Calibration Grid Search (-0.003 to -0.015, step 0.001) & Nested Selection.
  4. Final CatBoost Candidate Selection & Comparison against LightGBM 3rd Submission.
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

import config
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
print("Loading train.csv for CatBoost Experiments ...")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Preprocess X and y for all 3 folds
fold_datasets = []

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

    # Fill NaNs in categorical columns and convert to string for CatBoost
    for c in cat_cols:
        X_tr[c] = X_tr[c].fillna("MISSING").astype(str)
        X_va[c] = X_va[c].fillna("MISSING").astype(str)

    # Convert numerical columns to float32/float64
    num_cols = [c for c in X_va.columns if c not in cat_cols]
    for c in num_cols:
        X_tr[c] = X_tr[c].astype(np.float32)
        X_va[c] = X_va[c].astype(np.float32)

    fold_datasets.append({
        "X_tr": X_tr, "y_tr": y_tr,
        "X_va": X_va, "y_va": y_va,
        "cat_cols": cat_cols,
        "fold": fold
    })

print(f"Preprocessed 3 folds successfully. Features count: {fold_datasets[0]['X_tr'].shape[1]}")

# ------------------------------------------------------------------------------
# TASK 1: CatBoost Baseline Training (depth=6, l2_leaf_reg=3, lr=0.05, n_est=300)
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 1: CatBoost Baseline Training & 3-Fold Temporal CV")
print("======================================================================")

cb_base_briers = []
cb_base_skills = []
cb_base_aucs = []
cb_base_preds_all = []

for fi, fds in enumerate(fold_datasets):
    X_tr, y_tr = fds["X_tr"], fds["y_tr"]
    X_va, y_va = fds["X_va"], fds["y_va"]
    cat_cols = fds["cat_cols"]

    cb_base = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        cat_features=cat_cols
    )
    cb_base.fit(X_tr, y_tr)
    preds = cb_base.predict_proba(X_va)[:, 1]

    brier = calc_raw_brier(y_va, preds)
    skill, _, _, _ = calc_brier_skill_score(y_va, preds)
    auc = roc_auc_score(y_va, preds)

    cb_base_briers.append(brier)
    cb_base_skills.append(skill)
    cb_base_aucs.append(auc)
    cb_base_preds_all.append(preds)

    print(f"Fold {fi} ({fds['fold'].val_season}): Raw Brier={brier:.6f}, Skill={skill:.2f}, AUC={auc:.6f}, Mean Pred={np.mean(preds):.4f}, Std={np.std(preds):.4f}")

task1_res = {
    "mean_raw_brier": float(np.mean(cb_base_briers)),
    "mean_skill": float(np.mean(cb_base_skills)),
    "mean_auc": float(np.mean(cb_base_aucs)),
    "fold_briers": cb_base_briers,
    "fold_skills": cb_base_skills,
    "fold_aucs": cb_base_aucs
}
print(f"CatBoost Baseline 3-Fold Mean -> Raw Brier: {task1_res['mean_raw_brier']:.6f}, Skill: {task1_res['mean_skill']:.2f}, AUC: {task1_res['mean_auc']:.6f}")

# ------------------------------------------------------------------------------
# TASK 2: CatBoost Hyperparameter Regularization Spectrum Search
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 2: CatBoost Regularization Spectrum Search")
print("======================================================================")

hp_candidates = [
    {"name": "CB_Depth4_L2_3", "depth": 4, "l2_leaf_reg": 3.0},
    {"name": "CB_Depth6_L2_3 (Baseline)", "depth": 6, "l2_leaf_reg": 3.0},
    {"name": "CB_Depth6_L2_5", "depth": 6, "l2_leaf_reg": 5.0},
    {"name": "CB_Depth6_L2_10", "depth": 6, "l2_leaf_reg": 10.0},
    {"name": "CB_Depth8_L2_3", "depth": 8, "l2_leaf_reg": 3.0},
]

reg_results = []
candidate_preds = {}

for cand in hp_candidates:
    cname = cand["name"]
    d = cand["depth"]
    l2 = cand["l2_leaf_reg"]

    f_briers = []
    f_skills = []
    f_aucs = []
    c_preds = []

    for fi, fds in enumerate(fold_datasets):
        X_tr, y_tr = fds["X_tr"], fds["y_tr"]
        X_va, y_va = fds["X_va"], fds["y_va"]
        cat_cols = fds["cat_cols"]

        cb_model = CatBoostClassifier(
            iterations=300,
            depth=d,
            learning_rate=0.05,
            l2_leaf_reg=l2,
            random_seed=42,
            verbose=0,
            cat_features=cat_cols
        )
        cb_model.fit(X_tr, y_tr)
        preds = cb_model.predict_proba(X_va)[:, 1]

        brier = calc_raw_brier(y_va, preds)
        skill, _, _, _ = calc_brier_skill_score(y_va, preds)
        auc = roc_auc_score(y_va, preds)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)
        c_preds.append(preds)

    candidate_preds[cname] = c_preds
    all_preds_concat = np.concatenate(c_preds)
    mean_pred = float(np.mean(all_preds_concat))
    std_pred = float(np.std(all_preds_concat))

    # Inner folds (2022, 2023) mean raw Brier
    inner_brier = float(np.mean(f_briers[:2]))

    reg_results.append({
        "candidate": cname,
        "depth": d,
        "l2_leaf_reg": l2,
        "f0_brier": f_briers[0],
        "f1_brier": f_briers[1],
        "f2_brier": f_briers[2],
        "mean_raw_brier": float(np.mean(f_briers)),
        "mean_skill": float(np.mean(f_skills)),
        "mean_auc": float(np.mean(f_aucs)),
        "inner_brier": inner_brier,
        "mean_pred": mean_pred,
        "std_pred": std_pred
    })

reg_df = pd.DataFrame(reg_results)
print("=== CatBoost Regularization Spectrum Results ===")
print(reg_df[["candidate", "depth", "l2_leaf_reg", "inner_brier", "f2_brier", "mean_raw_brier", "mean_skill", "std_pred"]].to_string(index=False))

# Select best candidate on Inner Folds ONLY (Nested Selection)
best_reg_idx = reg_df["inner_brier"].idxmin()
best_reg_cand = reg_results[best_reg_idx]
print(f"\nNested Selection on Inner Folds -> Optimal Candidate: {best_reg_cand['candidate']}")

# ------------------------------------------------------------------------------
# TASK 3: CatBoost Recency Shift Calibration Grid Search
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 3: CatBoost Recency Shift Calibration Grid Search")
print("======================================================================")

best_cand_preds = candidate_preds[best_reg_cand["candidate"]]
SHIFTS = np.round(np.arange(0.000, -0.016, -0.001), 4)

shift_rows = []

for shift in SHIFTS:
    sf_briers = []
    sf_skills = []
    sf_aucs = []

    for fi, fds in enumerate(fold_datasets):
        y_va = fds["y_va"]
        p_shifted = np.clip(best_cand_preds[fi] + shift, 1e-6, 1.0 - 1e-6)

        brier = calc_raw_brier(y_va, p_shifted)
        skill, _, _, _ = calc_brier_skill_score(y_va, p_shifted)
        auc = roc_auc_score(y_va, p_shifted)

        sf_briers.append(brier)
        sf_skills.append(skill)
        sf_aucs.append(auc)

    inner_raw_brier = float(np.mean(sf_briers[:2]))

    shift_rows.append({
        "shift": float(shift),
        "f0_brier": sf_briers[0],
        "f1_brier": sf_briers[1],
        "f2_brier": sf_briers[2],
        "inner_raw_brier": inner_raw_brier,
        "mean_raw_brier": float(np.mean(sf_briers)),
        "mean_skill": float(np.mean(sf_skills)),
        "mean_auc": float(np.mean(sf_aucs))
    })

shift_df = pd.DataFrame(shift_rows)
print("=== CatBoost Shift Calibration Spectrum ===")
print(shift_df[["shift", "inner_raw_brier", "f2_brier", "mean_raw_brier", "mean_skill"]].to_string(index=False))

# Nested Selection of Shift on Inner Folds ONLY
best_shift_idx = shift_df["inner_raw_brier"].idxmin()
best_nested_shift = float(SHIFTS[best_shift_idx])
best_shift_row = shift_rows[best_shift_idx]

print(f"\nNested Selection for CatBoost Shift on Inner Folds -> Optimal Shift: {best_nested_shift:+.4f}")
print(f"CatBoost Final Nested Performance (Shift {best_nested_shift:+.4f}) -> Mean Raw Brier: {best_shift_row['mean_raw_brier']:.6f}, Skill: {best_shift_row['mean_skill']:.2f}")

# Save all task outputs to JSON
all_catboost_data = {
    "task1_baseline": task1_res,
    "task2_regularization": reg_results,
    "best_reg_cand": best_reg_cand,
    "task3_shift_grid": shift_rows,
    "best_nested_shift": best_nested_shift,
    "final_catboost_perf": best_shift_row
}

with open("~/LG_data/outputs/catboost_exp_summary.json", "w") as f:
    json.dump(all_catboost_data, f, indent=2)

print("\n======================================================================")
print("CatBoost Experiments Script Completed Successfully!")
print("======================================================================")
