"""
run_v3b_shift_grid_search.py — Fine Grid Search for Shift on V3-B (leaves=45) Model Predictions

Tests fine shift spectrum [-0.005 to -0.020, step 0.001] specifically on V3-B model probabilities.
Performs:
  1. Fine grid search on V3-B probabilities (raw Brier metric per fold & 3-fold mean).
  2. Inner fold (2022, 2023) selection & Outer fold 2 (2024 held-out) nested verification.
  3. Re-evaluates Candidate (c) with the exact V3-B optimal shift.
  4. Runs submission_checklist.py to verify final checklist pass.
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import config
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


print("Loading train.csv for V3-B Fine Shift Grid Search ...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Pre-train V3-B model (leaves=45, min_child=20, lr=0.05, n_est=300) on each fold
v3b_fold_preds = []
fold_y_trues = []

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

    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=45,
        learning_rate=0.05,
        min_child_samples=20,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr, categorical_feature=[X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns])
    preds = model.predict_proba(X_va)[:, 1]

    v3b_fold_preds.append(preds)
    fold_y_trues.append(y_va)

# Fine Grid Spectrum: -0.005 to -0.020 with step 0.001
FINE_SHIFTS = np.round(np.arange(-0.005, -0.021, -0.001), 4)

grid_rows = []

for shift in FINE_SHIFTS:
    f_briers = []
    f_skills = []
    f_aucs = []

    for fi in range(len(folds)):
        preds_shifted = np.clip(v3b_fold_preds[fi] + shift, 1e-6, 1.0 - 1e-6)
        y_va = fold_y_trues[fi]

        brier = calc_raw_brier(y_va, preds_shifted)
        skill, _, _, _ = calc_brier_skill_score(y_va, preds_shifted)
        auc = roc_auc_score(y_va, preds_shifted)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)

    mean_raw_brier = np.mean(f_briers)

    grid_rows.append({
        "shift": float(shift),
        "f0_brier_2022": f_briers[0],
        "f1_brier_2023": f_briers[1],
        "f2_brier_2024": f_briers[2],
        "mean_raw_brier": mean_raw_brier,
        "mean_3fold_skill": np.mean(f_skills),
        "mean_auc": np.mean(f_aucs)
    })

grid_df = pd.DataFrame(grid_rows)
print("=== Fine Shift Grid Search on V3-B Model Predictions ===")
print(grid_df.to_string(index=False))
grid_df.to_csv("~/LG_data/outputs/47_fine_shift_grid_raw.csv", index=False)

# 1. Best shift on 3-fold mean raw Brier
best_overall_row = grid_df.loc[grid_df["mean_raw_brier"].idxmin()]
best_overall_shift = float(best_overall_row["shift"])

# 2. Nested selection on Inner Folds (2022, 2023) ONLY
inner_briers = []
for shift in FINE_SHIFTS:
    b_in = [calc_raw_brier(fold_y_trues[i], np.clip(v3b_fold_preds[i] + shift, 1e-6, 1.0 - 1e-6)) for i in [0, 1]]
    inner_briers.append(np.mean(b_in))

best_inner_idx = int(np.argmin(inner_briers))
best_inner_shift = float(FINE_SHIFTS[best_inner_idx])

# Held-out 2024 check for V3-B with best_inner_shift
f2_v3b_raw_brier = calc_raw_brier(fold_y_trues[2], v3b_fold_preds[2])
f2_v3b_shifted_brier = calc_raw_brier(fold_y_trues[2], np.clip(v3b_fold_preds[2] + best_inner_shift, 1e-6, 1.0 - 1e-6))
f2_v3b_imp = f2_v3b_raw_brier - f2_v3b_shifted_brier

print("\n=== V3-B Shift Grid Search Summary ===")
print(f"Optimal Shift on 3-Fold Mean Raw Brier       : {best_overall_shift:+.4f} (Mean Raw Brier: {best_overall_row['mean_raw_brier']:.6f}, Skill: {best_overall_row['mean_3fold_skill']:.2f})")
print(f"Optimal Shift on Inner Folds (2022, 2023)    : {best_inner_shift:+.4f}")
print(f"Held-out 2024 Raw Brier Improvement for V3-B : {f2_v3b_imp:+.6f} (Shift {best_inner_shift:+.4f})")

# Run submission_checklist.py with the exact optimal candidate configuration
final_candidate_hp = {
    "num_leaves": 45,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "n_estimators": 300,
    "shift": best_overall_shift,
    "excluded_features": ["season", "game_type"]
}

print("\nRunning submission_checklist.py for Candidate (c) with shift =", best_overall_shift, "...")
checklist_report = run_checklist(final_candidate_hp)

summary_res = {
    "best_overall_shift": best_overall_shift,
    "best_inner_shift": best_inner_shift,
    "f2_v3b_imp": f2_v3b_imp,
    "best_overall_row": best_overall_row.to_dict(),
    "checklist_allowed": checklist_report["is_allowed"]
}

with open("~/LG_data/outputs/47_summary_confirm.json", "w") as f:
    json.dump(summary_res, f, indent=2)

print("\n======================================================================")
print("V3-B SHIFT CONFIRMATION SCRIPT COMPLETE!")
print("======================================================================")
