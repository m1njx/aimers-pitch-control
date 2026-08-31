"""
run_final_pre_submit_fix.py — Pre-Submission Audit, Nested Shift Fix (-0.007), Error Sign Standardization & Packaging

Tasks:
  1. Recalculates 3-fold metrics for Candidate (num_leaves=45, shift=-0.007).
  2. Investigates nested validation improvement discrepancy and reconciles definitions.
  3. Standardizes Error formula (Public LB - Local CV Skill) in submission_history.md and json.
  4. Retrains full model, updates submit_v3.zip and final_code_submission with shift=-0.007.
  5. Verifies 100% identity and runs submission_checklist.py.
"""

import sys, os, time, shutil, zipfile, subprocess, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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
print("TASK 1: Recalculate 3-Fold Metrics for Nested Shift = -0.007")
print("======================================================================")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

nested_shift = -0.007
f_briers = []
f_skills = []
f_aucs = []

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
    raw_preds = model.predict_proba(X_va)[:, 1]
    shifted_preds = np.clip(raw_preds + nested_shift, 1e-6, 1.0 - 1e-6)

    brier = calc_raw_brier(y_va, shifted_preds)
    skill, _, _, _ = calc_brier_skill_score(y_va, shifted_preds)
    auc = roc_auc_score(y_va, shifted_preds)

    f_briers.append(brier)
    f_skills.append(skill)
    f_aucs.append(auc)

    print(f"Fold {fi} ({fold.val_season}): Raw Brier={brier:.6f}, Skill={skill:.2f}, AUC={auc:.6f}")

mean_raw_brier = float(np.mean(f_briers))
mean_skill = float(np.mean(f_skills))
mean_auc = float(np.mean(f_aucs))

print(f"\nFinal Nested Model (leaves=45, shift=-0.007):")
print(f"  Mean Raw Brier : {mean_raw_brier:.6f}")
print(f"  Mean Skill     : {mean_skill:.2f} pts")
print(f"  Mean AUC       : {mean_auc:.6f}")

# Audit Nested Improvement Discrepancy
# Baseline (leaves=63, shift=0) Fold 2 Brier = 0.248834
# V3-B (leaves=45, shift=0) Fold 2 Brier = 0.248701
# V3-B + (-0.007) Fold 2 Brier = 0.248495
# V3-B + (-0.010) Fold 2 Brier = 0.248436
# V3-B + (-0.011) Fold 2 Brier = 0.248421

imp_v3b_vs_unshifted_v3b = 0.248701 - 0.248495  # 0.000206
imp_v3b_shift011_vs_baseline = 0.248834 - 0.248421  # 0.000414
imp_v3b_shift010_vs_baseline = 0.248834 - 0.248436  # 0.000398
imp_v3b_shift007_vs_baseline = 0.248834 - 0.248495  # 0.000339

print("\n--- Nested Improvement Discrepancy Audit ---")
print(f"  1. V3-B(shift=-0.007) vs V3-B(shift=0) 2024 Raw Brier Diff: +{imp_v3b_vs_unshifted_v3b:.6f} (used in 47_final_shift_confirmation.md)")
print(f"  2. V3-B(shift=-0.011) vs Baseline(shift=0) 2024 Raw Brier Diff: +{imp_v3b_shift011_vs_baseline:.6f} (used in checklist Rule 3 with shift=-0.011)")
print(f"  3. V3-B(shift=-0.010) vs Baseline(shift=0) 2024 Raw Brier Diff: +{imp_v3b_shift010_vs_baseline:.6f} (used in daily_candidate_template with shift=-0.010)")
print(f"  4. V3-B(shift=-0.007) vs Baseline(shift=0) 2024 Raw Brier Diff: +{imp_v3b_shift007_vs_baseline:.6f} (unified benchmark vs 1st Submit Baseline)")

audit_res = {
    "nested_shift": nested_shift,
    "f0_brier": f_briers[0], "f1_brier": f_briers[1], "f2_brier": f_briers[2],
    "f0_skill": f_skills[0], "f1_skill": f_skills[1], "f2_skill": f_skills[2],
    "mean_raw_brier": mean_raw_brier,
    "mean_skill": mean_skill,
    "mean_auc": mean_auc,
    "imp_v3b_vs_unshifted_v3b": imp_v3b_vs_unshifted_v3b,
    "imp_v3b_shift007_vs_baseline": imp_v3b_shift007_vs_baseline
}

with open("~/LG_data/outputs/48_audit_raw.json", "w") as f:
    json.dump(audit_res, f, indent=2)

print("\nTASK 1 & 2 AUDIT COMPLETE!")
