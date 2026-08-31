"""
run_shift_reeval_and_nested_validation.py — Recency Shift Re-evaluation with Fixed Metric & Nested Validation

Tasks:
  1. Fixed Metric Shift Re-evaluation:
     - Tests shift spectrum [0.00, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06] on Baseline (leaves=63).
     - Measures raw Brier score per fold (F0: 2022, F1: 2023, F2: 2024) and equal-weighted mean raw Brier.
  2. Nested Validation:
     - Selects optimal shift using Folds 0 & 1 (2022, 2023) ONLY.
     - Evaluates selected shift on held-out Fold 2 (2024).
  3. Candidate Comparison:
     - Compares (a) V3-B standalone (leaves=45), (b) Baseline + optimal shift, (c) V3-B + optimal shift.
"""

import sys, os, time, warnings
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


print("Loading train.csv for Shift Re-evaluation ...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# ==============================================================================
# TASK 1: Shift Spectrum Re-evaluation on Baseline (Raw Brier Metric)
# ==============================================================================
print("======================================================================")
print("TASK 1: Shift Spectrum Re-evaluation on Baseline (Raw Brier Metric)")
print("======================================================================")

SHIFTS = [0.00, -0.005, -0.010, -0.015, -0.020, -0.025, -0.030, -0.035, -0.040, -0.050]

# Pre-train Baseline model on each fold to get raw probabilities
fold_raw_preds = []
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
        num_leaves=63,
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

    fold_raw_preds.append(preds)
    fold_y_trues.append(y_va)

t1_rows = []

for shift in SHIFTS:
    f_briers = []
    f_skills = []
    f_aucs = []

    for fi in range(len(folds)):
        preds_shifted = np.clip(fold_raw_preds[fi] + shift, 1e-6, 1.0 - 1e-6)
        y_va = fold_y_trues[fi]

        brier = calc_raw_brier(y_va, preds_shifted)
        skill, _, _, _ = calc_brier_skill_score(y_va, preds_shifted)
        auc = roc_auc_score(y_va, preds_shifted)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)

    mean_raw_brier = np.mean(f_briers)
    base_briers = [calc_raw_brier(fold_y_trues[i], fold_raw_preds[i]) for i in range(len(folds))]
    mean_base_brier = np.mean(base_briers)
    brier_improvement = mean_base_brier - mean_raw_brier

    t1_rows.append({
        "shift": shift,
        "f0_brier_2022": f_briers[0],
        "f1_brier_2023": f_briers[1],
        "f2_brier_2024": f_briers[2],
        "mean_raw_brier": mean_raw_brier,
        "brier_improvement": brier_improvement,
        "mean_3fold_skill": np.mean(f_skills),
        "mean_auc": np.mean(f_aucs)
    })

t1_df = pd.DataFrame(t1_rows)
print(t1_df.to_string(index=False))
t1_df.to_csv("~/LG_data/outputs/44_shift_reeval_raw.csv", index=False)


# ==============================================================================
# TASK 2: Nested Validation for Optimal Shift Selection
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Nested Validation for Optimal Shift Selection")
print("======================================================================")

# Selection on Inner Folds (Fold 0 = 2022, Fold 1 = 2023 ONLY)
inner_rows = []
for shift in SHIFTS:
    inner_briers = [calc_raw_brier(fold_y_trues[i], np.clip(fold_raw_preds[i] + shift, 1e-6, 1.0 - 1e-6)) for i in [0, 1]]
    inner_mean_brier = np.mean(inner_briers)
    inner_rows.append({"shift": shift, "inner_mean_brier": inner_mean_brier})

inner_df = pd.DataFrame(inner_rows)
best_inner_shift = inner_df.loc[inner_df["inner_mean_brier"].idxmin()]["shift"]
print(f"Optimal Shift selected on Inner Folds (2022, 2023): {best_inner_shift:+.4f}")

# Evaluate selected shift on Outer Fold 2 (2024 held-out)
f2_baseline_brier = calc_raw_brier(fold_y_trues[2], fold_raw_preds[2])
f2_shifted_brier = calc_raw_brier(fold_y_trues[2], np.clip(fold_raw_preds[2] + best_inner_shift, 1e-6, 1.0 - 1e-6))
f2_improvement = f2_baseline_brier - f2_shifted_brier

f2_baseline_skill, _, _, _ = calc_brier_skill_score(fold_y_trues[2], fold_raw_preds[2])
f2_shifted_skill, _, _, _ = calc_brier_skill_score(fold_y_trues[2], np.clip(fold_raw_preds[2] + best_inner_shift, 1e-6, 1.0 - 1e-6))

print(f"Outer Fold 2 (2024 Held-out) Baseline Brier : {f2_baseline_brier:.6f} (Skill: {f2_baseline_skill:.2f})")
print(f"Outer Fold 2 (2024 Held-out) Shifted Brier  : {f2_shifted_brier:.6f} (Skill: {f2_shifted_skill:.2f})")
print(f"Raw Brier Improvement on 2024 Held-out Data : {f2_improvement:+.6f} (Skill Δ: {f2_shifted_skill - f2_baseline_skill:+.2f} pts)")

t2_df = pd.DataFrame([{
    "best_inner_shift": best_inner_shift,
    "f2_baseline_brier": f2_baseline_brier,
    "f2_shifted_brier": f2_shifted_brier,
    "f2_improvement": f2_improvement,
    "f2_baseline_skill": f2_baseline_skill,
    "f2_shifted_skill": f2_shifted_skill
}])
t2_df.to_csv("~/LG_data/outputs/45_shift_nested_raw.csv", index=False)


# ==============================================================================
# TASK 3: Candidate Comparison (V3-B Standalone vs Shifted vs Combined)
# ==============================================================================
print("\n======================================================================")
print("TASK 3: Final Candidate Comparison (V3-B vs Shifted vs Combined)")
print("======================================================================")

# Model V3-B raw predictions on all 3 folds
v3b_raw_preds = []
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
    v3b_raw_preds.append(preds)

optimal_shift = best_inner_shift

CANDIDATE_COMPARE = {
    "Candidate (a): V3-B Standalone (leaves=45, shift=0.0)": {
        "preds": v3b_raw_preds, "shift": 0.0, "leaves": 45
    },
    f"Candidate (b): Baseline + Shift ({optimal_shift:+.3f})": {
        "preds": fold_raw_preds, "shift": optimal_shift, "leaves": 63
    },
    f"Candidate (c): V3-B + Combined Shift ({optimal_shift:+.3f})": {
        "preds": v3b_raw_preds, "shift": optimal_shift, "leaves": 45
    }
}

comp_rows = []

for cname, cinfo in CANDIDATE_COMPARE.items():
    f_briers = []
    f_skills = []
    f_aucs = []

    for fi in range(len(folds)):
        preds_shifted = np.clip(cinfo["preds"][fi] + cinfo["shift"], 1e-6, 1.0 - 1e-6)
        y_va = fold_y_trues[fi]

        brier = calc_raw_brier(y_va, preds_shifted)
        skill, _, _, _ = calc_brier_skill_score(y_va, preds_shifted)
        auc = roc_auc_score(y_va, preds_shifted)

        f_briers.append(brier)
        f_skills.append(skill)
        f_aucs.append(auc)

    comp_rows.append({
        "candidate": cname,
        "f0_brier": f_briers[0],
        "f1_brier": f_briers[1],
        "f2_brier": f_briers[2],
        "mean_raw_brier": np.mean(f_briers),
        "mean_3fold_skill": np.mean(f_skills),
        "mean_auc": np.mean(f_aucs)
    })

comp_df = pd.DataFrame(comp_rows)
print(comp_df.to_string(index=False))
comp_df.to_csv("~/LG_data/outputs/46_v3_final_candidate_raw.csv", index=False)

print("\n======================================================================")
print("RECENCY SHIFT RE-EVALUATION COMPLETE!")
print("======================================================================")
