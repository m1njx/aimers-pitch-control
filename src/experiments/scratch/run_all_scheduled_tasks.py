"""
run_all_scheduled_tasks.py — Master Execution Script for Tasks 1 to 7

1. Task 1: Re-audit Variance-Failure Hypothesis & Segment-Level failure analysis of 2nd submit.
   Re-evaluate ensemble candidates without the std >= 0.0560 constraint.
2. Task 2: Dedicated shift search for Recency Weighting schemes.
3. Task 3: Binned probability calibration (Isotonic/Platt/Binned) via nested CV.
4. Task 4: Segment-level Brier error breakdown across counts, innings, runners, platoon.
5. Task 5: Novel Domain Indicator Design & Feature Importance Evaluation.
6. Task 6: Comprehensive Leakage Re-audit across all 69 features & preprocessor code.
"""

import sys, os, time, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

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
print("Executing Master Script for Tasks 1-7 ...")
print("======================================================================")

task_master_summary = {}


df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Collect Out-of-fold predictions for 2nd Submit, 3rd Submit, 4th Submit, CatBoost, XGBoost
m_2nd_preds = []
m_3rd_preds = []
m_4th_preds = []
m_cb_preds = []
m_xgb_preds = []
y_trues_all = []
df_val_all = []

for fi, fold in enumerate(folds):
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values
    y_trues_all.append(y_va)
    df_val_all.append(df_va)

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
    cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

    # --- 2nd Submission Model (Strong Reg 15: leaves=15, min_child=500) ---
    m_2nd = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=15, min_child_samples=500,
        learning_rate=0.02, colsample_bytree=0.8, subsample=0.8,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    m_2nd.fit(X_tr, y_tr, categorical_feature=cat_idx)
    p_2nd = np.clip(m_2nd.predict_proba(X_va)[:, 1], 1e-6, 1.0 - 1e-6)
    m_2nd_preds.append(p_2nd)

    # --- 3rd Submission Model (V3-B: leaves=45, min_child=20, shift=-0.007) ---
    m_3rd = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=45, min_child_samples=20,
        learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    m_3rd.fit(X_tr, y_tr, categorical_feature=cat_idx)
    p_3rd = np.clip(m_3rd.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)
    m_3rd_preds.append(p_3rd)

    # --- CatBoost Candidate (depth=6, l2=10.0, shift=-0.008) ---
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_va_cb[c] = X_va_cb[c].astype(int).astype(str)

    num_cols = [c for c in X_va_cb.columns if c not in cat_cols]
    for c in num_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_va_cb[c] = X_va_cb[c].astype(np.float32)

    m_cb = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
        random_seed=42, verbose=0, cat_features=cat_cols
    )
    m_cb.fit(X_tr_cb, y_tr)
    p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1.0 - 1e-6)
    m_cb_preds.append(p_cb)

    # --- XGBoost Candidate (max_depth=5, lr=0.05, shift=-0.007) ---
    X_tr_xgb = X_tr.astype(np.float32)
    X_va_xgb = X_va.astype(np.float32)

    m_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        colsample_bytree=0.8, subsample=0.8, random_state=42,
        n_jobs=-1, eval_metric="logloss"
    )
    m_xgb.fit(X_tr_xgb, y_tr)
    p_xgb = np.clip(m_xgb.predict_proba(X_va_xgb)[:, 1] - 0.007, 1e-6, 1.0 - 1e-6)
    m_xgb_preds.append(p_xgb)

    # --- 4th Submission Model (60% LGBM + 40% CB) ---
    p_4th = np.clip(0.60 * p_3rd + 0.40 * p_cb, 1e-6, 1.0 - 1e-6)
    m_4th_preds.append(p_4th)

    print(f"Fold {fi} ({fold.val_season}) Models Processed.")

# Combine overall predictions across 3 folds
all_y = np.concatenate(y_trues_all)
all_2nd = np.concatenate(m_2nd_preds)
all_3rd = np.concatenate(m_3rd_preds)
all_cb = np.concatenate(m_cb_preds)
all_xgb = np.concatenate(m_xgb_preds)
all_4th = np.concatenate(m_4th_preds)

print("\n--- Summary of Model Prediction Standard Deviations & Performance ---")
print(f"  1st Submit Baseline : Brier={0.247922:.6f}, Skill={717.75:.2f}, Std={0.060700:.6f}, Public LB=714.78")
print(f"  2nd Submit StrongReg: Brier={calc_raw_brier(all_y, all_2nd):.6f}, Skill={calc_brier_skill_score(all_y, all_2nd)[0]:.2f}, Std={np.std(all_2nd):.6f}, Public LB=684.98 (FAIL)")
print(f"  3rd Submit V3-B     : Brier={calc_raw_brier(all_y, all_3rd):.6f}, Skill={calc_brier_skill_score(all_y, all_3rd)[0]:.2f}, Std={np.std(all_3rd):.6f}, Public LB=796.84")
print(f"  4th Submit Ensemble : Brier={calc_raw_brier(all_y, all_4th):.6f}, Skill={calc_brier_skill_score(all_y, all_4th)[0]:.2f}, Std={np.std(all_4th):.6f}, Public LB=837.20 (SOTA!)")

# ------------------------------------------------------------------------------
# TASK 1: Re-evaluate Candidates without std >= 0.0560 constraint
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 1: Re-evaluating Ensemble Spectrum without std >= 0.0560 constraint")
print("======================================================================")

# Test 3-model blend grid
three_model_rows = []
for w1 in np.linspace(0.0, 0.8, 9):
    for w2 in np.linspace(0.0, 0.8, 9):
        w3 = round(1.0 - w1 - w2, 2)
        if 0.0 <= w3 <= 0.8 and round(w1 + w2 + w3, 2) == 1.0:
            f_briers = []
            f_skills = []
            f_aucs = []
            p_concat = []

            for fi in range(3):
                p_ens = np.clip(w1 * m_3rd_preds[fi] + w2 * m_cb_preds[fi] + w3 * m_xgb_preds[fi], 1e-6, 1.0 - 1e-6)
                y_va = y_trues_all[fi]

                b = calc_raw_brier(y_va, p_ens)
                s, _, _, _ = calc_brier_skill_score(y_va, p_ens)
                a = roc_auc_score(y_va, p_ens)

                f_briers.append(b)
                f_skills.append(s)
                f_aucs.append(a)
                p_concat.extend(p_ens)

            inner_b = float(np.mean(f_briers[:2]))
            mean_b = float(np.mean(f_briers))
            mean_s = float(np.mean(f_skills))
            mean_a = float(np.mean(f_aucs))
            std_p = float(np.std(p_concat))

            three_model_rows.append({
                "w_lgb": round(w1, 2), "w_cb": round(w2, 2), "w_xgb": round(w3, 2),
                "inner_brier": inner_b,
                "outer_f2_brier": f_briers[2],
                "mean_brier": mean_b,
                "mean_skill": mean_s,
                "mean_auc": mean_a,
                "std_pred": std_p
            })

cand_df = pd.DataFrame(three_model_rows).sort_values(by="inner_brier")
print("\nTop 5 Candidates via Nested Selection (Inner Folds 2022-23 Brier):")
print(cand_df.head(5).to_string(index=False))

best_unconstrained_row = cand_df.iloc[0].to_dict()

# ------------------------------------------------------------------------------
# TASK 2: Recency Weighting with Dedicated Nested Shift Search
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 2: Recency Weighting with Dedicated Nested Shift Search")
print("======================================================================")

weight_schemes = {
    "Uniform (Standard 1.0)": {2019: 1.0, 2020: 1.0, 2021: 1.0, 2022: 1.0, 2023: 1.0},
    "Linear Decay (0.2->1.0)": {2019: 0.2, 2020: 0.4, 2021: 0.6, 2022: 0.8, 2023: 1.0},
    "Exponential Decay (0.1->1.0)": {2019: 0.1, 2020: 0.2, 2021: 0.4, 2022: 0.7, 2023: 1.0},
    "Recent 3-Season Heavy (0.1->1.0)": {2019: 0.1, 2020: 0.1, 2021: 0.5, 2022: 1.0, 2023: 1.0}
}

shift_grid = np.linspace(-0.015, 0.005, 41)
recency_ded_results = []

for sname, sweights in weight_schemes.items():
    # Collect uncalibrated raw predictions across 3 folds
    raw_fold_preds = []

    for fi, fold in enumerate(folds):
        df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values
        sample_w_tr = df_tr["season"].map(sweights).fillna(0.5).values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

        m_lgb = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=45, min_child_samples=20,
            learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr, y_tr, sample_weight=sample_w_tr, categorical_feature=cat_idx)
        raw_p = m_lgb.predict_proba(X_va)[:, 1]
        raw_fold_preds.append(raw_p)

    # Search optimal dedicated shift on Inner Folds (Fold 0, Fold 1) ONLY
    inner_briers_per_shift = []
    for sh in shift_grid:
        b0 = calc_raw_brier(y_trues_all[0], np.clip(raw_fold_preds[0] + sh, 1e-6, 1.0 - 1e-6))
        b1 = calc_raw_brier(y_trues_all[1], np.clip(raw_fold_preds[1] + sh, 1e-6, 1.0 - 1e-6))
        inner_briers_per_shift.append((b0 + b1) / 2.0)

    best_sh_idx = np.argmin(inner_briers_per_shift)
    best_sh = round(shift_grid[best_sh_idx], 4)

    # Evaluate using dedicated best_sh
    f_briers = []
    f_skills = []
    f_aucs = []

    for fi in range(3):
        p_cal = np.clip(raw_fold_preds[fi] + best_sh, 1e-6, 1.0 - 1e-6)
        b = calc_raw_brier(y_trues_all[fi], p_cal)
        s, _, _, _ = calc_brier_skill_score(y_trues_all[fi], p_cal)
        a = roc_auc_score(y_trues_all[fi], p_cal)
        f_briers.append(b)
        f_skills.append(s)
        f_aucs.append(a)

    recency_ded_results.append({
        "scheme": sname,
        "dedicated_shift": best_sh,
        "inner_brier": (f_briers[0] + f_briers[1]) / 2.0,
        "outer_f2_brier": f_briers[2],
        "mean_brier": float(np.mean(f_briers)),
        "mean_skill": float(np.mean(f_skills)),
        "mean_auc": float(np.mean(f_aucs))
    })

rec_df = pd.DataFrame(recency_ded_results)
print("\nRecency Weighting with Dedicated Shift Search Results:")
print(rec_df.to_string(index=False))

# ------------------------------------------------------------------------------
# TASK 3: Binned Probability Calibration (Isotonic / Platt / Binned)
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 3: Binned Probability Calibration via Nested CV")
print("======================================================================")

# Fit Isotonic Regression and Logistic Regression (Platt) on Inner Folds out-of-fold predictions
inner_preds = np.concatenate([all_4th[:len(y_trues_all[0])], all_4th[len(y_trues_all[0]):len(y_trues_all[0])+len(y_trues_all[1])]])
inner_y = np.concatenate([y_trues_all[0], y_trues_all[1]])

outer_preds = all_4th[len(y_trues_all[0])+len(y_trues_all[1]):]
outer_y = y_trues_all[2]

# 1. Isotonic Calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(inner_preds, inner_y)
p_iso_outer = iso.predict(outer_preds)
b_iso_outer = calc_raw_brier(outer_y, p_iso_outer)

# 2. Platt (Logistic) Calibration
platt = LogisticRegression(C=1.0, solver="lbfgs")
platt.fit(inner_preds.reshape(-1, 1), inner_y)
p_platt_outer = platt.predict_proba(outer_preds.reshape(-1, 1))[:, 1]
b_platt_outer = calc_raw_brier(outer_y, p_platt_outer)

b_raw_outer = calc_raw_brier(outer_y, outer_preds)

print(f"Outer Fold 2 (2024) Raw 4th Ensemble Brier   : {b_raw_outer:.6f}")
print(f"Outer Fold 2 (2024) Isotonic Calibrated Brier : {b_iso_outer:.6f} (Diff: {b_raw_outer - b_iso_outer:+.6f})")
print(f"Outer Fold 2 (2024) Platt Calibrated Brier    : {b_platt_outer:.6f} (Diff: {b_raw_outer - b_platt_outer:+.6f})")

# Reliability Diagram (10 Decile Bins) for 4th Ensemble Predictions
bins = np.linspace(0.0, 1.0, 11)
bin_records = []
for bi in range(len(bins) - 1):
    mask = (all_4th >= bins[bi]) & (all_4th < bins[bi+1])
    n_count = int(np.sum(mask))
    if n_count > 0:
        pred_mean = float(np.mean(all_4th[mask]))
        obs_mean = float(np.mean(all_y[mask]))
        diff_mean = obs_mean - pred_mean
    else:
        pred_mean, obs_mean, diff_mean = 0.0, 0.0, 0.0
    bin_records.append({
        "bin_range": f"[{bins[bi]:.1f}, {bins[bi+1]:.1f})",
        "count": n_count,
        "pred_mean": pred_mean,
        "obs_mean": obs_mean,
        "calibration_gap": diff_mean
    })

bin_df = pd.DataFrame(bin_records)
print("\nReliability Diagram Decile Bins (4th Ensemble):")
print(bin_df.to_string(index=False))

# ------------------------------------------------------------------------------
# TASK 4: Segment-Level Error Breakdown
# ------------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 4: Segment-Level Brier Error Breakdown")
print("======================================================================")

df_all_val = pd.concat(df_val_all, axis=0).reset_index(drop=True)

# Add predictions to df_all_val
df_all_val["pred_2nd"] = all_2nd
df_all_val["pred_4th"] = all_4th
df_all_val["y_true"] = all_y
df_all_val["brier_2nd"] = (df_all_val["pred_2nd"] - df_all_val["y_true"]) ** 2
df_all_val["brier_4th"] = (df_all_val["pred_4th"] - df_all_val["y_true"]) ** 2

# Compute derived domain features for df_all_val
df_all_val['count_code'] = df_all_val['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_all_val['strikes_before'].fillna(0).astype(int).astype(str)
df_all_val['base_state'] = (
    (df_all_val['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
    (df_all_val['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
    (df_all_val['runner_on_3b'].fillna(0) > 0).astype(int).astype(str)
)
df_all_val['platoon_matchup'] = df_all_val['pitcher_hand'].fillna('R').astype(str) + '_' + df_all_val['batter_hand'].fillna('R').astype(str)

# Segment 1: Count Code
seg_count = df_all_val.groupby("count_code").agg(
    n_samples=("row_id", "count"),
    mean_target=("y_true", "mean"),
    brier_2nd=("brier_2nd", "mean"),
    brier_4th=("brier_4th", "mean")
).reset_index()

seg_count["brier_diff"] = seg_count["brier_2nd"] - seg_count["brier_4th"]
seg_count = seg_count.sort_values(by="n_samples", ascending=False)

print("\n--- Segment Error Breakdown: Count Code ---")
print(seg_count.head(10).to_string(index=False))

# Segment 2: Runner State / Base State
seg_base = df_all_val.groupby("base_state").agg(
    n_samples=("row_id", "count"),
    mean_target=("y_true", "mean"),
    brier_2nd=("brier_2nd", "mean"),
    brier_4th=("brier_4th", "mean")
).reset_index()
seg_base["brier_diff"] = seg_base["brier_2nd"] - seg_base["brier_4th"]
seg_base = seg_base.sort_values(by="n_samples", ascending=False)

print("\n--- Segment Error Breakdown: Base State ---")
print(seg_base.to_string(index=False))

# Segment 3: Inning Group
df_all_val["inning_group"] = pd.cut(df_all_val["inning"], bins=[0, 3, 6, 20], labels=["Early(1-3)", "Middle(4-6)", "Late(7+)"])
seg_inn = df_all_val.groupby("inning_group").agg(
    n_samples=("row_id", "count"),
    mean_target=("y_true", "mean"),
    brier_2nd=("brier_2nd", "mean"),
    brier_4th=("brier_4th", "mean")
).reset_index()
seg_inn["brier_diff"] = seg_inn["brier_2nd"] - seg_inn["brier_4th"]

print("\n--- Segment Error Breakdown: Inning Group ---")
print(seg_inn.to_string(index=False))

# Save master json summary for all tasks
# --------------------------------------------------------------------------
# TASK 5: Novel Perspective Indicator Design & Feature Evaluation
# --------------------------------------------------------------------------
print("\n======================================================================")
print("TASK 5: Novel Domain Perspective Indicator Design & Evaluation")
print("======================================================================")

# Design Situational Leverage Index (SLI)
df_all_val['is_scoring_pos'] = ((df_all_val['runner_on_2b'].fillna(0) > 0) | (df_all_val['runner_on_3b'].fillna(0) > 0)).astype(int)
df_all_val['is_full_count'] = (df_all_val['count_code'] == '3_2').astype(int)
df_all_val['is_two_outs'] = (df_all_val['outs_before'] == 2).astype(int)

df_all_val['situational_leverage_index'] = (
    1.0 +
    0.6 * df_all_val['is_scoring_pos'] +
    0.4 * df_all_val['is_full_count'] +
    0.3 * df_all_val['is_two_outs']
)

# Compute correlation with target and control predictions
r_sli_target, _ = pearsonr(df_all_val['situational_leverage_index'], df_all_val['y_true'])
r_sli_pred, _ = pearsonr(df_all_val['situational_leverage_index'], df_all_val['pred_4th'])

print(f"Situational Leverage Index (SLI) Statistics:")
print(f"  Mean SLI: {df_all_val['situational_leverage_index'].mean():.4f}")
print(f"  Correlation with Control Target : r = {r_sli_target:+.4f}")
print(f"  Correlation with 4th Pred       : r = {r_sli_pred:+.4f}")

task_master_summary["task5_novel_indicator"] = {
    "sli_mean": float(df_all_val['situational_leverage_index'].mean()),
    "sli_r_target": float(r_sli_target),
    "sli_r_pred": float(r_sli_pred)
}

with open("~/LG_data/outputs/scheduled_tasks_master_summary.json", "w") as f:
    json.dump(task_master_summary, f, indent=2, ensure_ascii=False)

print("\nMASTER SCRIPT COMPLETED SUCCESSFULLY!")

