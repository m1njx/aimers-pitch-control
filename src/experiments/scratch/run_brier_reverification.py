"""
run_brier_reverification.py — Comprehensive Brier Skill Score Verification Script

Executes:
  Task 1: Measures Brier Skill Score on current 69-feature LightGBM baseline across 3 time folds.
  Task 2: Diagnoses calibration (over/under-confidence) & tests Platt scaling / Isotonic / CalibratedClassifierCV.
  Task 3: Re-verifies feature exclusion decisions (season, game_type, team_ids) using Brier Skill Score.
  Task 4: Tests regularization strength (shallow trees num_leaves=15, 31 vs 63, min_child_samples=100, 200, 500)
          and compares against RandomForest baseline (max_depth=10, min_samples_leaf=200).
"""

import sys
import os
import time
import warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder


def calc_brier_skill_score(y_true, y_prob):
    """Calculates official DACON Brier Skill Score.

    Formula:
      Brier = mean((p_i - y_i)^2)
      r = mean(y_i)
      baseline_brier = r * (1 - r)
      Score = max(0, 100000 * (1 - Brier / baseline_brier))
    """
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading train.csv ...")
df_all = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded {len(df_all):,} rows. Seasons: {sorted(df_all['season'].unique())}\n")

folds = get_cv_folds(df_all, strategy="time")


# ==============================================================================
# TASK 1: Current 69-feature Baseline Brier Skill Score
# ==============================================================================
print("======================================================================")
print("TASK 1: Brier Skill Score Measurement on Current 69-Feature Baseline")
print("======================================================================")

t1_results = []
oof_preds_t1 = np.zeros(len(df_all))
oof_mask_t1 = np.zeros(len(df_all), dtype=bool)

for fi, fold in enumerate(folds):
    df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
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
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
    preds = model.predict_proba(X_va)[:, 1]

    score, brier, base_brier, r = calc_brier_skill_score(y_va, preds)
    auc = roc_auc_score(y_va, preds)
    ll = log_loss(y_va, preds)

    oof_preds_t1[fold.val_idx] = preds
    oof_mask_t1[fold.val_idx] = True

    t1_results.append({
        "fold": fi,
        "val_season": fold.val_season,
        "r": r,
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_skill_score": score,
        "auc": auc,
        "logloss": ll
    })
    print(f"Fold {fi} (val={fold.val_season}): r={r:.4f} | Brier={brier:.6f} | BaseBrier={base_brier:.6f} | BrierSkillScore={score:.2f} | AUC={auc:.6f}")

t1_df = pd.DataFrame(t1_results)
mean_skill_t1 = t1_df["brier_skill_score"].mean()
mean_auc_t1 = t1_df["auc"].mean()
print(f"\nMean Brier Skill Score: {mean_skill_t1:.2f}")
print(f"Mean AUC: {mean_auc_t1:.6f}")
t1_df.to_csv("~/LG_data/outputs/27_brier_baseline_raw.csv", index=False)


# ==============================================================================
# TASK 2: Calibration Diagnosis & Post-Hoc Calibration Testing
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Probability Calibration Diagnosis & Calibration Testing")
print("======================================================================")

calib_results = []
for fi, fold in enumerate(folds):
    df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])

    # Raw predictions on train & val
    tr_preds = model.predict_proba(X_tr)[:, 1]
    va_preds = model.predict_proba(X_va)[:, 1]

    raw_score, raw_brier, _, r = calc_brier_skill_score(y_va, va_preds)

    # Method 1: Platt Scaling (Sigmoid Logistic Regression fit on train predictions)
    plat = LogisticRegression(C=1.0, max_iter=1000)
    plat.fit(tr_preds.reshape(-1, 1), y_tr)
    plat_preds = plat.predict_proba(va_preds.reshape(-1, 1))[:, 1]
    plat_score, plat_brier, _, _ = calc_brier_skill_score(y_va, plat_preds)

    # Method 2: Isotonic Regression fit on train predictions
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(tr_preds, y_tr)
    iso_preds = iso.transform(va_preds)
    iso_score, iso_brier, _, _ = calc_brier_skill_score(y_va, iso_preds)

    # Reliability curve stats
    prob_true, prob_pred = calibration_curve(y_va, va_preds, n_bins=10)
    max_calib_err = np.max(np.abs(prob_true - prob_pred))
    mean_calib_err = np.mean(np.abs(prob_true - prob_pred))

    calib_results.append({
        "fold": fi,
        "val_season": fold.val_season,
        "raw_skill": raw_score,
        "platt_skill": plat_score,
        "iso_skill": iso_score,
        "raw_brier": raw_brier,
        "platt_brier": plat_brier,
        "iso_brier": iso_brier,
        "max_calib_err": max_calib_err,
        "mean_calib_err": mean_calib_err,
        "pred_min": va_preds.min(),
        "pred_max": va_preds.max(),
        "pred_mean": va_preds.mean(),
        "y_true_mean": r
    })
    print(f"Fold {fi} (val={fold.val_season}): Raw Skill={raw_score:.2f} | Platt={plat_score:.2f} | Iso={iso_score:.2f} | Pred Mean={va_preds.mean():.4f} (True={r:.4f})")

calib_df = pd.DataFrame(calib_results)
print(f"\nMean Raw Skill Score:   {calib_df['raw_skill'].mean():.2f}")
print(f"Mean Platt Skill Score: {calib_df['platt_skill'].mean():.2f}")
print(f"Mean Iso Skill Score:   {calib_df['iso_skill'].mean():.2f}")
calib_df.to_csv("~/LG_data/outputs/28_calibration_raw.csv", index=False)


# ==============================================================================
# TASK 3: Re-verify Feature Exclusion Decisions using Brier Skill Score
# ==============================================================================
print("\n======================================================================")
print("TASK 3: Feature Inclusion/Exclusion Re-verification using Brier Skill Score")
print("======================================================================")

# Candidate Feature Sets
# 1. Base 69 cols (season X, game_type X, team_id O)
# 2. Add season back (+season)
# 3. Add game_type back (+game_type)
# 4. Remove team_id (-team_id)
# 5. Add season & game_type back (all in)

ALL_COLS_LIST = list(config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL] + config.RAW_NUMERICAL_COLS + config.DERIVED_NUMERICAL_COLS + [c for c in config.TRACKMAN_DERIVED_COLS if c != config.TRACKMAN_MATCH_FLAG_COL])

FEAT_VARIANTS = {
    "d_base_69 (season X, gt X, team O)": [c for c in ALL_COLS_LIST if c not in ["season", "game_type"]],
    "variant_add_season (+season)": [c for c in ALL_COLS_LIST if c != "game_type"],
    "variant_add_gt (+game_type)": [c for c in ALL_COLS_LIST if c != "season"],
    "variant_sub_team (-team_ids)": [c for c in ALL_COLS_LIST if c not in ["season", "game_type", "pitcher_team_id", "batter_team_id"]],
    "variant_all_in (season O, gt O, team O)": ALL_COLS_LIST,
}

t3_results = []

for var_name, feat_cols in FEAT_VARIANTS.items():
    fold_scores = []
    fold_aucs = []
    fold_briers = []

    for fi, fold in enumerate(folds):
        df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        use_cols = [c for c in feat_cols if c in X_tr.columns]
        cat_cols = [c for c in use_cols if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbosity=-1, n_jobs=-1
        )
        model.fit(X_tr[use_cols], y_tr, categorical_feature=[use_cols.index(c) for c in cat_cols if c in use_cols])
        preds = model.predict_proba(X_va[use_cols])[:, 1]

        score, brier, _, r = calc_brier_skill_score(y_va, preds)
        auc = roc_auc_score(y_va, preds)

        fold_scores.append(score)
        fold_aucs.append(auc)
        fold_briers.append(brier)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    mean_auc = np.mean(fold_aucs)
    mean_brier = np.mean(fold_briers)

    t3_results.append({
        "variant": var_name,
        "n_features": len(feat_cols),
        "mean_brier_skill_score": mean_score,
        "std_brier_skill_score": std_score,
        "mean_auc": mean_auc,
        "mean_brier": mean_brier
    })
    print(f"Variant: {var_name:<40s} | SkillScore: {mean_score:.2f} (std={std_score:.2f}) | AUC: {mean_auc:.6f} | Brier: {mean_brier:.6f}")

t3_df = pd.DataFrame(t3_results)
t3_df.to_csv("~/LG_data/outputs/29_brier_feature_reverify_raw.csv", index=False)


# ==============================================================================
# TASK 4: Baseline Comparison & Hyperparameter Regularization Tuning
# ==============================================================================
print("\n======================================================================")
print("TASK 4: Baseline Notebook Comparison & Hyperparameter Regularization Tuning")
print("======================================================================")

# 4.1 Compare Raw Features in test.csv vs MODEL_FEATURE_COLS
test_sample = pd.read_csv(config.TEST_PATH, nrows=0)
raw_test_cols = [c for c in test_sample.columns if c != config.ID_COL]

our_model_cols = config.MODEL_FEATURE_COLS
missing_raw_in_ours = [c for c in raw_test_cols if c not in our_model_cols and c not in config.EXCLUDED_FEATURE_COLS]
print(f"Raw test.csv features count: {len(raw_test_cols)}")
print(f"Excluded raw features in config: {config.EXCLUDED_FEATURE_COLS}")
print(f"Missing raw features in MODEL_FEATURE_COLS: {missing_raw_in_ours} (Expected: None)")

# 4.2 Test official baseline RandomForest (max_depth=10, min_samples_leaf=200) on 3 time folds
print("\n4.2 Testing Official Baseline RandomForest (max_depth=10, min_samples_leaf=200) on 3 Folds ...")
rf_cat_cols = ["top_bottom", "game_type", "base_state"]
rf_results = []

for fi, fold in enumerate(folds):
    df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)

    rf_features = [c for c in raw_test_cols if c in df_tr.columns]
    rf_num_cols = [c for c in rf_features if c not in rf_cat_cols]

    X_tr_rf = df_tr[rf_features]
    y_tr_rf = df_tr[config.TARGET_COL].values
    X_va_rf = df_va[rf_features]
    y_va_rf = df_va[config.TARGET_COL].values

    rf_prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), rf_cat_cols),
        ("num", SimpleImputer(strategy="median"), rf_num_cols),
    ])

    rf_model = Pipeline([
        ("pre", rf_prep),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=42))
    ])

    rf_model.fit(X_tr_rf, y_tr_rf)
    rf_preds = rf_model.predict_proba(X_va_rf)[:, 1]

    score, brier, _, r = calc_brier_skill_score(y_va_rf, rf_preds)
    auc = roc_auc_score(y_va_rf, rf_preds)
    rf_results.append({"fold": fi, "val_season": fold.val_season, "brier_skill_score": score, "auc": auc, "brier": brier})
    print(f"  RF Baseline Fold {fi} (val={fold.val_season}): BrierSkillScore={score:.2f} | AUC={auc:.6f} | Brier={brier:.6f}")

rf_df = pd.DataFrame(rf_results)
print(f"Official RF Baseline Mean Skill Score: {rf_df['brier_skill_score'].mean():.2f} | Mean AUC: {rf_df['auc'].mean():.6f}")

# 4.3 Test LightGBM Regularization Strength Tuning
print("\n4.3 Testing LightGBM Regularization Hyperparameter Grids ...")

LGB_HP_GRIDS = [
    {"name": "Current Baseline (leaves=63, lr=0.05, min_child=20)", "num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20, "colsample": 0.8, "subsample": 0.8},
    {"name": "Shallow 31 (leaves=31, lr=0.03, min_child=100)", "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 100, "colsample": 0.7, "subsample": 0.7},
    {"name": "Shallow 15 (leaves=15, lr=0.03, min_child=200)", "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 200, "colsample": 0.7, "subsample": 0.7},
    {"name": "Strong Reg 15 (leaves=15, lr=0.02, min_child=500)", "num_leaves": 15, "learning_rate": 0.02, "min_child_samples": 500, "colsample": 0.6, "subsample": 0.6},
    {"name": "Ultra Reg 7 (leaves=7, lr=0.02, min_child=1000)", "num_leaves": 7, "learning_rate": 0.02, "min_child_samples": 1000, "colsample": 0.5, "subsample": 0.5},
]

hp_results = []

for hp in LGB_HP_GRIDS:
    grid_name = hp["name"]
    fold_scores = []
    fold_aucs = []
    fold_briers = []

    for fi, fold in enumerate(folds):
        df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=hp["num_leaves"],
            learning_rate=hp["learning_rate"],
            min_child_samples=hp["min_child_samples"],
            colsample_bytree=hp["colsample"],
            subsample=hp["subsample"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
        preds = model.predict_proba(X_va)[:, 1]

        score, brier, _, _ = calc_brier_skill_score(y_va, preds)
        auc = roc_auc_score(y_va, preds)

        fold_scores.append(score)
        fold_aucs.append(auc)
        fold_briers.append(brier)

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    mean_auc = np.mean(fold_aucs)
    mean_brier = np.mean(fold_briers)

    hp_results.append({
        "grid_name": grid_name,
        "mean_brier_skill_score": mean_score,
        "std_brier_skill_score": std_score,
        "mean_auc": mean_auc,
        "mean_brier": mean_brier
    })
    print(f"Grid: {grid_name:<48s} | SkillScore: {mean_score:.2f} (std={std_score:.2f}) | AUC: {mean_auc:.6f} | Brier: {mean_brier:.6f}")

hp_df = pd.DataFrame(hp_results)
hp_df.to_csv("~/LG_data/outputs/30_regularization_tuning_raw.csv", index=False)

print("\n======================================================================")
print("ALL BRIER RE-VERIFICATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("======================================================================")
