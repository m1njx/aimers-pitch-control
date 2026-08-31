"""
run_nested_and_interaction_experiments.py — Deep Investigation Script:
  Task 1: Investigates the super-additive effect of variant_all_in (season + game_type).
          Examines season x game_type crosstab, row counts, target means, target variance,
          and LightGBM tree split features/thresholds.
  Task 2: Performs Nested Temporal Validation:
          - Inner Tuning Folds: Fold 0 (val=2022) & Fold 1 (val=2023) ONLY. (Fold 2 val=2024 is strictly held out).
          - Evaluates Hyperparameter candidates (Baseline, Shallow 31, Strong Reg 15, etc.) on Inner Folds.
          - Evaluates selected candidates ONCE on Held-out Fold 2 (val=2024).
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor


def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r


print("Loading train.csv ...")
df_all = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded {len(df_all):,} rows.\n")

folds = get_cv_folds(df_all, strategy="time")

ALL_COLS_LIST = (
    config.CATEGORICAL_COLS
    + config.DERIVED_CATEGORICAL_COLS
    + [config.TRACKMAN_MATCH_FLAG_COL]
    + config.RAW_NUMERICAL_COLS
    + config.DERIVED_NUMERICAL_COLS
    + [c for c in config.TRACKMAN_DERIVED_COLS if c != config.TRACKMAN_MATCH_FLAG_COL]
)

# ==============================================================================
# TASK 1: Investigation of variant_all_in (season x game_type) Super-Additive Effect
# ==============================================================================
print("======================================================================")
print("TASK 1: Investigating variant_all_in (season x game_type) Interaction")
print("======================================================================")

# 1.1 Crosstab of season x game_type: Count, Target Mean, Target Variance
ct_cnt = pd.crosstab(df_all['season'], df_all['game_type'])
ct_target_mean = df_all.groupby(['season', 'game_type'])['control_success'].mean().unstack()
ct_target_var = df_all.groupby(['season', 'game_type'])['control_success'].var().unstack()

print("Crosstab Counts (season x game_type):")
print(ct_cnt)
print("\nTarget Means (control_success rate):")
print(ct_target_mean.round(4))
print("\nTarget Variance:")
print(ct_target_var.round(4))

# 1.2 Train LightGBM model on variant_all_in (71 cols) and inspect tree splits
print("\nTraining LightGBM on variant_all_in and inspecting splits on (season, game_type)...")
df_tr0 = df_all.iloc[folds[0].train_idx].reset_index(drop=True)
df_va0 = df_all.iloc[folds[0].val_idx].reset_index(drop=True)
y_tr0 = df_tr0[config.TARGET_COL].values
y_va0 = df_va0[config.TARGET_COL].values

prep0 = PitchPreprocessor()
prep0.fit(df_tr0, as_of_season=folds[0].fold_max_season, is_final=False, feature_whitelist_override=ALL_COLS_LIST)

X_tr0 = prep0.transform(df_tr0)
X_va0 = prep0.transform(df_va0)

cat_cols0 = [c for c in ALL_COLS_LIST if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

m_all = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    random_state=42, verbosity=-1, n_jobs=-1
)
m_all.fit(X_tr0, y_tr0, categorical_feature=[ALL_COLS_LIST.index(c) for c in cat_cols0 if c in ALL_COLS_LIST])
preds0 = m_all.predict_proba(X_va0)[:, 1]

score0, brier0, base0, r0 = calc_brier_skill_score(y_va0, preds0)
print(f"Fold 0 variant_all_in: Skill Score = {score0:.2f} | Brier = {brier0:.6f} | BaseBrier = {base0:.6f} | r = {r0:.4f}")

# Tree dump & split inspection for season & game_type
tree_dump = m_all.booster_.dump_model()['tree_info']
season_gt_splits = []

for tree in tree_dump:
    def parse_node(node, depth=0):
        if 'split_feature' in node:
            fname = m_all.booster_.feature_name()[node['split_feature']]
            if fname in ['season', 'game_type']:
                season_gt_splits.append({
                    'depth': depth,
                    'feature': fname,
                    'threshold': node.get('threshold'),
                    'gain': node.get('split_gain')
                })
            if 'left_child' in node:
                parse_node(node['left_child'], depth+1)
            if 'right_child' in node:
                parse_node(node['right_child'], depth+1)
    parse_node(tree['tree_structure'])

splits_df = pd.DataFrame(season_gt_splits)
print(f"Found {len(splits_df)} splits involving season or game_type.")
print("Top 10 highest gain splits on season or game_type:")
if not splits_df.empty:
    print(splits_df.sort_values('gain', ascending=False).head(10).to_string(index=False))


# ==============================================================================
# TASK 2: Nested Temporal Validation (Inner Folds 0&1 for tuning, Held-out Fold 2 for final test)
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Nested Temporal Validation (Inner Folds 0 & 1 vs Held-Out Fold 2)")
print("======================================================================")

# Hyperparameter Candidates to evaluate
HP_CANDIDATES = {
    "Baseline (leaves=63, lr=0.05, min_child=20)": {"leaves": 63, "lr": 0.05, "min_child": 20, "colsample": 0.8, "subsample": 0.8, "n_est": 300},
    "Shallow 31 (leaves=31, lr=0.03, min_child=100)": {"leaves": 31, "lr": 0.03, "min_child": 100, "colsample": 0.7, "subsample": 0.7, "n_est": 400},
    "Shallow 15 (leaves=15, lr=0.03, min_child=200)": {"leaves": 15, "lr": 0.03, "min_child": 200, "colsample": 0.7, "subsample": 0.7, "n_est": 400},
    "Strong Reg 15 (leaves=15, lr=0.02, min_child=500)": {"leaves": 15, "lr": 0.02, "min_child": 500, "colsample": 0.6, "subsample": 0.6, "n_est": 400},
    "Ultra Reg 7 (leaves=7, lr=0.02, min_child=1000)": {"leaves": 7, "lr": 0.02, "min_child": 1000, "colsample": 0.5, "subsample": 0.5, "n_est": 400},
}

# Features to use: Current clean 69 features (season X, game_type X, team_ids O)
model_features_69 = [c for c in ALL_COLS_LIST if c not in ["season", "game_type"]]

nested_results = []

for cand_name, hp in HP_CANDIDATES.items():
    print(f"\nEvaluating Candidate: {cand_name} ...")
    inner_fold_scores = []
    inner_fold_unclipped = []
    inner_fold_aucs = []
    inner_fold_briers = []

    # Run ONLY Inner Folds (Fold 0: val=2022, Fold 1: val=2023) for candidate selection
    for fi in [0, 1]:
        fold = folds[fi]
        df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False, feature_whitelist_override=model_features_69)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in model_features_69 if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=hp["n_est"],
            num_leaves=hp["leaves"],
            learning_rate=hp["lr"],
            min_child_samples=hp["min_child"],
            colsample_bytree=hp["colsample"],
            subsample=hp["subsample"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=[model_features_69.index(c) for c in cat_cols if c in model_features_69])
        preds = model.predict_proba(X_va)[:, 1]

        r = float(np.mean(y_va))
        brier = float(np.mean((preds - y_va) ** 2))
        base_brier = float(r * (1.0 - r))
        unclipped_skill = 100000.0 * (1.0 - (brier / base_brier))
        clipped_skill = max(0.0, unclipped_skill)
        auc = roc_auc_score(y_va, preds)

        inner_fold_scores.append(clipped_skill)
        inner_fold_unclipped.append(unclipped_skill)
        inner_fold_aucs.append(auc)
        inner_fold_briers.append(brier)

    inner_mean_clipped = np.mean(inner_fold_scores)
    inner_mean_unclipped = np.mean(inner_fold_unclipped)
    inner_mean_auc = np.mean(inner_fold_aucs)

    # NOW: Evaluate ONCE on Held-out Fold 2 (val=2024)
    fold2 = folds[2]
    df_tr2 = df_all.iloc[fold2.train_idx].reset_index(drop=True)
    df_va2 = df_all.iloc[fold2.val_idx].reset_index(drop=True)
    y_tr2 = df_tr2[config.TARGET_COL].values
    y_va2 = df_va2[config.TARGET_COL].values

    prep2 = PitchPreprocessor()
    prep2.fit(df_tr2, as_of_season=fold2.fold_max_season, is_final=False, feature_whitelist_override=model_features_69)

    X_tr2 = prep2.transform(df_tr2)
    X_va2 = prep2.transform(df_va2)

    cat_cols2 = [c for c in model_features_69 if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

    model2 = lgb.LGBMClassifier(
        n_estimators=hp["n_est"],
        num_leaves=hp["leaves"],
        learning_rate=hp["lr"],
        min_child_samples=hp["min_child"],
        colsample_bytree=hp["colsample"],
        subsample=hp["subsample"],
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model2.fit(X_tr2, y_tr2, categorical_feature=[model_features_69.index(c) for c in cat_cols2 if c in model_features_69])
    preds2 = model2.predict_proba(X_va2)[:, 1]

    r2 = float(np.mean(y_va2))
    brier2 = float(np.mean((preds2 - y_va2) ** 2))
    base_brier2 = float(r2 * (1.0 - r2))
    unclipped_skill2 = 100000.0 * (1.0 - (brier2 / base_brier2))
    clipped_skill2 = max(0.0, unclipped_skill2)
    auc2 = roc_auc_score(y_va2, preds2)

    print(f"  INNER FOLDS (0 & 1): ClippedSkill={inner_mean_clipped:.2f} | UnclippedSkill={inner_mean_unclipped:.2f} | AUC={inner_mean_auc:.6f}")
    print(f"  HELD-OUT FOLD 2 (val=2024): SkillScore={clipped_skill2:.2f} | UnclippedSkill={unclipped_skill2:.2f} | AUC={auc2:.6f}")

    nested_results.append({
        "candidate": cand_name,
        "inner_mean_clipped_skill": inner_mean_clipped,
        "inner_mean_unclipped_skill": inner_mean_unclipped,
        "inner_mean_auc": inner_mean_auc,
        "heldout_fold2_skill": clipped_skill2,
        "heldout_fold2_unclipped_skill": unclipped_skill2,
        "heldout_fold2_auc": auc2,
        "heldout_fold2_brier": brier2,
        "f0_unclipped": inner_fold_unclipped[0],
        "f1_unclipped": inner_fold_unclipped[1],
    })

nested_df = pd.DataFrame(nested_results)
nested_df.to_csv("~/LG_data/outputs/35_nested_validation_raw.csv", index=False)

print("\n======================================================================")
print("NESTED TEMPORAL VALIDATION SUMMARY")
print("======================================================================")
print(nested_df[["candidate", "inner_mean_unclipped_skill", "heldout_fold2_unclipped_skill", "heldout_fold2_auc"]].to_string(index=False))

print("\nDONE — All deep investigation experiments completed successfully!")
