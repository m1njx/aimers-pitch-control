"""
run_team_id_drift_check.py — Verifies whether pitcher_team_id and batter_team_id
act as target drift sources or year proxies.

1. Calculates team x season control_success crosstab & rank volatility for pitcher/batter teams.
2. Runs 3-fold Time CV comparing:
   - Baseline combo (d) [season & game_type excluded] (69 features)
   - Combo (d) WITHOUT pitcher_team_id & batter_team_id (67 features)
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

# ── 1. Team x Season Crosstab Analysis ────────────────────────────────────────
def analyze_team_drift(df):
    print("=== 1. Pitcher Team ID x Season Target Rate Crosstab ===")
    p_team_ct = df.groupby(['pitcher_team_id', 'season'])['control_success'].agg(['count', 'mean']).unstack()
    p_team_means = df.groupby(['pitcher_team_id', 'season'])['control_success'].mean().unstack()
    print("Pitcher Team Success Rates per Season:")
    print(p_team_means.round(4))

    print("\nPitcher Team Season-to-Season Statistics:")
    p_team_stats = pd.DataFrame({
        'overall_mean': df.groupby('pitcher_team_id')['control_success'].mean(),
        'season_std': p_team_means.std(axis=1),
        'min_season_rate': p_team_means.min(axis=1),
        'max_season_rate': p_team_means.max(axis=1),
        'range': p_team_means.max(axis=1) - p_team_means.min(axis=1)
    })
    print(p_team_stats.round(4))

    print("\n=== 2. Batter Team ID x Season Target Rate Crosstab ===")
    b_team_means = df.groupby(['batter_team_id', 'season'])['control_success'].mean().unstack()
    print("Batter Team Success Rates per Season:")
    print(b_team_means.round(4))

    b_team_stats = pd.DataFrame({
        'overall_mean': df.groupby('batter_team_id')['control_success'].mean(),
        'season_std': b_team_means.std(axis=1),
        'min_season_rate': b_team_means.min(axis=1),
        'max_season_rate': b_team_means.max(axis=1),
        'range': b_team_means.max(axis=1) - b_team_means.min(axis=1)
    })
    print(b_team_stats.round(4))

    # Rank volatility across seasons
    p_ranks = p_team_means.rank(ascending=False)
    print("\nPitcher Team Rank per Season (1=Highest Success Rate):")
    print(p_ranks)
    print("\nPitcher Team Rank Std across Seasons:")
    print(p_ranks.std(axis=1).round(2))

    return p_team_means, p_team_stats, b_team_means, b_team_stats, p_ranks


# ── 2. 3-Fold Time CV Experiment ───────────────────────────────────────────────
def run_cv_experiment(df):
    print("\n=== 3. 3-Fold Time CV Experiment (WITH vs WITHOUT Team IDs) ===")
    folds = get_cv_folds(df, strategy="time")

    # Current combination (d) feature list (69 features)
    base_cols = list(config.MODEL_FEATURE_COLS)
    no_team_cols = [c for c in base_cols if c not in ["pitcher_team_id", "batter_team_id"]]

    results = []

    for fi, fold in enumerate(folds):
        df_tr = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df.iloc[fold.val_idx].reset_index(drop=True)

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr_full = prep.transform(df_tr)
        X_va_full = prep.transform(df_va)

        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        # A) WITH team_ids (69 features)
        cols_A = [c for c in base_cols if c in X_tr_full.columns]
        cat_A = [c for c in cols_A if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        mA = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
        mA.fit(X_tr_full[cols_A], y_tr, categorical_feature=[cols_A.index(c) for c in cat_A if c in cols_A])
        auc_A = roc_auc_score(y_va, mA.predict_proba(X_va_full[cols_A])[:, 1])
        ll_A = log_loss(y_va, mA.predict_proba(X_va_full[cols_A])[:, 1])

        # B) WITHOUT team_ids (67 features)
        cols_B = [c for c in no_team_cols if c in X_tr_full.columns]
        cat_B = [c for c in cols_B if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        mB = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
        mB.fit(X_tr_full[cols_B], y_tr, categorical_feature=[cols_B.index(c) for c in cat_B if c in cols_B])
        auc_B = roc_auc_score(y_va, mB.predict_proba(X_va_full[cols_B])[:, 1])
        ll_B = log_loss(y_va, mB.predict_proba(X_va_full[cols_B])[:, 1])

        results.append({
            'fold': fi,
            'val_season': fold.val_season,
            'auc_with_teams': auc_A,
            'auc_no_teams': auc_B,
            'auc_diff': auc_B - auc_A,
            'll_with_teams': ll_A,
            'll_no_teams': ll_B,
            'll_diff': ll_B - ll_A
        })
        print(f"  Fold {fi} (val={fold.val_season}): WITH teams AUC={auc_A:.6f} | WITHOUT teams AUC={auc_B:.6f} | ΔAUC={auc_B - auc_A:+.6f}")

    rdf = pd.DataFrame(results)
    print("\nCV Results Summary:")
    print(rdf.to_string(index=False))
    print(f"\nMean AUC WITH team_ids:    {rdf['auc_with_teams'].mean():.6f}")
    print(f"Mean AUC WITHOUT team_ids: {rdf['auc_no_teams'].mean():.6f}")
    print(f"Mean ΔAUC (no_teams - with_teams): {rdf['auc_diff'].mean():+.6f}")

    return rdf


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows.")

    p_team_means, p_team_stats, b_team_means, b_team_stats, p_ranks = analyze_team_drift(df)
    cv_rdf = run_cv_experiment(df)

    # Save raw outputs to CSV for report generation
    p_team_means.to_csv("~/LG_data/outputs/22_pitcher_team_means.csv")
    b_team_means.to_csv("~/LG_data/outputs/22_batter_team_means.csv")
    cv_rdf.to_csv("~/LG_data/outputs/22_team_cv_raw.csv", index=False)

    t_end = time.perf_counter()
    print(f"\nCompleted in {t_end - t_start:.2f}s")
