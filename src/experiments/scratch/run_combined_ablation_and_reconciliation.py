"""
run_combined_ablation_and_reconciliation.py — Runs combined feature exclusion experiment
and reconciliation analysis for Fold 1 (2023) low AUC.

4 Combinations (3-Fold Time CV, LightGBM n_estimators=300):
  a) Both included (season & game_type included)
  b) Only season excluded
  c) Only game_type excluded
  d) Both excluded (final candidate)
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
from trackman_features import TrackmanFeatureBuilder
from preprocessing import PitchPreprocessor

# Base model features from config (71 features)
FULL_MODEL_COLS = list(config.MODEL_FEATURE_COLS)

COMBINATIONS = {
    "a_both_included": FULL_MODEL_COLS,
    "b_no_season":     [c for c in FULL_MODEL_COLS if c != "season"],
    "c_no_game_type":  [c for c in FULL_MODEL_COLS if c != "game_type"],
    "d_both_excluded": [c for c in FULL_MODEL_COLS if c not in ["season", "game_type"]],
}


def fit_eval_model(df_tr, df_va, feature_cols, fold_max_season):
    # Fit preprocessor on train
    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values

    # Restrict to specified feature_cols
    cols_to_use = [c for c in feature_cols if c in X_tr.columns]
    X_tr_sub = X_tr[cols_to_use]
    X_va_sub = X_va[cols_to_use]

    cat_cols = [c for c in cols_to_use if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_tr_sub, y_tr, categorical_feature=[cols_to_use.index(c) for c in cat_cols if c in cols_to_use])
    preds = model.predict_proba(X_va_sub)[:, 1]

    auc = roc_auc_score(y_va, preds)
    ll = log_loss(y_va, preds)
    return model, auc, ll, cols_to_use


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows.")

    folds = get_cv_folds(df, strategy="time")
    results = []
    importance_d = []

    for fi, fold in enumerate(folds):
        print(f"\n=======================================================")
        print(f"Fold {fi}: train max={fold.fold_max_season}, val={fold.val_season}")
        print(f"=======================================================")
        df_tr = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df.iloc[fold.val_idx].reset_index(drop=True)

        for combo_name, feat_cols in COMBINATIONS.items():
            t0 = time.perf_counter()
            model, auc, ll, used_cols = fit_eval_model(df_tr, df_va, feat_cols, fold.fold_max_season)
            elapsed = time.perf_counter() - t0

            results.append({
                'fold': fi,
                'val_season': fold.val_season,
                'combo': combo_name,
                'n_features': len(used_cols),
                'auc': auc,
                'logloss': ll,
                'elapsed_s': elapsed
            })
            print(f"  [{combo_name:<16s}] n={len(used_cols):2d} | AUC={auc:.6f} | LogLoss={ll:.6f} | {elapsed:.1f}s")

            # Collect feature importance for combo (d) across all folds
            if combo_name == "d_both_excluded":
                imp = pd.DataFrame({
                    'feature': used_cols,
                    'gain': model.booster_.feature_importance(importance_type="gain"),
                    'fold': fi
                })
                importance_d.append(imp)

    rdf = pd.DataFrame(results)
    rdf.to_csv("~/LG_data/outputs/20_combined_ablation_raw.csv", index=False)

    imp_df = pd.concat(importance_d, ignore_index=True)
    mean_imp = imp_df.groupby('feature')['gain'].mean().sort_values(ascending=False).reset_index()
    mean_imp.to_csv("~/LG_data/outputs/20_importance_combo_d.csv", index=False)

    print("\n=======================================================")
    print("COMBINED EXCLUSION ABLATION SUMMARY (3 Folds)")
    print("=======================================================")
    summary = rdf.groupby('combo').agg(
        mean_auc=('auc', 'mean'),
        std_auc=('auc', 'std'),
        mean_logloss=('logloss', 'mean'),
        std_logloss=('logloss', 'std'),
        n_feats=('n_features', 'first')
    ).reset_index()

    # Calculate AUC delta vs baseline (combo a)
    base_auc = summary.loc[summary['combo'] == 'a_both_included', 'mean_auc'].values[0]
    summary['delta_auc_vs_a'] = summary['mean_auc'] - base_auc
    print(summary.to_string(index=False))

    print("\nTop 15 Features for Combo (d) [Both Excluded] (Mean Gain across 3 Folds):")
    print(mean_imp.head(15).to_string(index=False))

    t_end = time.perf_counter()
    print(f"\nAll experiments finished in {t_end - t_start:.2f}s")
