"""
Benchmark: Compares temporal leakage (full trackman) vs fold-aware (as_of_season)
Trackman integration across 3 time-based CV folds using a lightweight LightGBM model.

Run: python3 LG_data/scratch/benchmark_trackman_leakage.py
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/LG_data'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor

# Try to import lightgbm; fallback to sklearn GBM if unavailable
try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM")
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    USE_LGB = False
    print("LightGBM not available, using HistGradientBoostingClassifier")


def fit_model(X_tr, y_tr):
    if USE_LGB:
        cat_features = [c for c in X_tr.columns
                        if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                        or c == config.TRACKMAN_MATCH_FLAG_COL]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr.values, y_tr,
                  categorical_feature=[X_tr.columns.get_loc(c)
                                       for c in cat_features if c in X_tr.columns])
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=20, random_state=42
        )
        model.fit(X_tr, y_tr)
    return model


def predict_proba(model, X_va):
    if USE_LGB:
        return model.predict_proba(X_va.values)[:, 1]
    else:
        return model.predict_proba(X_va)[:, 1]


def run_cv_strategy(df, folds, use_fold_aware: bool):
    """Run 3-fold CV with either full or fold-aware Trackman.
    
    Args:
        df: Full training dataframe
        folds: List[FoldInfo] from get_cv_folds
        use_fold_aware: If True, pass fold.fold_max_season to PitchPreprocessor.fit()
    """
    strategy_name = "FOLD-AWARE" if use_fold_aware else "FULL-TRACKMAN"
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    results = []
    
    for fold_i, fold in enumerate(folds):
        t_fold_start = time.perf_counter()
        print(f"\n── Fold {fold_i} ──")
        print(f"  {fold.notes[:80]}...")
        
        df_tr = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values
        
        # Fit preprocessor
        as_of = fold.fold_max_season if use_fold_aware else None
        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=as_of, is_final=(not use_fold_aware))
        
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)
        
        # Fit model
        t_model_start = time.perf_counter()
        model = fit_model(X_tr, y_tr)
        t_model_end = time.perf_counter()
        
        # Score
        preds = predict_proba(model, X_va)
        auc = roc_auc_score(y_va, preds)
        ll = log_loss(y_va, preds)
        
        t_fold_end = time.perf_counter()
        
        result = {
            'fold': fold_i,
            'val_season': fold.val_season,
            'as_of_season': as_of,
            'n_train': len(df_tr),
            'n_val': len(df_va),
            'auc': auc,
            'logloss': ll,
            'model_fit_s': t_model_end - t_model_start,
            'total_s': t_fold_end - t_fold_start,
        }
        results.append(result)
        print(f"  AUC={auc:.6f}  LogLoss={ll:.6f}  "
              f"(model_fit={result['model_fit_s']:.1f}s, total={result['total_s']:.1f}s)")
    
    rdf = pd.DataFrame(results)
    print(f"\n[{strategy_name}] Mean AUC={rdf['auc'].mean():.6f}  "
          f"Mean LogLoss={rdf['logloss'].mean():.6f}")
    return rdf


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_wall = time.perf_counter()
    
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows. Seasons: {sorted(df['season'].unique())}\n")
    
    print("Building CV folds ...")
    folds = get_cv_folds(df, strategy="time")
    print(f"Total folds: {len(folds)}\n")
    
    # Strategy A: Full trackman (original, temporal leakage in CV)
    results_full = run_cv_strategy(df, folds, use_fold_aware=False)
    
    # Strategy B: Fold-aware trackman (as_of_season=fold_max_season)
    results_aware = run_cv_strategy(df, folds, use_fold_aware=True)
    
    # Merge for comparison
    comp = results_full[['fold','val_season','auc','logloss']].merge(
        results_aware[['fold','auc','logloss']],
        on='fold', suffixes=('_full', '_aware')
    )
    comp['auc_diff'] = comp['auc_aware'] - comp['auc_full']
    comp['logloss_diff'] = comp['logloss_aware'] - comp['logloss_full']
    
    print("\n" + "="*60)
    print("COMPARISON TABLE: FOLD-AWARE vs FULL-TRACKMAN")
    print("="*60)
    print(comp[['fold','val_season','auc_full','auc_aware','auc_diff',
                'logloss_full','logloss_aware','logloss_diff']].to_string(index=False))
    print(f"\nMean AUC diff (aware - full): {comp['auc_diff'].mean():+.6f}")
    print(f"Mean LogLoss diff (aware - full): {comp['logloss_diff'].mean():+.6f}")
    
    # Trackman physical feature stability across seasons (full vs restricted)
    print("\n=== Computing Trackman Feature Distribution Stability ===")
    import importlib
    from trackman_features import TrackmanFeatureBuilder
    
    df_track = pd.read_csv(config.TRACKMAN_PATH)
    df_track['top_bottom'] = df_track['top_bottom'].map({'Top':'T','Bottom':'B'})
    
    phys_cols = ['rel_speed','spin_rate','induced_vert_break','horz_break',
                 'extension','rel_height','rel_side','zone_speed']
    
    season_stats = []
    for yr in sorted(df_track['season'].unique()):
        sub = df_track[df_track['season'] == yr]
        row = {'season': yr, 'n': len(sub)}
        for col in phys_cols:
            row[f'{col}_mean'] = sub[col].mean()
        season_stats.append(row)
    
    stats_df = pd.DataFrame(season_stats)
    print("\nSeason-level mean of physical features:")
    print(stats_df[['season','n'] + [f'{c}_mean' for c in phys_cols]].to_string(index=False))
    
    # Compute std of season-means (= inter-season variability)
    mean_cols = [f'{c}_mean' for c in phys_cols]
    inter_season_std = stats_df[mean_cols].std()
    overall_mean = stats_df[mean_cols].mean()
    cv_pct = (inter_season_std / overall_mean.abs() * 100)
    print("\nInter-season coefficient of variation (CV%) for each physical feature:")
    for col in mean_cols:
        print(f"  {col.replace('_mean','')}: {cv_pct[col]:.2f}%")
    
    print(f"\nMean CV% across all physical features: {cv_pct.mean():.2f}%")
    
    # Save results to file
    results_path = '~/LG_data/outputs/13_benchmark_raw.csv'
    comp.to_csv(results_path, index=False)
    stats_df.to_csv('~/LG_data/outputs/13_trackman_season_stats.csv', index=False)
    print(f"\nResults saved to {results_path}")
    
    t_total = time.perf_counter() - t_wall
    print(f"\nTotal benchmark time: {t_total/60:.1f} min")
    print("DONE")
