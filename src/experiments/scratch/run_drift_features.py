"""
Seasonal drift feature experiment for induced_vert_break and horz_break.

For each train split in the 3-fold CV, computes:
  - situation-level means from ONLY the most recent 2 seasons of trackman data
    (as a "recent drift-corrected prior")
  - Adds these as extra features alongside the existing 7-key priors
  - Compares AUC vs the baseline (c_with_tkm_aware) from the ablation study

If improvement < 0.001 AUC on average → "complexity not worth adding" conclusion.
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

JOIN_KEYS = config.TRACKMAN_JOIN_KEYS  # 7-key (no season)
DRIFT_COLS = ["induced_vert_break", "horz_break"]
DRIFT_WINDOW = 2   # last N seasons


def build_recent_prior(trackman_path, as_of_season, window=2):
    """Build situation-level mean for drift columns using only recent N seasons."""
    df_track = pd.read_csv(trackman_path)
    df_track["top_bottom"] = df_track["top_bottom"].map({"Top": "T", "Bottom": "B"})

    # Filter to as_of_season window: [as_of_season - window + 1 .. as_of_season]
    min_season = as_of_season - window + 1
    df_recent = df_track[
        (df_track["season"] >= min_season) & (df_track["season"] <= as_of_season)
    ].copy()
    n_total = len(df_track[df_track["season"] <= as_of_season])
    n_recent = len(df_recent)
    print(f"    Recent prior: seasons [{min_season}-{as_of_season}], "
          f"{n_recent:,}/{n_total:,} rows used ({n_recent/n_total*100:.1f}%)")

    agg = df_recent.groupby(JOIN_KEYS)[DRIFT_COLS].mean()
    agg.columns = [f"tkm_{c}_recent{window}s_mean" for c in agg.columns]
    agg = agg.reset_index()
    global_means = {col: float(agg[col].mean()) for col in agg.columns if col not in JOIN_KEYS}
    return agg, global_means


def add_recent_prior(df_raw, recent_agg, recent_global_means, recent_feature_names):
    """Left-merge recent drift features and fill unmatched with global mean."""
    df = df_raw.copy()
    df = pd.merge(df, recent_agg, on=JOIN_KEYS, how="left", validate="many_to_one")
    for col in recent_feature_names:
        if df[col].isnull().any():
            df[col] = df[col].fillna(recent_global_means.get(col, 0.0))
    return df


def prepare_all_features(df_raw, tkm_builder, recent_agg, recent_global_means,
                          recent_cols, use_recent=False):
    """Full preprocessing for the drift experiment."""
    df = df_raw.copy()

    # Derived features
    if "balls_before" in df.columns:
        df["count_code"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    if "pitcher_hand" in df.columns:
        df["platoon_matchup"] = df["pitcher_hand"].astype(str) + "v" + df["batter_hand"].astype(str)
    if "score_diff_pitcher_team" in df.columns:
        df["is_leading"] = (df["score_diff_pitcher_team"] > 0).astype(int)
        df["is_tied"] = (df["score_diff_pitcher_team"] == 0).astype(int)
        df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()
    if "runner_on_2b" in df.columns:
        df["is_scoring_position"] = ((df["runner_on_2b"]==1)|(df["runner_on_3b"]==1)).astype(int)
    if "asof_pitcher_prev1_game_success_rate" in df.columns:
        df["pitcher_success_trend_1g"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
        df["pitcher_success_trend_3g"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]

    # Trackman prior (fold-aware)
    df = tkm_builder.transform(df)

    # Recent drift prior (optional)
    if use_recent and recent_agg is not None:
        df = add_recent_prior(df, recent_agg, recent_global_means, recent_cols)

    # Encode cat cols
    for col in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.replace(-1, 0)

    # Determine feature columns
    base_cols = (config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS +
                 config.RAW_NUMERICAL_COLS + config.DERIVED_NUMERICAL_COLS +
                 config.TRACKMAN_DERIVED_COLS)
    feat_cols = list(base_cols)
    if use_recent:
        feat_cols += recent_cols

    # Fill and return
    missing = [c for c in feat_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    return X.values, feat_cols


def lgbm_auc(X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, preds), log_loss(y_va, preds)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_wall = time.perf_counter()
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows\n")

    folds = get_cv_folds(df, strategy="time")
    results = []

    for fi, fold in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"Fold {fi}: train max={fold.fold_max_season}, val={fold.val_season}")
        print(f"{'='*55}")

        df_tr = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        # Build fold-aware trackman (historical prior, no drift)
        print("  Building fold-aware trackman prior ...")
        tkm = TrackmanFeatureBuilder()
        tkm.fit(as_of_season=fold.fold_max_season)

        # Build recent drift prior (last 2 seasons)
        print(f"  Building recent drift prior (last {DRIFT_WINDOW} seasons) ...")
        recent_agg, recent_global = build_recent_prior(
            config.TRACKMAN_PATH, fold.fold_max_season, window=DRIFT_WINDOW
        )
        recent_cols = [c for c in recent_agg.columns if c not in JOIN_KEYS]

        # Baseline: fold-aware trackman only (= config c in ablation)
        t0 = time.perf_counter()
        X_tr_base, _ = prepare_all_features(df_tr, tkm, None, None, [], use_recent=False)
        X_va_base, _ = prepare_all_features(df_va, tkm, None, None, [], use_recent=False)
        auc_base, ll_base = lgbm_auc(X_tr_base, y_tr, X_va_base, y_va)
        t_base = time.perf_counter() - t0
        print(f"  BASELINE (fold-aware, no drift): AUC={auc_base:.5f}, "
              f"LogLoss={ll_base:.5f} ({t_base:.1f}s)")

        # With recent drift features
        t0 = time.perf_counter()
        X_tr_drift, feat_list = prepare_all_features(
            df_tr, tkm, recent_agg, recent_global, recent_cols, use_recent=True
        )
        X_va_drift, _ = prepare_all_features(
            df_va, tkm, recent_agg, recent_global, recent_cols, use_recent=True
        )
        auc_drift, ll_drift = lgbm_auc(X_tr_drift, y_tr, X_va_drift, y_va)
        t_drift = time.perf_counter() - t0
        delta = auc_drift - auc_base
        print(f"  WITH DRIFT FEATURES (recent {DRIFT_WINDOW}s): AUC={auc_drift:.5f}, "
              f"LogLoss={ll_drift:.5f} ({t_drift:.1f}s) | ΔAUC={delta:+.5f}")

        results.append({
            "fold": fi, "val_season": fold.val_season,
            "auc_baseline": auc_base, "ll_baseline": ll_base,
            "auc_drift": auc_drift, "ll_drift": ll_drift,
            "delta_auc": delta, "n_drift_feats": len(recent_cols)
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv("~/LG_data/outputs/16_drift_raw.csv", index=False)

    print("\n" + "="*55)
    print("SEASONAL DRIFT FEATURE EXPERIMENT SUMMARY")
    print("="*55)
    print(rdf[["fold","val_season","auc_baseline","auc_drift","delta_auc"]].to_string(index=False))
    print(f"\nMean ΔAUC: {rdf['delta_auc'].mean():+.5f}")
    print(f"Max |ΔAUC|: {rdf['delta_auc'].abs().max():.5f}")

    verdict = "ADOPT" if rdf["delta_auc"].mean() > 0.001 else "SKIP (improvement < 0.001)"
    print(f"\nVerdict: {verdict}")
    print(f"Saved to outputs/16_drift_raw.csv")

    t_total = time.perf_counter() - t_wall
    print(f"Total time: {t_total/60:.1f} min\nDONE")
