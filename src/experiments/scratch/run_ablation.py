"""
Feature Ablation Study — measures the marginal contribution of each feature group.

Compares 4 configurations across 3-fold time-based CV (LightGBM, n_estimators=300):
  (a) asof_* only
  (b) asof_* + game context (categorical + count/score/runner numerics) — no trackman
  (c) (b) + Trackman prior (fold-aware, as_of_season)
  (d) (b) + Trackman prior (full, no as_of_season cutoff)

Also computes LightGBM feature importances (gain) and bootstrap CI for AUC.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import bootstrap as scipy_bootstrap

import lightgbm as lgb
import config
from cv_utils import get_cv_folds
from trackman_features import TrackmanFeatureBuilder

# ── Feature group definitions ─────────────────────────────────────────────────
ASOF_COLS = [
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

CAT_COLS = config.CATEGORICAL_COLS  # top_bottom, game_type, base_state, pitcher_hand, batter_hand, team_ids
DERIVED_CAT = config.DERIVED_CATEGORICAL_COLS  # count_code, platoon_matchup

CONTEXT_NUM_COLS = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
]
DERIVED_NUM_COLS = config.DERIVED_NUMERICAL_COLS  # is_leading, is_tied, score_diff_abs, etc.

TKM_COLS = [c for c in config.TRACKMAN_DERIVED_COLS]  # includes tkm_match

# Feature sets
SETS = {
    "a_asof_only":      ASOF_COLS,
    "b_asof_context":   CAT_COLS + DERIVED_CAT + CONTEXT_NUM_COLS + DERIVED_NUM_COLS + ASOF_COLS,
    "c_with_tkm_aware": CAT_COLS + DERIVED_CAT + CONTEXT_NUM_COLS + DERIVED_NUM_COLS + ASOF_COLS + TKM_COLS,
    "d_with_tkm_full":  CAT_COLS + DERIVED_CAT + CONTEXT_NUM_COLS + DERIVED_NUM_COLS + ASOF_COLS + TKM_COLS,
}


def prepare_features(df_raw, tkm_builder, feature_set_name, feature_cols):
    """Merge trackman if needed, encode categoricals, fill NaN with median."""
    df = df_raw.copy()

    # Add derived cat & num features
    if "balls_before" in df.columns and "strikes_before" in df.columns:
        df["count_code"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    if "pitcher_hand" in df.columns and "batter_hand" in df.columns:
        df["platoon_matchup"] = df["pitcher_hand"].astype(str) + "v" + df["batter_hand"].astype(str)
    if "score_diff_pitcher_team" in df.columns:
        df["is_leading"] = (df["score_diff_pitcher_team"] > 0).astype(int)
        df["is_tied"] = (df["score_diff_pitcher_team"] == 0).astype(int)
        df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()
    if "runner_on_2b" in df.columns and "runner_on_3b" in df.columns:
        df["is_scoring_position"] = ((df["runner_on_2b"]==1)|(df["runner_on_3b"]==1)).astype(int)
    if "asof_pitcher_prev1_game_success_rate" in df.columns:
        df["pitcher_success_trend_1g"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
        df["pitcher_success_trend_3g"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]

    # Add trackman features if needed
    if tkm_builder is not None and any(c.startswith("tkm_") for c in feature_cols):
        df = tkm_builder.transform(df)

    # Encode categoricals as ordinal integers
    all_cat = CAT_COLS + DERIVED_CAT
    for col in all_cat:
        if col in df.columns and col in feature_cols:
            df[col] = df[col].astype("category").cat.codes.replace(-1, 0)

    # Select and fill NaN
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X.values


def lgbm_fit_predict(X_tr, y_tr, X_va):
    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_va)[:, 1]
    return model, preds


def bootstrap_auc_ci(y_true, y_score, n_boot=500, ci=0.95):
    """Return (lower, upper) bootstrap CI for AUC."""
    rng = np.random.default_rng(42)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = np.percentile(boot_aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_aucs, (1 + ci) / 2 * 100)
    return lo, hi


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_wall = time.perf_counter()
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows\n")

    folds = get_cv_folds(df, strategy="time")
    results_all = []
    importance_records = []

    # Pre-build TWO trackman builders per fold: fold-aware & full
    for fi, fold in enumerate(folds):
        print(f"\n{'='*60}\nFold {fi}: train={fold.val_season-1} max, val={fold.val_season}\n{'='*60}")
        df_tr_raw = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va_raw = df.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr_raw[config.TARGET_COL].values
        y_va = df_va_raw[config.TARGET_COL].values

        # Build trackman builders
        tkm_aware = TrackmanFeatureBuilder()
        tkm_aware.fit(as_of_season=fold.fold_max_season)

        tkm_full = TrackmanFeatureBuilder()
        tkm_full.fit(as_of_season=None)

        for set_name, feat_cols in SETS.items():
            t0 = time.perf_counter()
            # Determine which builder to use
            if set_name == "c_with_tkm_aware":
                tkm = tkm_aware
            elif set_name == "d_with_tkm_full":
                tkm = tkm_full
            else:
                tkm = None

            X_tr = prepare_features(df_tr_raw, tkm, set_name, feat_cols)
            X_va = prepare_features(df_va_raw, tkm, set_name, feat_cols)

            model, preds = lgbm_fit_predict(X_tr, y_tr, X_va)
            auc = roc_auc_score(y_va, preds)
            ll = log_loss(y_va, preds)
            lo, hi = bootstrap_auc_ci(y_va, preds, n_boot=300)
            elapsed = time.perf_counter() - t0

            row = {
                "fold": fi, "val_season": fold.val_season, "set": set_name,
                "n_features": len(feat_cols), "auc": auc, "logloss": ll,
                "auc_ci_lo": lo, "auc_ci_hi": hi, "elapsed_s": elapsed
            }
            results_all.append(row)
            above_random = "YES" if lo > 0.5 else "NO (CI overlaps 0.5)"
            print(f"  [{set_name}] n={len(feat_cols)} feats | "
                  f"AUC={auc:.5f} [{lo:.4f},{hi:.4f}] | LogLoss={ll:.5f} | "
                  f">{0.5}? {above_random} | {elapsed:.1f}s")

            # Feature importance (gain) for the full set (c) — last fold only (fold 2)
            if set_name == "c_with_tkm_aware" and fi == 2:
                imp = pd.DataFrame({
                    "feature": feat_cols,
                    "gain": model.booster_.feature_importance(importance_type="gain")
                }).sort_values("gain", ascending=False).reset_index(drop=True)
                imp["fold"] = fi
                importance_records.append(imp)

    results_df = pd.DataFrame(results_all)
    results_df.to_csv("~/LG_data/outputs/14_ablation_raw.csv", index=False)

    if importance_records:
        imp_df = pd.concat(importance_records, ignore_index=True)
        imp_df.to_csv("~/LG_data/outputs/14_importance_raw.csv", index=False)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ABLATION SUMMARY (mean across 3 folds)")
    print("="*70)
    summary = results_df.groupby("set").agg(
        mean_auc=("auc","mean"), std_auc=("auc","std"),
        mean_logloss=("logloss","mean"), std_logloss=("logloss","std"),
        mean_n_feats=("n_features","first")
    ).reset_index()
    print(summary.to_string(index=False))

    t_total = time.perf_counter() - t_wall
    print(f"\nTotal ablation time: {t_total/60:.1f} min")
    print("DONE — saved to outputs/14_ablation_raw.csv and 14_importance_raw.csv")
