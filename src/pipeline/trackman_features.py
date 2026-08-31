"""
trackman_features.py — TrackmanFeatureBuilder for DACON Aimers 9th competition.

Produces situation-level Prior Statistics from trackman_history.csv by grouping on
the composite join key confirmed in 03_data_validation.md (98.43% state-match rate,
99.76% row-match rate after top_bottom mapping).

Design Principles:
  - fit()       : Reads trackman_history.csv, optionally filters to as_of_season,
                  maps top_bottom encoding, aggregates numeric physical stats
                  (rel_speed, spin_rate, etc.) by join key, saves artifact.
                  No label data (control_success) is ever used — pure physical prior.
  - as_of_season: Enables fold-aware temporal CV.
                  - CV mode   : pass as_of_season = max(train_fold seasons)
                    e.g. Fold0 train=[2019-2021] → as_of_season=2021
                    Prevents future trackman data (2022-2024) from leaking into
                    earlier validation folds.
                  - Final mode: as_of_season=None → uses all 2019-2024 trackman.
                    Correct for real 2025 prediction since all 2019-2024 data
                    exists at inference time.
  - transform() : Loads saved artifact and left-merges onto the input DataFrame.
                  Unmatched rows → filled with global means + tkm_match=0 flag.
"""

import os
import joblib
import time
import pandas as pd
import numpy as np
from typing import Optional

import config


# Trackman physical numeric columns to aggregate
_TRACKMAN_NUMERIC_COLS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed"
]


class TrackmanFeatureBuilder:
    """Builds situation-level Trackman Prior Statistics and merges onto train/test.

    Uses only config.TRACKMAN_JOIN_KEYS as the composite join key (no hardcoding).
    Supports fold-aware temporal CV via as_of_season parameter.
    """

    def __init__(self):
        self.join_keys = config.TRACKMAN_JOIN_KEYS
        self.numeric_cols = _TRACKMAN_NUMERIC_COLS
        self.artifacts: dict = {}
        self.is_fitted = False

    def fit(
        self,
        trackman_path: Optional[str] = None,
        as_of_season: Optional[int] = None,
        is_final: bool = None,
    ) -> "TrackmanFeatureBuilder":
        """Load trackman_history.csv, compute situation-level prior stats, save artifact.

        Args:
            trackman_path (str, optional): Path to trackman_history.csv.
                                           Defaults to config.TRACKMAN_PATH.
            as_of_season (int, optional): If provided, only trackman rows with
                                          season <= as_of_season are used. This
                                          enables fold-aware temporal CV.
                                          - CV mode  : set to max(train_fold seasons)
                                          - Final mode: leave as None (uses all data)
            is_final (bool, optional):  Deprecated convenience flag.
                                        If True and as_of_season is None → same as
                                        as_of_season=None (all data).
                                        If False and as_of_season is None → logs a
                                        warning that fold-aware as_of_season was not set.

        Returns:
            self: Fitted TrackmanFeatureBuilder instance.
        """
        if trackman_path is None:
            trackman_path = config.TRACKMAN_PATH

        # Resolve final/cv mode from is_final if as_of_season not explicitly set
        if as_of_season is None and is_final is not None and not is_final:
            print("[TrackmanFeatureBuilder] WARNING: CV mode detected (is_final=False) "
                  "but as_of_season is None. Using full trackman history — "
                  "consider passing as_of_season=max(train_seasons) for strict temporal CV.")

        print(f"[TrackmanFeatureBuilder] Loading {trackman_path} ...")
        t0 = time.perf_counter()
        df_track = pd.read_csv(trackman_path)
        t1 = time.perf_counter()
        print(f"[TrackmanFeatureBuilder] Loaded {len(df_track):,} rows in {t1-t0:.2f}s")

        # ── Temporal filter: restrict to seasons up to as_of_season ──────────
        if as_of_season is not None:
            n_before = len(df_track)
            df_track = df_track[df_track["season"] <= as_of_season].copy()
            n_after = len(df_track)
            pct = n_after / n_before * 100
            print(f"[TrackmanFeatureBuilder] Temporal filter: season <= {as_of_season} → "
                  f"{n_after:,}/{n_before:,} rows retained ({pct:.1f}%)")
        else:
            print(f"[TrackmanFeatureBuilder] No temporal filter (final mode): "
                  f"all {len(df_track):,} rows used.")

        # Map top_bottom encoding: 'Top'/'Bottom' → 'T'/'B' to match train/test
        df_track["top_bottom"] = df_track["top_bottom"].map({"Top": "T", "Bottom": "B"})
        unmapped = df_track["top_bottom"].isnull().sum()
        if unmapped > 0:
            print(f"[TrackmanFeatureBuilder] Warning: {unmapped} rows with unknown "
                  f"top_bottom mapping. Dropping.")
            df_track = df_track.dropna(subset=["top_bottom"])

        # Aggregate numeric physical stats per join key state
        print("[TrackmanFeatureBuilder] Computing situation-level aggregations ...")
        t2 = time.perf_counter()
        agg_df = df_track.groupby(self.join_keys)[self.numeric_cols].agg(["mean", "std"])
        agg_df.columns = [f"tkm_{col}_{stat}" for col, stat in agg_df.columns]
        agg_df = agg_df.reset_index()

        # Pitch count per situation (proxy for coverage reliability)
        cnt_df = (
            df_track.groupby(self.join_keys)["rel_speed"]
            .count()
            .reset_index()
            .rename(columns={"rel_speed": "tkm_n_pitches"})
        )
        agg_df = pd.merge(agg_df, cnt_df, on=self.join_keys, how="left")
        t3 = time.perf_counter()
        print(f"[TrackmanFeatureBuilder] Aggregation done: {agg_df.shape} in {t3-t2:.2f}s")

        # Global means — fallback for unmatched rows
        tkm_feature_cols = [c for c in agg_df.columns if c not in self.join_keys]
        global_means = {col: float(agg_df[col].mean()) for col in tkm_feature_cols}

        self.artifacts = {
            "agg_df": agg_df,
            "tkm_feature_cols": tkm_feature_cols,
            "global_means": global_means,
            "as_of_season": as_of_season,
            "is_final": is_final if is_final is not None else (as_of_season is None),
        }
        self.is_fitted = True
        mode_str = "FINAL (all seasons)" if as_of_season is None else f"CV (season <= {as_of_season})"
        print(f"[TrackmanFeatureBuilder] Fit complete. {len(tkm_feature_cols)} prior features. "
              f"Mode: {mode_str}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Left-merge Trackman prior stats onto input DataFrame.

        Unmatched rows (join key absent in aggregation table) are filled with global
        means and flagged with tkm_match=0.

        Args:
            df (pd.DataFrame): Input DataFrame containing TRACKMAN_JOIN_KEYS columns.

        Returns:
            pd.DataFrame: Input DataFrame with Trackman prior feature columns appended.
                          Original row order and index are preserved.
        """
        if not self.is_fitted:
            raise RuntimeError("TrackmanFeatureBuilder must be fitted before calling transform().")

        agg_df = self.artifacts["agg_df"]
        tkm_feature_cols = self.artifacts["tkm_feature_cols"]
        global_means = self.artifacts["global_means"]

        # Left merge — preserves all rows in df
        df_out = df.copy()
        df_out = pd.merge(
            df_out,
            agg_df,
            on=self.join_keys,
            how="left",
            validate="many_to_one"
        )

        # Match flag: 1 = matched, 0 = unmatched (filled with global mean)
        any_null_mask = df_out[tkm_feature_cols[0]].isnull()
        df_out["tkm_match"] = (~any_null_mask).astype(int)

        n_unmatched = int(any_null_mask.sum())
        n_total = len(df_out)
        match_rate = (n_total - n_unmatched) / n_total * 100
        print(f"[TrackmanFeatureBuilder] Transform: {n_total-n_unmatched:,}/{n_total:,} rows matched "
              f"({match_rate:.2f}%). Unmatched: {n_unmatched:,} rows filled with global means.")

        for col in tkm_feature_cols:
            if df_out[col].isnull().any():
                df_out[col] = df_out[col].fillna(global_means.get(col, 0.0))

        return df_out

    def save(self, path: str) -> None:
        """Save fitted artifacts to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.artifacts, path)
        print(f"[TrackmanFeatureBuilder] Artifacts saved to {path}")

    def load(self, path: str) -> "TrackmanFeatureBuilder":
        """Load fitted artifacts from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        self.artifacts = joblib.load(path)
        self.is_fitted = True
        as_of = self.artifacts.get("as_of_season")
        mode_str = "FINAL" if as_of is None else f"CV season<={as_of}"
        print(f"[TrackmanFeatureBuilder] Artifacts loaded from {path} (mode: {mode_str})")
        return self


if __name__ == "__main__":
    print("=== TrackmanFeatureBuilder Fold-Aware Smoke Test ===\n")

    artifact_dir = config.ARTIFACTS_DIR

    # 1. Final mode (as_of_season=None)
    print("── FINAL MODE ──")
    builder_final = TrackmanFeatureBuilder()
    builder_final.fit(as_of_season=None)
    path_final = os.path.join(artifact_dir, "trackman_final.pkl")
    builder_final.save(path_final)

    # 2. CV mode: Fold 0 (train=2019-2021 → as_of_season=2021)
    print("\n── CV MODE: as_of_season=2021 ──")
    builder_cv = TrackmanFeatureBuilder()
    builder_cv.fit(as_of_season=2021)
    path_cv = os.path.join(artifact_dir, "trackman_cv_2021.pkl")
    builder_cv.save(path_cv)

    # 3. Transform a sample with both and compare
    sample = pd.read_csv(config.TRAIN_PATH, nrows=1000)
    r_final = TrackmanFeatureBuilder().load(path_final).transform(sample)
    r_cv = TrackmanFeatureBuilder().load(path_cv).transform(sample)

    # Compute mean absolute difference on tkm_ columns
    tkm_cols = [c for c in r_final.columns if c.startswith("tkm_") and c != "tkm_match"]
    diffs = (r_final[tkm_cols] - r_cv[tkm_cols]).abs().mean()
    print(f"\nMean per-column diff (final vs cv-2021) on 1000-row sample:")
    print(diffs.round(6).to_string())
    print(f"\nOverall mean diff: {diffs.mean():.6f}")
    print("=== Smoke Test Passed ===")
