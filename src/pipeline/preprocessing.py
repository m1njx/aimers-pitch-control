"""
preprocessing.py — PitchPreprocessor implementation for DACON Aimers 9th competition.

Updated to integrate TrackmanFeatureBuilder for Trackman Prior Statistics.

Design:
  - PitchPreprocessor.fit(df, is_final=True/False):
      is_final=True  → Full train data, for final submission artifact.
      is_final=False → Called once per CV fold with fold's train split.
      TrackmanFeatureBuilder is always fit from the full trackman_history.csv
      regardless of is_final, since trackman stats are historical priors
      independent of train fold labels (no CV leakage).
  - PitchPreprocessor.transform(df): Uses saved artifacts only. Fast and offline-safe.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

import config
from trackman_features import TrackmanFeatureBuilder


class PitchPreprocessor:
    """End-to-end Preprocessor for Pitcher Control Success Probability Prediction.

    Integrates:
    1. Domain feature engineering (derived numeric/categorical features)
    2. Trackman Prior Statistics (via TrackmanFeatureBuilder)
    3. Categorical ordinal encoding with whitelist enforcement
    4. Numeric median imputation with whitelist enforcement

    All column decisions are driven by config.py. No hardcoding allowed.
    """

    def __init__(self):
        self.raw_feature_whitelist: List[str] = config.ALL_FEATURE_COLS
        self.model_feature_whitelist: List[str] = config.MODEL_FEATURE_COLS
        self.cat_cols_whitelist: List[str] = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
        self.id_cols: List[str] = [config.ID_COL, config.TARGET_COL] + config.ID_ONLY_COLS
        self.trackman_builder: Optional[TrackmanFeatureBuilder] = None
        self.artifacts: Dict[str, Any] = {}
        self.is_fitted: bool = False

    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive domain-specific features without target leakage or future row lookahead.

        Note on Group/Time-Order Shift Direction:
        - All rolling/expanding features provided by host (asof_*) use PAST pitches only
          (verified in 08_asof_leakage_check.md: shift direction is t-1 -> t, no leakage found).
        - Derived features below use ONLY the current pitch's count/score/runner state,
          no inter-row operations, no lookahead into future pitches.
        """
        df = df.copy()

        # 1. Count state interaction (e.g. balls=3, strikes=2 -> "3-2")
        if "balls_before" in df.columns and "strikes_before" in df.columns:
            df["count_code"] = (
                df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
            )

        # 2. Score & Runner binary interactions (no row-group dependency)
        if "score_diff_pitcher_team" in df.columns:
            df["is_leading"] = (df["score_diff_pitcher_team"] > 0).astype(int)
            df["is_tied"] = (df["score_diff_pitcher_team"] == 0).astype(int)
            df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()

        if ("runner_on_2b" in df.columns) and ("runner_on_3b" in df.columns):
            df["is_scoring_position"] = (
                (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)
            ).astype(int)

        # 3. Recent trend diffs (both cols are past-only, no leakage)
        if (
            "asof_pitcher_prev1_game_success_rate" in df.columns
            and "asof_pitcher_success_rate" in df.columns
        ):
            df["pitcher_success_trend_1g"] = (
                df["asof_pitcher_prev1_game_success_rate"]
                - df["asof_pitcher_success_rate"]
            )

        if (
            "asof_pitcher_prev3_game_success_rate" in df.columns
            and "asof_pitcher_success_rate" in df.columns
        ):
            df["pitcher_success_trend_3g"] = (
                df["asof_pitcher_prev3_game_success_rate"]
                - df["asof_pitcher_success_rate"]
            )

        # 4. Hand matchup (platoon advantage)
        if "pitcher_hand" in df.columns and "batter_hand" in df.columns:
            df["platoon_matchup"] = (
                df["pitcher_hand"].astype(str) + "v" + df["batter_hand"].astype(str)
            )

        return df

    def fit(
        self,
        df_train: pd.DataFrame,
        as_of_season: Optional[int] = None,
        is_final: bool = True,
        trackman_path: Optional[str] = None,
        feature_whitelist_override: Optional[List[str]] = None
    ) -> "PitchPreprocessor":
        """Fit all preprocessing components.

        Args:
            df_train (pd.DataFrame): Raw training data (full train or CV fold's train split).
            as_of_season (int, optional): If set, TrackmanFeatureBuilder uses only
                                          trackman data up to this season (inclusive).
            is_final (bool): If True, trackman_builder uses all data without temporal filtering.
            trackman_path (str, optional): Path to trackman_history.csv.
            feature_whitelist_override (list, optional): Overrides config.MODEL_FEATURE_COLS
                                                          for feature ablation experiments.

        Returns:
            self: Fitted PitchPreprocessor.
        """
        if feature_whitelist_override is not None:
            self.model_feature_whitelist = list(feature_whitelist_override)
        else:
            self.model_feature_whitelist = list(config.MODEL_FEATURE_COLS)

        t_start = time.perf_counter()
        mode_str = "FINAL" if (as_of_season is None and is_final) else f"CV (season<={as_of_season})"
        print(f"\n[PitchPreprocessor] Fitting in {mode_str} mode on {len(df_train):,} rows ...")

        # Step 1: Fit and apply TrackmanFeatureBuilder
        print("[PitchPreprocessor] Step 1/3: Fitting TrackmanFeatureBuilder ...")
        self.trackman_builder = TrackmanFeatureBuilder()
        self.trackman_builder.fit(
            trackman_path=trackman_path,
            as_of_season=as_of_season,
            is_final=is_final
        )

        tkm_artifact_path = os.path.join(config.ARTIFACTS_DIR, "trackman_artifacts.pkl")
        self.trackman_builder.save(tkm_artifact_path)

        # Apply trackman transform to train for learning imputer stats
        df_with_tkm = self.trackman_builder.transform(df_train)

        # Step 2: Extract domain features
        print("[PitchPreprocessor] Step 2/3: Extracting domain features ...")
        df_processed = self._extract_domain_features(df_with_tkm)

        # Step 3: Build whitelist-enforced encoding & imputation artifacts
        print("[PitchPreprocessor] Step 3/3: Building encoding and imputation artifacts ...")
        print("\n--- Column Whitelist Scan ---")
        for col in df_processed.columns:
            if col in self.id_cols:
                print(f"  [EXCLUDED] '{col}' — Reserved ID / Target / Grouping column")
            elif col not in self.model_feature_whitelist:
                print(f"  [EXCLUDED] '{col}' — Not in config.MODEL_FEATURE_COLS whitelist")

        fitted_cat_cols: List[str] = []
        fitted_num_cols: List[str] = []
        cat_maps: Dict[str, Dict[str, int]] = {}
        num_medians: Dict[str, float] = {}

        for col in self.model_feature_whitelist:
            if col not in df_processed.columns:
                print(f"  [WARNING] Whitelisted feature '{col}' missing from DataFrame. Using fallback.")
                continue

            if col in self.cat_cols_whitelist or col == config.TRACKMAN_MATCH_FLAG_COL:
                # Categorical encoding: 0 reserved for unknown/missing
                unique_vals = sorted(df_processed[col].dropna().astype(str).unique().tolist())
                cat_maps[col] = {v: i + 1 for i, v in enumerate(unique_vals)}
                fitted_cat_cols.append(col)
            else:
                # Numeric: safe conversion check
                numeric_series = pd.to_numeric(df_processed[col], errors="coerce")
                n_unconvertible = int(numeric_series.isnull().sum() - df_processed[col].isnull().sum())
                if n_unconvertible > 0 and df_processed[col].dtype == object:
                    print(f"  [WARNING] '{col}' has {n_unconvertible} non-numeric values. Excluding.")
                    continue
                med_val = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0
                num_medians[col] = med_val
                fitted_num_cols.append(col)

        fitted_model_features = fitted_cat_cols + fitted_num_cols

        self.artifacts = {
            "cat_maps": cat_maps,
            "num_medians": num_medians,
            "fitted_cat_cols": fitted_cat_cols,
            "fitted_num_cols": fitted_num_cols,
            "model_feature_cols": fitted_model_features,
            "is_final": is_final,
            "as_of_season": as_of_season,
            "tkm_artifact_path": tkm_artifact_path
        }
        self.is_fitted = True

        t_end = time.perf_counter()
        print(f"\n[PitchPreprocessor] Fit complete in {t_end - t_start:.2f}s")
        print(f"  Categorical features : {len(fitted_cat_cols)}")
        print(f"  Numerical features   : {len(fitted_num_cols)}")
        print(f"  Total model features : {len(fitted_model_features)}\n")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw input into model-ready feature matrix.

        Uses ONLY saved artifacts — no recomputation. Offline and runtime safe.

        Args:
            df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Feature matrix matching config.MODEL_FEATURE_COLS exactly.
        """
        if not self.is_fitted:
            raise RuntimeError("PitchPreprocessor must be fitted or loaded before transform().")

        # Step 1: Trackman prior merge
        df_with_tkm = self.trackman_builder.transform(df)

        # Step 2: Domain features
        df_processed = self._extract_domain_features(df_with_tkm)

        cat_maps = self.artifacts["cat_maps"]
        num_medians = self.artifacts["num_medians"]
        fitted_cat_cols = self.artifacts["fitted_cat_cols"]
        fitted_num_cols = self.artifacts["fitted_num_cols"]
        model_feature_cols = self.artifacts["model_feature_cols"]

        X_out = pd.DataFrame(index=df_processed.index)

        # Categorical transforms
        for col in fitted_cat_cols:
            val_map = cat_maps.get(col, {})
            if col in df_processed.columns:
                X_out[col] = df_processed[col].astype(str).map(val_map).fillna(0).astype(int)
            else:
                X_out[col] = 0

        # Numerical transforms with median imputation
        for col in fitted_num_cols:
            med_val = num_medians.get(col, 0.0)
            if col in df_processed.columns:
                X_out[col] = pd.to_numeric(df_processed[col], errors="coerce").fillna(med_val).astype(float)
            else:
                X_out[col] = med_val

        return X_out[model_feature_cols]

    def save(self, path: str) -> None:
        """Save PitchPreprocessor artifacts to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_dict = dict(self.artifacts)
        # Save trackman builder separately (already saved during fit); store path reference only
        joblib.dump(save_dict, path)
        print(f"[PitchPreprocessor] Artifacts saved to {path}")

    def load(self, path: str, trackman_artifact_path: Optional[str] = None) -> "PitchPreprocessor":
        """Load PitchPreprocessor artifacts and reconstruct TrackmanFeatureBuilder.

        Args:
            path (str): Path to PitchPreprocessor artifact (.pkl).
            trackman_artifact_path (str, optional): Override trackman artifact path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {path}")
        self.artifacts = joblib.load(path)

        # Reload TrackmanFeatureBuilder from its artifact
        tkm_path = trackman_artifact_path or self.artifacts.get(
            "tkm_artifact_path",
            os.path.join(config.ARTIFACTS_DIR, "trackman_artifacts.pkl")
        )
        self.trackman_builder = TrackmanFeatureBuilder().load(tkm_path)
        self.is_fitted = True
        print(f"[PitchPreprocessor] Artifacts loaded from {path}")
        return self


if __name__ == "__main__":
    import time as _time

    print("=" * 60)
    print("FULL PIPELINE VERIFICATION")
    print("train.csv (1,475,092 rows) + trackman_history.csv (1,793,078 rows)")
    print("=" * 60)

    t_wall_start = _time.perf_counter()

    # 1. Load full train
    print("\nLoading train.csv ...")
    df_train_full = pd.read_csv(config.TRAIN_PATH)
    print(f"  Loaded: {df_train_full.shape}")

    # 2. Fit (final mode)
    preprocessor = PitchPreprocessor()
    preprocessor.fit(df_train_full, is_final=True)

    # 3. Transform full train
    print("Transforming full training dataset ...")
    t0 = _time.perf_counter()
    X_train = preprocessor.transform(df_train_full)
    t1 = _time.perf_counter()
    print(f"  Transform time: {t1 - t0:.2f}s")
    print(f"  Output shape: {X_train.shape}")
    print(f"  NaN count: {X_train.isnull().sum().sum()}")

    # 4. Assert exact column match
    expected = config.MODEL_FEATURE_COLS
    actual = list(X_train.columns)
    assert actual == expected, (
        f"Column mismatch!\n  Expected ({len(expected)}): {expected}\n  Got ({len(actual)}): {actual}"
    )
    print(f"\n✅ Assertion PASSED: Transform output matches config.MODEL_FEATURE_COLS exactly! ({len(actual)} cols)")

    # 5. Save and reload test
    pp_artifact = os.path.join(config.ARTIFACTS_DIR, "preprocessor_artifacts.pkl")
    preprocessor.save(pp_artifact)

    clean = PitchPreprocessor().load(pp_artifact)
    df_test_sample = pd.read_csv(config.TEST_PATH)
    X_test = clean.transform(df_test_sample)
    assert list(X_test.columns) == expected
    print("✅ Reload + test.csv transform: PASSED")

    t_wall_end = _time.perf_counter()
    total = t_wall_end - t_wall_start
    limit = config.SERVER_CONSTRAINTS["time_limit_seconds"]
    print(f"\n⏱  Total pipeline time: {total:.2f}s ({total/60:.2f} min)")
    print(f"   10-min limit remaining: {(limit - total)/limit*100:.1f}%")
    print("\n" + "=" * 60)
