"""
config.py — DACON Aimers 9th Pitcher Control Success Prediction Project Configuration.

All paths are written as relative paths from the LG_data root directory.
No hardcoded paths or column names should exist in downstream scripts.
"""

import os

# ==========================================
# 1. PATH CONFIGURATIONS (Relative to LG_data)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "open", "data")
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(RAW_DATA_DIR, "sample_submission.csv")
TRACKMAN_PATH = os.path.join(RAW_DATA_DIR, "trackman_history.csv")
DATA_DOC_PATH = os.path.join(BASE_DIR, "open", "data_description.md")

# Working & Output Paths
WORK_DIR = os.path.join(BASE_DIR, "work")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACTS_DIR = os.path.join(WORK_DIR, "artifacts")
MODEL_DIR = os.path.join(WORK_DIR, "model")
SUBMISSION_DIR = os.path.join(WORK_DIR, "output")

# Submission output path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# 2. COLUMN DEFINITIONS
# ==========================================
# Identifiers & Target
ID_COL = "row_id"
TARGET_COL = "control_success"

# Identifiers ONLY (Excluded from Model Features, used for CV Grouping / Trackman Validation)
ID_ONLY_COLS = [
    "pitcher_id",
    "batter_id"
]

# Categorical Features (Raw)
CATEGORICAL_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id"
]

# Derived Categorical Features (Added in Preprocessing)
DERIVED_CATEGORICAL_COLS = [
    "count_code",
    "platoon_matchup"
]

# Trackman Join Keys flag column (1=matched, 0=unmatched/fallback to global mean)
TRACKMAN_MATCH_FLAG_COL = "tkm_match"

# Trackman Prior Feature Columns (produced by TrackmanFeatureBuilder)
TRACKMAN_DERIVED_COLS = [
    "tkm_rel_speed_mean",
    "tkm_rel_speed_std",
    "tkm_spin_rate_mean",
    "tkm_spin_rate_std",
    "tkm_induced_vert_break_mean",
    "tkm_induced_vert_break_std",
    "tkm_horz_break_mean",
    "tkm_horz_break_std",
    "tkm_extension_mean",
    "tkm_extension_std",
    "tkm_rel_height_mean",
    "tkm_rel_height_std",
    "tkm_rel_side_mean",
    "tkm_rel_side_std",
    "tkm_zone_speed_mean",
    "tkm_zone_speed_std",
    "tkm_n_pitches",
    TRACKMAN_MATCH_FLAG_COL,
]

# Numerical Features (Raw)
RAW_NUMERICAL_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate"
]

# Derived Numerical Features (Added in Preprocessing)
DERIVED_NUMERICAL_COLS = [
    "is_leading",
    "is_tied",
    "score_diff_abs",
    "is_scoring_position",
    "pitcher_success_trend_1g",
    "pitcher_success_trend_3g"
]

# Raw Feature Whitelist (All valid input features from raw csv excluding IDs/Target)
ALL_FEATURE_COLS = CATEGORICAL_COLS + RAW_NUMERICAL_COLS

# Explicitly Excluded Feature Whitelist
# Rationale:
#   - "season"   : Causes GBDT tree extrapolation error on test season=2025 without AUC gain (17_season_drift_risk.md)
#   - "game_type": Overfits to exhibition game ('F') vs regular season ('R') target mean, causing 2023 CV AUC drop (19_game_type_analysis.md & 20_combined_exclusion_ablation.md)
EXCLUDED_FEATURE_COLS = [
    "season",
    "game_type"
]

# Full Model Input Feature Whitelist (Raw Features + Derived Features + Trackman Prior Features - Excluded Features)
MODEL_FEATURE_COLS = [
    c for c in (
        CATEGORICAL_COLS
        + DERIVED_CATEGORICAL_COLS
        + [TRACKMAN_MATCH_FLAG_COL]
        + RAW_NUMERICAL_COLS
        + DERIVED_NUMERICAL_COLS
        + [c for c in TRACKMAN_DERIVED_COLS if c != TRACKMAN_MATCH_FLAG_COL]
    )
    if c not in EXCLUDED_FEATURE_COLS
]

# ==========================================
# 3. CROSS-VALIDATION STRATEGY
# ==========================================
# CONFIRMED (11_test_split_strategy.md):
#   train = 2019~2024 (6 seasons), test = 2025 (1 full season, season=2025 confirmed)
#   Correct CV strategy: TEMPORAL (season-based expanding window)
#
# CV_STRATEGY options:
#   "time"  → Season expanding window (recommended, matches train/test temporal split)
#   "group" → GroupKFold on pitcher_id (available as alternative; may overfit)
#   "hybrid" → Season-based outer split + pitcher-group awareness (future option)

CV_STRATEGY = "time"          # Primary: season-based expanding window
CV_SEASON_COL = "season"      # Column used to define time-based fold boundaries

# Seasons used in expanding window CV (chronological order)
# Fold k: train on CV_SEASONS[:k+MIN_TRAIN_SEASONS], validate on CV_SEASONS[k+MIN_TRAIN_SEASONS]
CV_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
CV_MIN_TRAIN_SEASONS = 3      # Minimum seasons before starting a validation fold
# → Folds: [2019-2021→2022], [2019-2022→2023], [2019-2023→2024]  (3 folds)

# ⚠️  COVID-shortened season note:
# 2020 was a COVID-shortened season (May~Oct only; no March/April games).
# All folds that include 2020 in their train set (= ALL 3 folds here) should be
# interpreted with this in mind: early-season pitch distributions are underrepresented
# in 2020 relative to other seasons.
# cv_utils.py / FoldInfo.notes will automatically surface this warning per-fold.
CV_COVID_SEASONS = [2020]     # Seasons with known structural differences

# Group col kept for GroupKFold alternative experiments (compare vs CV_STRATEGY=time)
CV_GROUP_COL = "pitcher_id"   # Used when CV_STRATEGY="group" (alternative baseline)

# Grouping & Time Order metadata
TIME_ORDER_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom"
]

# Trackman Join Keys (7-key, season-agnostic)
# NOTE: 'season' is intentionally EXCLUDED from the join key.
# Rationale: Trackman physical stats (rel_speed, spin_rate, etc.) measure
# situation-level biomechanics, NOT year-specific effects. Including 'season'
# creates 4.2x more unique states with fewer samples each, AND causes
# fold-aware CV val sets (different season than train) to get 0% match —
# degrading those features to global-mean fallback.
# With 7-key (no season): val match rate = 99.3~99.5% across all folds.
# (Confirmed in 13_trackman_temporal_leakage_check.md)
TRACKMAN_JOIN_KEYS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before"
]

# ==========================================
# 3. EVALUATION SERVER CONSTRAINTS & PARAMS
# ==========================================
SERVER_CONSTRAINTS = {
    "time_limit_seconds": 600,       # 10 minutes
    "install_time_limit_seconds": 600,
    "eval_rows": 245789,             # Expected test set size
    "max_ram_gb": 28,
    "vram_gb": 22.4,
    "cpus": 6,
    "python_version": "3.11.15",
    "preinstalled_packages": {
        "torch": "2.7.1+cu128",
        "pandas": "2.0.3",
        "numpy": "1.26.4",
        "scikit-learn": "1.8.0",
        "scipy": "1.15.3"
    }
}
