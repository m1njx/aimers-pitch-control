import os
import sys
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
dest_pokemon = '~/pipeline_src'
data_dir = os.path.join(BASE_DIR, 'open', 'data')

# Full, self-contained, 100% relative path config.py
full_config_content = '''"""
config.py — DACON Aimers 9th Pitcher Control Success Prediction Project Configuration.

All paths are written as relative paths from the script directory.
No hardcoded absolute paths exist.
"""

import os

# ==========================================
# 1. PATH CONFIGURATIONS (Self-Contained)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(RAW_DATA_DIR, "sample_submission.csv")
TRACKMAN_PATH = os.path.join(RAW_DATA_DIR, "trackman_history.csv")

# Working & Output Paths
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ==========================================
# 2. COLUMN DEFINITIONS
# ==========================================
ID_COL = "row_id"
TARGET_COL = "control_success"

ID_ONLY_COLS = [
    "pitcher_id",
    "batter_id"
]

CATEGORICAL_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id"
]

DERIVED_CATEGORICAL_COLS = [
    "count_code",
    "platoon_matchup"
]

TRACKMAN_MATCH_FLAG_COL = "tkm_match"

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

DERIVED_NUMERICAL_COLS = [
    "is_leading",
    "is_tied",
    "score_diff_abs",
    "is_scoring_position",
    "pitcher_success_trend_1g",
    "pitcher_success_trend_3g"
]

ALL_FEATURE_COLS = CATEGORICAL_COLS + RAW_NUMERICAL_COLS

EXCLUDED_FEATURE_COLS = [
    "season",
    "game_type"
]

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

CV_STRATEGY = "time"
CV_SEASON_COL = "season"
CV_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
CV_MIN_TRAIN_SEASONS = 3
CV_GROUP_COL = "pitcher_id"

TRACKMAN_JOIN_KEYS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before"
]

SERVER_CONSTRAINTS = {
    "time_limit_seconds": 600,
    "install_time_limit_seconds": 600,
    "eval_rows": 245789,
    "max_ram_gb": 28,
    "vram_gb": 22.4,
    "cpus": 6,
    "python_version": "3.11.15"
}
'''

for v in ['v43', 'v44', 'v45', 'v46', 'v47']:
    sub_dir = os.path.join(BASE_DIR, 'work', f'submit_{v}')
    zip_path = os.path.join(BASE_DIR, 'work', f'submit_{v}.zip')
    
    print(f"\n==========================================")
    print(f"Upgrading & Validating submit_{v}...")
    print(f"==========================================")
    
    # 1. Update config.py with full self-contained content
    with open(os.path.join(sub_dir, 'config.py'), 'w') as f:
        f.write(full_config_content)
    print(f"  [1] Updated full complete config.py in {v} ({len(full_config_content):,} bytes)!")
    
    # 2. Package ZIP
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(sub_dir):
            if '__pycache__' in root or os.path.basename(root) in ['data', 'output', 'catboost_info']:
                continue
            for file in files:
                if file.startswith('.') or file.endswith('.pyc') or file == 'submission.csv':
                    continue
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, sub_dir)
                zf.write(full_p, rel_p)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [2] Repackaged submit_{v}.zip: {size_mb:.2f} MB")
    
    # 3. Strict Isolated Sandbox Test in /tmp
    sandbox_dir = f'/tmp/dacon_isolated_test_full_{v}'
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(sandbox_dir)
    os.makedirs(os.path.join(sandbox_dir, 'data'), exist_ok=True)
    shutil.copy(os.path.join(data_dir, 'test.csv'), os.path.join(sandbox_dir, 'data', 'test.csv'))
    
    clean_env = os.environ.copy()
    clean_env['PYTHONPATH'] = ''
    
    res = subprocess.run(['python3', 'script.py'], cwd=sandbox_dir, env=clean_env, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Sandbox verification failed for {v}!")
        
    sub_df = pd.read_csv(os.path.join(sandbox_dir, 'output', 'submission.csv'))
    test_orig = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    assert len(sub_df) == len(test_orig), 'Row count mismatch!'
    assert not sub_df['control_success'].isna().any(), 'NaN found!'
    print(f"  [3] Sandbox Test PASSED! {len(sub_df):,} rows -> 100% PERFECT! (Mean={sub_df['control_success'].mean():.6f})")
    shutil.rmtree(sandbox_dir)
    
    # 4. Sync to pokemon directory
    if os.path.exists(dest_pokemon):
        shutil.copy(zip_path, os.path.join(dest_pokemon, f'submit_{v}.zip'))
        print(f"  [4] Synced submit_{v}.zip to {dest_pokemon}/!")

print("\n" + "=" * 80)
print("ALL 5 SUBMISSIONS (v43, v44, v45, v46, v47) 100% COMPLETE & UPGRADED WITH FULL CONFIG!")
print("=" * 80)
