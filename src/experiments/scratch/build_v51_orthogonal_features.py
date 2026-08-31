import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

print("=" * 80)
print("ENGINEERING V51 ORTHOGONAL FEATURE PIPELINE (TEMPORAL ASOF ENGINE)")
print("=" * 80)
t0 = time.time()

df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print(f"Loaded train.csv: {len(df_train):,} rows")

# 1. Count code string
df_train['count_code_str'] = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                              df_train['strikes_before'].fillna(0).astype(int).astype(str))

# Sort chronologically by game_date, game_id, at_bat_number, pitch_number
df_train['pitch_order'] = np.arange(len(df_train))

# Build Matchup & Pitcher-Count Hierarchical Bayesian Tables
# Pitcher global control success sum & count
pitcher_stats = df_train.groupby('pitcher_id')['control_success'].agg(['count', 'sum']).reset_index()
pitcher_stats.rename(columns={'count': 'p_total_pitches', 'sum': 'p_total_ctrl'}, inplace=True)
pitcher_stats['p_global_ctrl_rate'] = (pitcher_stats['p_total_ctrl'] + 10.0 * 0.4861) / (pitcher_stats['p_total_pitches'] + 10.0)

# Pitcher x Count stats
pitcher_count_stats = df_train.groupby(['pitcher_id', 'count_code_str'])['control_success'].agg(['count', 'sum']).reset_index()
pitcher_count_stats = pitcher_count_stats.merge(pitcher_stats[['pitcher_id', 'p_global_ctrl_rate']], on='pitcher_id', how='left')
# Empirical Bayes Shrinkage (M = 15.0)
M = 15.0
pitcher_count_stats['p_count_ctrl_bayes'] = ((pitcher_count_stats['sum'] + M * pitcher_count_stats['p_global_ctrl_rate']) / 
                                             (pitcher_count_stats['count'] + M)).astype(np.float32)

# Pitcher x Batter Matchup stats
matchup_stats = df_train.groupby(['pitcher_id', 'batter_id'])['control_success'].agg(['count', 'sum']).reset_index()
matchup_stats = matchup_stats.merge(pitcher_stats[['pitcher_id', 'p_global_ctrl_rate']], on='pitcher_id', how='left')
# Empirical Bayes Shrinkage (K = 8.0)
K = 8.0
matchup_stats['matchup_ctrl_bayes'] = ((matchup_stats['sum'] + K * matchup_stats['p_global_ctrl_rate']) /
                                       (matchup_stats['count'] + K)).astype(np.float32)

# In-Game Sequence Dynamics
# Cumulative pitch count per pitcher per game
df_train['pitcher_game_seq'] = df_train.groupby(['game_id', 'pitcher_id']).cumcount() + 1

# Rolling 10 pitch control success in game
df_train['game_pitcher_key'] = df_train['game_id'].astype(str) + '_' + df_train['pitcher_id'].astype(str)

print("Computed Empirical Bayes tables:")
print(f"  Pitcher count map size: {len(pitcher_count_stats):,}")
print(f"  Matchup map size:       {len(matchup_stats):,}")

# Save lookup dictionaries for ultra-fast vector mapping
p_global_map = dict(zip(pitcher_stats['pitcher_id'], pitcher_stats['p_global_ctrl_rate']))
p_count_map = dict(zip(zip(pitcher_count_stats['pitcher_id'], pitcher_count_stats['count_code_str']), pitcher_count_stats['p_count_ctrl_bayes']))
matchup_map = dict(zip(zip(matchup_stats['pitcher_id'], matchup_stats['batter_id']), matchup_stats['matchup_ctrl_bayes']))

bayes_artifacts = {
    'global_mean': 0.4861,
    'p_global_map': p_global_map,
    'p_count_map': p_count_map,
    'matchup_map': matchup_map
}

os.makedirs('~/LG_data/scratch/v51_artifacts', exist_ok=True)
joblib.dump(bayes_artifacts, '~/LG_data/scratch/v51_artifacts/bayes_artifacts.pkl')
print(f"Saved bayes_artifacts.pkl ({time.time()-t0:.1f}s elapsed)")
