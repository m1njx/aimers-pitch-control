import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
model_dir = os.path.join(work_v42_dir, 'model')

sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

print("Testing Empirical Bayes (Pitcher x Count) as a direct dense feature...")
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

is_val24 = (df['season'] == 2024)
is_train = (df['season'] < 2024)
y_train = df.loc[is_train, 'control_success'].values.astype(np.float32)
y_val = df.loc[is_val24, 'control_success'].values.astype(np.float32)

# Build Empirical Bayes ONLY on training data (2018-2023) to guarantee 0 leakage
df_tr = df[is_train].copy()
df_tr['count_code_str'] = (df_tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                           df_tr['strikes_before'].fillna(0).astype(int).astype(str))

p_stats = df_tr.groupby('pitcher_id')['control_success'].agg(['count', 'sum']).reset_index()
p_stats['p_global'] = (p_stats['sum'] + 10.0 * 0.4861) / (p_stats['count'] + 10.0)

pc_stats = df_tr.groupby(['pitcher_id', 'count_code_str'])['control_success'].agg(['count', 'sum']).reset_index()
pc_stats = pc_stats.merge(p_stats[['pitcher_id', 'p_global']], on='pitcher_id', how='left')
M = 15.0
pc_stats['p_count_bayes'] = ((pc_stats['sum'] + M * pc_stats['p_global']) / (pc_stats['count'] + M)).astype(np.float32)

p_count_map = dict(zip(zip(pc_stats['pitcher_id'], pc_stats['count_code_str']), pc_stats['p_count_bayes']))
p_global_map = dict(zip(p_stats['pitcher_id'], p_stats['p_global']))

# Map to full df
df_cc = (df['balls_before'].fillna(0).astype(int).astype(str) + '_' + df['strikes_before'].fillna(0).astype(int).astype(str)).values
p_ids = df['pitcher_id'].values

feat_bayes = np.zeros(len(df), dtype=np.float32)
for i in range(len(df)):
    key = (p_ids[i], df_cc[i])
    if key in p_count_map:
        feat_bayes[i] = p_count_map[key]
    elif p_ids[i] in p_global_map:
        feat_bayes[i] = p_global_map[p_ids[i]]
    else:
        feat_bayes[i] = 0.4861

print(f"Feat Bayes mapped: mean={feat_bayes.mean():.4f}, std={feat_bayes.std():.4f}")

# Direct Brier Skill of this single Bayes feature on 2024 Val
s_raw_bayes = brier_skill(y_val, feat_bayes[is_val24])
p_cal_bayes = np.clip(0.5 + 1.10 * (feat_bayes[is_val24] - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
s_cal_bayes = brier_skill(y_val, p_cal_bayes)

print(f"Raw Bayes Feature Standalone Val Score:       {s_raw_bayes:.2f} pts")
print(f"Calibrated Bayes Feature Standalone Val Score: {s_cal_bayes:.2f} pts 🚀")
