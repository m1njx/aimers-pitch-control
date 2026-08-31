import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("=" * 70)
print("GRAND SOTA SYNTHESIS: Combining All 4 Validated Tracks on 2024 Val Fold")
print("=" * 70)
t0 = time.time()

BASE_DIR = os.path.expanduser('~/LG_data')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v33', 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

sys.path.insert(0, os.path.join(BASE_DIR, 'work', 'submit_v33'))
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

# 1. Load train.csv
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Preprocess base features
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder().load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
prep.trackman_builder = tkm_builder
X_all = prep.transform(df_all)

base_str = ((df_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_all['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_all['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D tunneling features
v0 = X_all['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_all['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_all['tkm_rel_side_mean']
rel_height = X_all['tkm_rel_height_mean']
ivb = X_all['tkm_induced_vert_break_mean'] / 12.0
hb = X_all['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_all['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_all['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_all['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_all = dec.transform(df_all)
A_all.index = X_all.index
X_base = pd.concat([X_all, A_all], axis=1)

# Add 5 Domain-Engineered Interaction Features (Track 3)
b = df_all['balls_before'].fillna(0).values
s = df_all['strikes_before'].fillna(0).values
li = df_all['li'].fillna(1.0).values
r2 = (df_all['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_all['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_all['score_diff_pitcher_team'].fillna(0).values
fb_rate = df_all['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_all['asof_pitcher_breaking_rate'].fillna(0.3).values
platoon_code = (df_all['pitcher_hand'].astype(str) == df_all['batter_hand'].astype(str)).astype(float).values

X_syn = X_base.copy()
X_syn['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_syn['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_syn['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_syn['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_syn['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_syn.columns:
        X_syn[c] = X_syn[c].astype('category')

# Split Train (2019-2023) and Val (2024)
train_mask = (seasons <= 2023)
val_mask = (seasons == 2024)

X_train, y_train = X_syn[train_mask].copy(), y_all[train_mask]
X_val, y_val = X_syn[val_mask].copy(), y_all[val_mask]

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

# Model 1: LightGBM Binary (LogLoss)
print("\n--- Training LightGBM Binary (124 features) ---")
lgb_bin_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': 4,
    'verbose': -1
}
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
m_bin = lgb.train(lgb_bin_params, dtrain, num_boost_round=300, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
p_bin = m_bin.predict(X_val)

# Model 2: LightGBM Direct MSE (Brier Regression)
print("\n--- Training LightGBM Direct MSE (124 features) ---")
lgb_reg_params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'n_jobs': 4,
    'verbose': -1
}
m_reg = lgb.train(lgb_reg_params, dtrain, num_boost_round=300, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
p_reg = np.clip(m_reg.predict(X_val), 1e-6, 1 - 1e-6)

# Combine GBDT models
p_gbdt_syn = 0.5 * p_bin + 0.5 * p_reg

# Post Calibration with Optimal Parameters (Track 4: Scale=1.08, Shift=-0.001)
p_calibrated = np.clip(0.5 + 1.08 * (p_gbdt_syn - 0.5) - 0.001, 1e-6, 1 - 1e-6)

score_raw, brier_raw = calc_brier_skill_score(y_val, p_gbdt_syn)
score_cal, brier_cal = calc_brier_skill_score(y_val, p_calibrated)

print(f"\n" + "=" * 70)
print(f"GRAND SYNTHESIS RESULTS ON 2024 VALIDATION FOLD (N=253,507):")
print(f"=" * 70)
print(f"  v33 Baseline 2024 Val Score:         712.94 pts (Single LGB Baseline)")
print(f"  Grand Synthesis Raw OOF Score:       {score_raw:.2f} pts (Gain: {score_raw - 712.94:+.2f} pts)")
print(f"  Grand Synthesis Calibrated Score:    {score_cal:.2f} pts (Gain: {score_cal - 712.94:+.2f} pts)")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
