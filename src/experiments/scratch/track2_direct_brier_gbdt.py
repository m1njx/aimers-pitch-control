import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

print("=" * 70)
print("TRACK 2: Direct Brier (MSE Regression) vs LogLoss (Binary) on GBDT")
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

# Preprocess features
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
X_all = pd.concat([X_all, A_all], axis=1)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_all.columns:
        X_all[c] = X_all[c].astype('category')

# Split Train (2019-2023) and Val (2024)
train_mask = (seasons <= 2023)
val_mask = (seasons == 2024)

X_train, y_train = X_all[train_mask].copy(), y_all[train_mask]
X_val, y_val = X_all[val_mask].copy(), y_all[val_mask]
print(f"Train set (2019-2023): {len(X_train):,} rows | Val set (2024): {len(X_val):,} rows")

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

# 1. Train LightGBM Binary (LogLoss)
print("\n--- Training LightGBM Binary (LogLoss) ---")
lgb_binary_params = {
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

model_bin = lgb.train(lgb_binary_params, dtrain, num_boost_round=300, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
p_bin_val = model_bin.predict(X_val)
score_bin, brier_bin = calc_brier_skill_score(y_val, p_bin_val)
print(f"LightGBM Binary 2024 Val Skill Score: {score_bin:.2f} (Brier: {brier_bin:.6f})")

# 2. Train LightGBM Regression (Direct MSE / Brier Objective)
print("\n--- Training LightGBM Regression (Direct MSE / Brier Objective) ---")
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
model_reg = lgb.train(lgb_reg_params, dtrain, num_boost_round=300, valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
p_reg_val = np.clip(model_reg.predict(X_val), 1e-6, 1 - 1e-6)
score_reg, brier_reg = calc_brier_skill_score(y_val, p_reg_val)
print(f"LightGBM Regression (Direct MSE) 2024 Val Skill Score: {score_reg:.2f} (Brier: {brier_reg:.6f})")

# Ensemble Blend
p_blend = 0.5 * p_bin_val + 0.5 * p_reg_val
score_blend, brier_blend = calc_brier_skill_score(y_val, p_blend)
print(f"\n[SUMMARY] Direct Brier vs Binary LogLoss Comparison on 2024 Val:")
print(f"  Binary (LogLoss) Skill Score:      {score_bin:.2f}")
print(f"  Regression (Direct MSE) Score:    {score_reg:.2f}")
print(f"  Binary + Regression 50:50 Blend:   {score_blend:.2f} (Gain vs Binary: {score_blend - score_bin:+.2f} pts)")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
