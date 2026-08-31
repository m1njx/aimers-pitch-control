import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

print("=" * 70)
print("TRACK 3: Rule 4 Compliant Domain-Engineered Baseball Features")
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
X_base = pd.concat([X_all, A_all], axis=1)

# New Domain Features (Rule 4 row-independent strictly calculated per row)
b = df_all['balls_before'].fillna(0).values
s = df_all['strikes_before'].fillna(0).values
li = df_all['li'].fillna(1.0).values
r2 = (df_all['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_all['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_all['score_diff_pitcher_team'].fillna(0).values

fb_rate = df_all['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_all['asof_pitcher_breaking_rate'].fillna(0.3).values
platoon_code = (df_all['pitcher_hand'].astype(str) == df_all['batter_hand'].astype(str)).astype(float).values

X_new = X_base.copy()
X_new['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_new['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_new['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_new['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_new['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_base.columns:
        X_base[c] = X_base[c].astype('category')
        X_new[c] = X_new[c].astype('category')

# Split Train (2019-2023) and Val (2024)
train_mask = (seasons <= 2023)
val_mask = (seasons == 2024)

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

lgb_params = {
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

# 1. Baseline Model (without new features)
print("\n--- Training LightGBM Baseline (119 features) ---")
dtrain_base = lgb.Dataset(X_base[train_mask], label=y_all[train_mask])
dval_base = lgb.Dataset(X_base[val_mask], label=y_all[val_mask], reference=dtrain_base)
model_base = lgb.train(lgb_params, dtrain_base, num_boost_round=300, valid_sets=[dval_base], callbacks=[lgb.early_stopping(50, verbose=False)])
p_base_val = model_base.predict(X_base[val_mask])
score_base, brier_base = calc_brier_skill_score(y_all[val_mask], p_base_val)
print(f"Base Features 2024 Val Skill Score: {score_base:.2f} (Brier: {brier_base:.6f})")

# 2. Model with New Domain Features
print("\n--- Training LightGBM with Domain Features (124 features) ---")
dtrain_new = lgb.Dataset(X_new[train_mask], label=y_all[train_mask])
dval_new = lgb.Dataset(X_new[val_mask], label=y_all[val_mask], reference=dtrain_new)
model_new = lgb.train(lgb_params, dtrain_new, num_boost_round=300, valid_sets=[dval_new], callbacks=[lgb.early_stopping(50, verbose=False)])
p_new_val = model_new.predict(X_new[val_mask])
score_new, brier_new = calc_brier_skill_score(y_all[val_mask], p_new_val)
print(f"With Domain Features 2024 Val Skill Score: {score_new:.2f} (Brier: {brier_new:.6f})")

print(f"\n[SUMMARY] Domain Feature Engineering Impact on 2024 Val:")
print(f"  Base Skill Score:         {score_base:.2f}")
print(f"  With New Features Score:  {score_new:.2f}")
print(f"  Gain vs Baseline:         {score_new - score_base:+.2f} pts")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
