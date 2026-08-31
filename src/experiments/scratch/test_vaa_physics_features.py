import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_DIR = os.path.expanduser('~/LG_data')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v33', 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

sys.path.insert(0, os.path.join(BASE_DIR, 'work', 'submit_v33'))
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

print("=" * 80)
print("TESTING VAA, EFFECTIVE VELOCITY & SPIN EFFICIENCY PHYSICS FEATURES")
print("=" * 80)

# Load data
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder().load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
prep.trackman_builder = tkm_builder
X_base = prep.transform(df_all)

base_str = ((df_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_all['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D tunneling features
v0 = X_base['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_base['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_base['tkm_rel_side_mean']
rel_height = X_base['tkm_rel_height_mean']
ivb = X_base['tkm_induced_vert_break_mean'] / 12.0
hb = X_base['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_all = dec.transform(df_all)
A_all.index = X_base.index
X_base = pd.concat([X_base, A_all], axis=1)

# New Sabermetric Physics Features
v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_phys = X_base.copy()
X_phys['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_phys['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_phys['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_phys['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_base.columns:
        X_base[c] = X_base[c].astype('category')
        X_phys[c] = X_phys[c].astype('category')

tr_m = (seasons <= 2023)
val_m = (seasons == 2024)

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

# 1. Base Model
dtrain_b = lgb.Dataset(X_base[tr_m], label=y_all[tr_m])
dval_b = lgb.Dataset(X_base[val_m], label=y_all[val_m], reference=dtrain_b)
m_b = lgb.train(lgb_params, dtrain_b, num_boost_round=300, valid_sets=[dval_b], callbacks=[lgb.early_stopping(40, verbose=False)])
p_b = m_b.predict(X_base[val_m])
sc_b, _ = calc_brier_skill_score(y_all[val_m], p_b)

# 2. Physics Augmented Model
dtrain_p = lgb.Dataset(X_phys[tr_m], label=y_all[tr_m])
dval_p = lgb.Dataset(X_phys[val_m], label=y_all[val_m], reference=dtrain_p)
m_p = lgb.train(lgb_params, dtrain_p, num_boost_round=300, valid_sets=[dval_p], callbacks=[lgb.early_stopping(40, verbose=False)])
p_p = m_p.predict(X_phys[val_m])
sc_p, _ = calc_brier_skill_score(y_all[val_m], p_p)

print(f"Base Features (119f) Score:           {sc_b:.2f} pts")
print(f"With 4 Sabermetric Physics (123f):   {sc_p:.2f} pts (Gain: {sc_p - sc_b:+.2f} pts)")
