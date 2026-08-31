import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

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

print("=" * 80)
print("HEAD-TO-HEAD COMPARISON: v42 (133 Feats) vs v51 (143 Feats) FULL PIPELINE")
print("=" * 80)

df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Preprocessing
tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))

X_base = prep.transform(df)

base_str = ((df['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D Tunneling
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

A_all = dec.transform(df)
A_all.index = X_base.index

v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

b = df['balls_before'].fillna(0).values
s = df['strikes_before'].fillna(0).values
li = df['li'].fillna(1.0).values
r2 = (df['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df['score_diff_pitcher_team'].fillna(0).values
inning = df['inning'].fillna(1).values
fb_rate = df['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df['pitcher_hand'].astype(str) == df['batter_hand'].astype(str)).astype(float).values

p_succ = df['asof_pitcher_success_rate'].fillna(0.4861).values
b_succ = df['asof_batter_success_rate'].fillna(0.4861).values
p_prev1 = df['asof_pitcher_prev1_game_success_rate'].fillna(df['asof_pitcher_success_rate']).fillna(0.4861).values
p_prev3 = df['asof_pitcher_prev3_game_success_rate'].fillna(df['asof_pitcher_success_rate']).fillna(0.4861).values
p_prev5 = df['asof_pitcher_prev5_game_success_rate'].fillna(df['asof_pitcher_success_rate']).fillna(0.4861).values
p_ball = df['asof_pitcher_ball_rate'].fillna(0.35).values
p_str = df['asof_pitcher_strike_rate'].fillna(0.65).values
p_mid = df['asof_pitcher_middle_rate'].fillna(0.08).values
b_mid = df['asof_batter_middle_rate'].fillna(0.08).values

# 133 features (v42 baseline)
extra_133 = {
    'tkm_tunnel_dist_015s': d_tunnel.astype(np.float32),
    'tkm_plate_break_divergence': ((d_plate - d_tunnel) / 0.15).astype(np.float32),
    'tkm_deception_index': (d_plate / (d_tunnel + 0.1)).astype(np.float32),
    'phys_effective_velocity': (v_rel * (60.5 / dist_to_plate)).astype(np.float32),
    'phys_vaa_proxy': (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32),
    'phys_haa_proxy': (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32),
    'phys_spin_efficiency': (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32),
    'feat_count_advantage': (s - 1.5 * b).astype(np.float32),
    'feat_full_count': ((b == 3) & (s == 2)).astype(np.float32),
    'feat_pitcher_ahead': ((s > b) & (s >= 2)).astype(np.float32),
    'feat_pitcher_behind': ((b > s) & (b >= 2)).astype(np.float32),
    'feat_clutch_pressure': (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32),
    'feat_scoring_position': (r2 + r3).astype(np.float32),
    'feat_platoon_fastball_inter': (platoon_code * fb_rate).astype(np.float32),
    'feat_platoon_breaking_inter': (platoon_code * br_rate).astype(np.float32),
    'feat_platoon_offspeed_inter': (platoon_code * off_rate).astype(np.float32),
    'feat_late_inning_clutch': ((inning >= 7).astype(float) * li).astype(np.float32),
}

extra_143 = extra_133.copy()
extra_143.update({
    'feat_pitcher_batter_edge': (p_succ - b_succ).astype(np.float32),
    'feat_pitcher_batter_ratio': (p_succ / (b_succ + 0.01)).astype(np.float32),
    'feat_pitcher_form_delta_1g': (p_prev1 - p_succ).astype(np.float32),
    'feat_pitcher_form_delta_3g': (p_prev3 - p_succ).astype(np.float32),
    'feat_pitcher_form_delta_5g': (p_prev5 - p_succ).astype(np.float32),
    'feat_pitcher_form_acceleration': (p_prev1 - p_prev3).astype(np.float32),
    'feat_pitcher_strike_command_ratio': (p_str / (p_ball + 0.01)).astype(np.float32),
    'feat_middle_zone_mistake_risk': (p_mid * b_mid * 100.0).astype(np.float32),
    'feat_pitcher_stamina_drain': (inning * (1.0 - p_succ)).astype(np.float32),
    'feat_clutch_x_momentum': (li * (p_prev1 - 0.4861)).astype(np.float32)
})

X_133 = pd.concat([X_base, A_all, pd.DataFrame(extra_133, index=X_base.index)], axis=1)
X_143 = pd.concat([X_base, A_all, pd.DataFrame(extra_143, index=X_base.index)], axis=1)

is_val24 = (df['season'] == 2024)
is_train = (df['season'] < 2024)

y_train = df.loc[is_train, 'control_success'].values.astype(np.float32)
y_val = df.loc[is_val24, 'control_success'].values.astype(np.float32)

# Compare 5-seed CatBoost on 133 vs 143
print("\n[1] Training 5-Seed CatBoost (Binary Logloss) on 133 vs 143...")
cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']

X_133_cb = X_133.copy()
X_143_cb = X_143.copy()
for c in cat_cols:
    X_133_cb[c] = pd.to_numeric(X_133_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
    X_143_cb[c] = pd.to_numeric(X_143_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_133_cb.columns if col not in cat_cols]:
    X_133_cb[c] = pd.to_numeric(X_133_cb[c], errors='coerce').fillna(0.0).astype(np.float32)
for c in [col for col in X_143_cb.columns if col not in cat_cols]:
    X_143_cb[c] = pd.to_numeric(X_143_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

SEEDS = [7, 123, 2025]

cb_params = {
    'iterations': 400,
    'learning_rate': 0.08,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'cat_features': cat_cols,
    'thread_count': 4,
    'verbose': 0
}

p_cb_133 = np.zeros(len(y_val))
p_cb_143 = np.zeros(len(y_val))

for s in SEEDS:
    cb_params['random_seed'] = s
    m133 = CatBoostClassifier(**cb_params)
    m133.fit(X_133_cb[is_train], y_train)
    p_cb_133 += m133.predict_proba(X_133_cb[is_val24])[:, 1] / len(SEEDS)
    
    m143 = CatBoostClassifier(**cb_params)
    m143.fit(X_143_cb[is_train], y_train)
    p_cb_143 += m143.predict_proba(X_143_cb[is_val24])[:, 1] / len(SEEDS)

# Calibration test
p_cb_133_cal = np.clip(0.5 + 1.10 * (p_cb_133 - 0.008 - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
p_cb_143_cal = np.clip(0.5 + 1.10 * (p_cb_143 - 0.008 - 0.5) - 0.0035, 1e-6, 1 - 1e-6)

s_133 = brier_skill(y_val, p_cb_133_cal)
s_143 = brier_skill(y_val, p_cb_143_cal)

print(f"CatBoost 133 Features Val Score: {s_133:.2f} pts")
print(f"CatBoost 143 Features Val Score: {s_143:.2f} pts (Gain: +{s_143 - s_133:.2f} pts 🚀)")
