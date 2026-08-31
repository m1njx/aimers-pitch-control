#!/usr/bin/env python3
"""
frontier6_platoon_mixture_of_experts.py — Frontier 6: Platoon-Specific Mixture of Experts (1150+ Target)

Trains dedicated specialized sub-models:
- Expert 1: Same-Hand Matchup (RHP vs RHB, LHP vs LHB)
- Expert 2: Opposite-Hand Matchup (RHP vs LHB, LHP vs RHB)
Evaluates Brier Skill gain on 2024 Val Fold.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v40_dir = os.path.join(BASE_DIR, 'work', 'submit_v40')
if submit_v40_dir not in sys.path:
    sys.path.insert(0, submit_v40_dir)

model_dir = os.path.join(submit_v40_dir, 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
report_dir = os.path.join(BASE_DIR, 'gemini_reports_for_ai')
output_dir = os.path.join(BASE_DIR, 'outputs')

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

log("=" * 80)
log("STARTING FRONTIER 6: PLATOON MIXTURE OF EXPERTS (1150+ TARGET)")
log("=" * 80)

# Load data and 136 features
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
tkm_builder = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
if hasattr(tkm_builder, 'transform'):
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

# 4 Sabermetric Physics Features
v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_all_f = X_base.copy()
X_all_f['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_all_f['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_all_f['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_all_f['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

# 10 Domain Interaction Features
b = df_all['balls_before'].fillna(0).values
s = df_all['strikes_before'].fillna(0).values
li = df_all['li'].fillna(1.0).values
r1 = (df_all['runner_on_1b'].fillna(0) > 0).astype(float).values
r2 = (df_all['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_all['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_all['score_diff_pitcher_team'].fillna(0).values
inning = df_all['inning'].fillna(1).values

fb_rate = df_all['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_all['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df_all['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df_all['pitcher_hand'].astype(str) == df_all['batter_hand'].astype(str)).astype(float).values

X_all_f['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_all_f['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_all_f['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_all_f['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_all_f['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_all_f['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_all_f['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_all_f['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_all_f['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_all_f['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

# 3 Pitch Tunneling Differentials
X_all_f['tunnel_delta_speed'] = (v_rel - 92.0).astype(np.float32)
X_all_f['tunnel_delta_ivb'] = (ivb - 1.2).astype(np.float32)
X_all_f['tunnel_delta_hb'] = np.abs(hb).astype(np.float32)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_all_f.columns:
        X_all_f[c] = X_all_f[c].astype('category')

tr_2024 = (seasons <= 2023)
val_2024 = (seasons == 2024)

# Platoon Matchup Masks
same_hand_tr = tr_2024 & (df_all['pitcher_hand'].astype(str) == df_all['batter_hand'].astype(str))
opp_hand_tr = tr_2024 & (df_all['pitcher_hand'].astype(str) != df_all['batter_hand'].astype(str))

same_hand_val = val_2024 & (df_all['pitcher_hand'].astype(str) == df_all['batter_hand'].astype(str))
opp_hand_val = val_2024 & (df_all['pitcher_hand'].astype(str) != df_all['batter_hand'].astype(str))

log(f"Same-Hand Matchup Count: Train={same_hand_tr.sum():,}, Val={same_hand_val.sum():,}")
log(f"Opposite-Hand Matchup Count: Train={opp_hand_tr.sum():,}, Val={opp_hand_val.sum():,}")

# Train Expert 1 (Same Hand)
log("Training Expert 1 (Same-Hand Matchup Direct MSE)...")
dtr_same = lgb.Dataset(X_all_f[same_hand_tr], label=y_all[same_hand_tr])
dv_same = lgb.Dataset(X_all_f[same_hand_val], label=y_all[same_hand_val], reference=dtr_same)
m_same = lgb.train({
    'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
    'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'random_state': 42, 'n_jobs': 4, 'verbose': -1
}, dtr_same, num_boost_round=350, valid_sets=[dv_same], callbacks=[lgb.early_stopping(30, verbose=False)])

p_same_val = np.clip(m_same.predict(X_all_f[same_hand_val]), 1e-6, 1 - 1e-6)

# Train Expert 2 (Opposite Hand)
log("Training Expert 2 (Opposite-Hand Matchup Direct MSE)...")
dtr_opp = lgb.Dataset(X_all_f[opp_hand_tr], label=y_all[opp_hand_tr])
dv_opp = lgb.Dataset(X_all_f[opp_hand_val], label=y_all[opp_hand_val], reference=dtr_opp)
m_opp = lgb.train({
    'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
    'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'random_state': 42, 'n_jobs': 4, 'verbose': -1
}, dtr_opp, num_boost_round=350, valid_sets=[dv_opp], callbacks=[lgb.early_stopping(30, verbose=False)])

p_opp_val = np.clip(m_opp.predict(X_all_f[opp_hand_val]), 1e-6, 1 - 1e-6)

# Combine predictions
p_moe_val = np.zeros(val_2024.sum(), dtype=np.float32)
val_idx = np.where(val_2024)[0]
same_val_local = (df_all.loc[val_2024, 'pitcher_hand'].astype(str) == df_all.loc[val_2024, 'batter_hand'].astype(str)).values
opp_val_local = ~same_val_local

p_moe_val[same_val_local] = p_same_val
p_moe_val[opp_val_local] = p_opp_val

sc_moe, brier_moe = calc_brier_skill_score(y_all[val_2024], p_moe_val)
log(f"Platoon MoE Combined Solo Score (2024 Val): {sc_moe:.2f} pts (Brier: {brier_moe:.6f})")

# Write Report 317
rep317_path = os.path.join(report_dir, '317_platoon_mixture_of_experts.md')
with open(rep317_path, 'w') as f:
    f.write(f"""# 📊 [실측 보고서] Exp 317: 플래툰 전문가 혼합 모델 (Platoon MoE) 실측

- **구성**:
  - Expert 1 (동손 매치업, R-R / L-L, N={same_hand_tr.sum():,})
  - Expert 2 (이손 매치업, R-L / L-R, N={opp_hand_tr.sum():,})
- **검증 데이터**: 2024 Val Fold (N = 253,507)

## 실측 결과
- Platoon MoE 단독 2024 Val Score: **{sc_moe:.2f}점**
""")
os.system(f"cp {rep317_path} {os.path.join(output_dir, '317_platoon_mixture_of_experts.md')}")
log("Saved Report 317!")
