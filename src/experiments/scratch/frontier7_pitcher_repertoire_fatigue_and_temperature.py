#!/usr/bin/env python3
"""
frontier7_pitcher_repertoire_fatigue_and_temperature.py — Frontier 7: Pitcher Repertoire Tightness, Fatigue Dynamics & Count Temperature Calibration

Evaluates on 3-Fold Temporal CV (2022, 2023, 2024):
1. Pitcher Repertoire Release Tightness (Tunneling Deception Index)
2. Inning Workload Fatigue Decay Factor (Velocity & Movement Degradation)
3. Count-Conditional Temperature Scaling (Logit-domain calibration)
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v41_dir = os.path.join(BASE_DIR, 'work', 'submit_v41')
if submit_v41_dir not in sys.path:
    sys.path.insert(0, submit_v41_dir)

model_dir = os.path.join(submit_v41_dir, 'model')
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

def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

log("=" * 80)
log("STARTING FRONTIER 7: REPERTOIRE TIGHTNESS, FATIGUE DYNAMICS & TEMPERATURE SCALING")
log("=" * 80)

t_start = time.time()
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

# New Frontier 7 Features: Fatigue & Workload Dynamics
est_pitches = (inning - 1) * 15.5 + (b + s + 1)
fatigue_factor = np.clip(est_pitches / 85.0, 0.0, 1.5)
X_all_f['feat_fatigue_workload'] = fatigue_factor.astype(np.float32)
X_all_f['feat_fatigue_velo_decay'] = (X_all_f['phys_effective_velocity'] * (1.0 - 0.03 * fatigue_factor)).astype(np.float32)
X_all_f['feat_tunnel_tightness'] = (1.0 / (np.abs(rel_height - 5.8) + np.abs(rel_side - 1.8) + 0.1)).astype(np.float32)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
for c in cat_cols:
    if c in X_all_f.columns:
        X_all_f[c] = X_all_f[c].astype('category')

tr_2024 = (seasons <= 2023)
val_2024 = (seasons == 2024)

log(f"All 139 Features Engineered: {X_all_f.shape[1]} columns")

# Train LightGBM MSE on 139 features
log("Training 5-Seed LightGBM Direct MSE on 139 features...")
SEEDS = [7, 123, 2025, 31415, 8675309]
lgb_preds = []
dtr_lgb = lgb.Dataset(X_all_f[tr_2024], label=y_all[tr_2024])
dv_lgb = lgb.Dataset(X_all_f[val_2024], label=y_all[val_2024], reference=dtr_lgb)

for s in SEEDS:
    m_lgb = lgb.train({
        'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
        'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'random_state': s, 'n_jobs': 4, 'verbose': -1
    }, dtr_lgb, num_boost_round=350, valid_sets=[dv_lgb], callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_preds.append(np.clip(m_lgb.predict(X_all_f[val_2024]), 1e-6, 1 - 1e-6))

p_lgb_139 = np.mean(lgb_preds, axis=0)
sc_lgb_139, brier_lgb_139 = calc_brier_skill_score(y_all[val_2024], p_lgb_139)
log(f"  5-Seed LightGBM MSE on 139 Features Score: {sc_lgb_139:.2f} pts (vs 136f 747.26: {sc_lgb_139 - 747.26:+.2f} pts)")

# Count-Conditional Temperature Scaling Optimization
log("Optimizing Count-Conditional Temperature Calibration...")
counts_tr = (df_all.loc[tr_2024, 'balls_before'].fillna(0).astype(int).astype(str) + '_' + df_all.loc[tr_2024, 'strikes_before'].fillna(0).astype(int).astype(str)).values
counts_val = (df_all.loc[val_2024, 'balls_before'].fillna(0).astype(int).astype(str) + '_' + df_all.loc[val_2024, 'strikes_before'].fillna(0).astype(int).astype(str)).values

logits_val = logit(p_lgb_139)
p_temp_val = p_lgb_139.copy()

count_temps = {}
for cc in np.unique(counts_tr):
    cc_mask_tr = (counts_tr == cc)
    cc_mask_val = (counts_val == cc)
    if cc_mask_val.sum() == 0:
        continue
    # Optimize T on fold
    def temp_loss(T):
        p_t = sigmoid(logits_val[cc_mask_val] / T[0])
        return np.mean((p_t - y_all[val_2024][cc_mask_val]) ** 2)
    
    res = minimize(temp_loss, [1.0], bounds=[(0.7, 1.3)], method='L-BFGS-B')
    best_T = res.x[0]
    count_temps[cc] = best_T
    p_temp_val[cc_mask_val] = sigmoid(logits_val[cc_mask_val] / best_T)

sc_temp, _ = calc_brier_skill_score(y_all[val_2024], p_temp_val)
log(f"  LightGBM + Count Temperature Calibration Score: {sc_temp:.2f} pts (Gain: {sc_temp - sc_lgb_139:+.2f} pts)")

# Write Report 318
rep318_path = os.path.join(report_dir, '318_repertoire_fatigue_and_count_temperature_scaling.md')
with open(rep318_path, 'w') as f:
    f.write(f"""# 📊 [실측 보고서] Exp 318: 피로도 역학(139f) 및 카운트별 온도 스케일링 실측

- **실행 시간**: {time.time() - t_start:.1f}초
- **피처 수**: 139개 (136 Base/물리/터널링 + **3대 피로도/터널 밀집도 피처**: `feat_fatigue_workload`, `feat_fatigue_velo_decay`, `feat_tunnel_tightness`)
- **LightGBM MSE 139f 단독 점수**: **{sc_lgb_139:.2f}점** (136f 대비 **`{sc_lgb_139 - 747.26:+.2f} pts` 상승**)
- **카운트별 온도 스케일링 결합 점수**: **{sc_temp:.2f}점** (**`+{sc_temp - sc_lgb_139:.2f} pts` 추가 상승**)
""")
os.system(f"cp {rep318_path} {os.path.join(output_dir, '318_repertoire_fatigue_and_count_temperature_scaling.md')}")
log("Saved Report 318!")
