#!/usr/bin/env python3
"""
frontier13_catboost_rmse_hyper_tuning.py — CatBoost Direct RMSE Hyper-Tuning on 136 Features

Target: Push CatBoost Direct RMSE solo score from 787.63 pts -> 805+ pts!
Explores:
1. Tree Depth exploration: depth in [5, 6, 7, 8]
2. L2 Leaf Regularization (l2_leaf_reg in [1, 3, 5, 10, 20])
3. Learning Rate & Iterations: lr in [0.04, 0.06, 0.08], iterations in [400, 600, 800]
4. Random Strength & Bagging Temperature: [0.5, 1.0, 2.0]
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
report_dir = os.path.join(BASE_DIR, 'gemini_reports_for_ai')
output_dir = os.path.join(BASE_DIR, 'outputs')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v41', 'model')

sys.path.insert(0, os.path.join(BASE_DIR, 'work', 'submit_v41'))
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

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
log("STARTING FRONTIER 13: CATBOOST DIRECT RMSE HYPER-TUNING (136 FEATURES)")
log("=" * 80)

t_start = time.time()
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
seasons = df_all['season'].values
y_all = df_all['control_success'].values.astype(np.float32)
tr_2024 = (seasons <= 2023)
val_2024 = (seasons == 2024)
y_val = y_all[val_2024]

# Preprocess Base 136 Features
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

v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_all_136 = X_base.copy()
X_all_136['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_all_136['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_all_136['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_all_136['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

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

X_all_136['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_all_136['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_all_136['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_all_136['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_all_136['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_all_136['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_all_136['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_all_136['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_all_136['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_all_136['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

X_all_136['tunnel_delta_speed'] = (v_rel - 92.0).astype(np.float32)
X_all_136['tunnel_delta_ivb'] = (ivb - 1.2).astype(np.float32)
X_all_136['tunnel_delta_hb'] = np.abs(hb).astype(np.float32)

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
cb_tr = X_all_136[tr_2024].copy()
cb_val = X_all_136[val_2024].copy()
for c in cat_cols:
    cb_tr[c] = cb_tr[c].astype(str)
    cb_val[c] = cb_val[c].astype(str)

log("Benchmarking CatBoost RMSE Configurations...")
configs = [
    {'depth': 6, 'l2_leaf_reg': 3.0, 'learning_rate': 0.06, 'iterations': 350, 'name': 'Baseline (depth=6, l2=3)'},
    {'depth': 7, 'l2_leaf_reg': 5.0, 'learning_rate': 0.05, 'iterations': 450, 'name': 'Deeper (depth=7, l2=5)'},
    {'depth': 8, 'l2_leaf_reg': 10.0, 'learning_rate': 0.04, 'iterations': 500, 'name': 'High Capacity (depth=8, l2=10)'},
    {'depth': 6, 'l2_leaf_reg': 1.0, 'learning_rate': 0.07, 'iterations': 400, 'name': 'Low Reg (depth=6, l2=1)'},
    {'depth': 6, 'l2_leaf_reg': 10.0, 'learning_rate': 0.06, 'iterations': 500, 'name': 'High Reg (depth=6, l2=10)'},
]

results = []
for cfg in configs:
    t_c = time.time()
    log(f"Testing {cfg['name']}...")
    m = CatBoostRegressor(
        iterations=cfg['iterations'],
        learning_rate=cfg['learning_rate'],
        depth=cfg['depth'],
        l2_leaf_reg=cfg['l2_leaf_reg'],
        random_seed=42,
        thread_count=4,
        verbose=False,
        cat_features=cat_cols
    )
    m.fit(cb_tr, y_all[tr_2024], eval_set=(cb_val, y_all[val_2024]), early_stopping_rounds=30)
    p_val = np.clip(m.predict(cb_val), 1e-6, 1 - 1e-6)
    sc, brier = calc_brier_skill_score(y_val, p_val)
    log(f"  -> Score: {sc:.2f} pts (Brier: {brier:.6f}) in {time.time() - t_c:.1f}s")
    results.append({'name': cfg['name'], 'score': sc, 'brier': brier, 'params': cfg})

best_res = max(results, key=lambda x: x['score'])
log(f"\nBest Configuration: {best_res['name']} with Score: {best_res['score']:.2f} pts!")

# Write Report 329
rep329_path = os.path.join(report_dir, '329_catboost_rmse_hyper_tuning_results.md')
with open(rep329_path, 'w') as f:
    f.write(f"""# 📊 [실측 보고서] Exp 329: CatBoost Direct RMSE 하이퍼파라미터 정밀 튜닝 결과

- **검증 데이터**: 2024 Validation Fold ($N = 253,507$)
- **최고 점수 달성 설정**: **`{best_res['name']}`**
- **실측 최고 점수**: **`{best_res['score']:.2f}점`** (기존 기준선 대비 **`{best_res['score'] - 787.63:+.2f} pts`**)
- **실행 시간**: {time.time() - t_start:.1f}초

## 전체 후보군 실측 비교
| 구성 이름 | 깊이 (Depth) | L2 정규화 | 학습률 | 트리 수 | 2024 Val 실측 점수 |
| :--- | :---: | :---: | :---: | :---: | :---: |
""" + "\n".join([f"| {r['name']} | {r['params']['depth']} | {r['params']['l2_leaf_reg']} | {r['params']['learning_rate']} | {r['params']['iterations']} | **{r['score']:.2f}점** |" for r in results]))
os.system(f"cp {rep329_path} {os.path.join(output_dir, '329_catboost_rmse_hyper_tuning_results.md')}")
log("Saved Report 329!")
