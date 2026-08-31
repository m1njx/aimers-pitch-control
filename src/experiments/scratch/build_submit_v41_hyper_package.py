#!/usr/bin/env python3
"""
build_submit_v41_hyper_package.py — Build submit_v41.zip (1060~1100+ Target Hyper-Ensemble)

Packages:
1. 136 Features
2. 15-Seed GBDT Binary
3. 5-Seed LightGBM Direct MSE
4. 5-Seed CatBoost Direct RMSE
5. 5-Seed SimpleMLP Direct MSE
6. Count-Conditional Calibration + Affine Post-Processing
7. Local rehearsal and zip packaging
"""

import os
import sys
import time
import shutil
import zipfile
import subprocess

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v40_dir = os.path.join(BASE_DIR, 'work', 'submit_v40')
submit_v41_dir = os.path.join(BASE_DIR, 'work', 'submit_v41')
model_dir = os.path.join(submit_v41_dir, 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

if submit_v40_dir not in sys.path:
    sys.path.insert(0, submit_v40_dir)

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

log("=" * 80)
log("BUILDING SUBMIT_V41 HYPER-ENSEMBLE PACKAGE (1060~1100+ TARGET)")
log("=" * 80)

# Step 1: Copy submit_v40 to submit_v41
if os.path.exists(submit_v41_dir):
    shutil.rmtree(submit_v41_dir)
shutil.copytree(submit_v40_dir, submit_v41_dir)
log("Copied submit_v40 to work/submit_v41/")

# Step 2: Load full dataset and build 136 features
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
log(f"Loaded train.csv: {len(df_all):,} rows")

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
log(f"All 136 Features Engineered: {X_all_f.shape[1]} columns")

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
num_cols = [c for c in X_all_f.columns if c not in cat_cols]

# Step 3: Train 5-Seed CatBoost Direct RMSE on full data
SEEDS = [7, 123, 2025, 31415, 8675309]
cb_all = X_all_f.copy()
for c in cat_cols:
    cb_all[c] = cb_all[c].astype(str)

for s in SEEDS:
    log(f"Training CatBoost Direct RMSE Seed {s} on full 1.47M rows...")
    m_cb_rmse = CatBoostRegressor(iterations=350, learning_rate=0.06, depth=6, random_seed=s, thread_count=4, verbose=False, cat_features=cat_cols)
    m_cb_rmse.fit(cb_all, y_all)
    m_cb_rmse.save_model(os.path.join(model_dir, f'catboost_rmse_model_seed{s}.cbm'))

log("Saved all 5 CatBoost RMSE models!")

# Step 4: Train 5-Seed LightGBM Direct MSE on full data
dtr_lgb = lgb.Dataset(X_all_f, label=y_all)
for s in SEEDS:
    log(f"Training LightGBM Direct MSE Seed {s} on full 1.47M rows...")
    m_lgb_mse = lgb.train({
        'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
        'num_leaves': 63, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'random_state': s, 'n_jobs': 4, 'verbose': -1
    }, dtr_lgb, num_boost_round=350)
    m_lgb_mse.save_model(os.path.join(model_dir, f'lgbm_mse_model_seed{s}.txt'))

log("Saved all 5 LightGBM MSE models!")

# Step 5: Train 5-Seed SimpleMLP Direct MSE on 136 features
mean = X_all_f[num_cols].mean(axis=0).values.astype(np.float32)
std = X_all_f[num_cols].std(axis=0).values.astype(np.float32)
std[std < 1e-6] = 1.0

cat_vocabs = {}
cat_cardinalities = []
for c in cat_cols:
    vals = X_all_f[c].astype(str).unique()
    vocab = {v: i for i, v in enumerate(sorted(vals))}
    cat_vocabs[c] = vocab
    cat_cardinalities.append(len(vocab) + 1)

mlp_artifacts = {
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'mean': mean,
    'std': std,
    'cat_vocabs': cat_vocabs,
    'cat_cardinalities': cat_cardinalities,
    'num_dim': len(num_cols),
    'mlp_shifts': {s: 0.0 for s in SEEDS}
}
joblib.dump(mlp_artifacts, os.path.join(model_dir, 'mlp_artifacts.pkl'))

def encode_df(df_x):
    x_num = ((df_x[num_cols].values - mean) / std).astype(np.float32)
    x_num = np.nan_to_num(x_num, nan=0.0)
    x_cat_list = []
    for i, c in enumerate(cat_cols):
        v_map = cat_vocabs[c]
        def_idx = len(v_map)
        col_enc = df_x[c].astype(str).map(lambda v: v_map.get(v, def_idx)).values
        x_cat_list.append(col_enc)
    x_cat = np.column_stack(x_cat_list).astype(np.int64)
    return torch.tensor(x_num), torch.tensor(x_cat)

t_num, t_cat = encode_df(X_all_f)
t_y = torch.tensor(y_all)

class CatEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(max_emb_dim, max(2, int(card ** 0.25 * emb_dim))))
            for card in cat_cardinalities
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        if len(self.embs) == 0:
            return torch.zeros(x_cat.shape[0], 0, device=x_cat.device)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

class SimpleMLP_MSE(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

ds = TensorDataset(t_num, t_cat, t_y)
loader = DataLoader(ds, batch_size=4096, shuffle=True)
criterion_mse = nn.MSELoss()

for s_idx, seed in enumerate(SEEDS):
    log(f"Training SimpleMLP_MSE Seed {seed} ({s_idx+1}/5) on 136 features...")
    torch.manual_seed(seed)
    m = SimpleMLP_MSE(len(num_cols), cat_cardinalities, hidden=(128, 64), dropout=0.12)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(5):
        m.train()
        for b_num, b_cat, b_y in loader:
            opt.zero_grad()
            p = m(b_num, b_cat)
            loss = criterion_mse(p, b_y)
            loss.backward()
            opt.step()
    torch.save(m.state_dict(), os.path.join(model_dir, f'mlp_model_seed{seed}.pt'))

log("Saved all 5 SimpleMLP models!")

# Step 6: Write submit_v41 script.py
script_py_v41 = """import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
import torch
import torch.nn as nn
from agent2_asof_decomp2 import AsofDecomposer2

t0 = time.time()
print("Starting DACON 1100+ Hyper SOTA Inference Pipeline (v41 Grand 30-Model Hyper Ensemble)...")

DEVICE = torch.device('cpu')
SEEDS = [7, 123, 2025, 31415, 8675309]

# Grand Quad-Blend Weights
W_GBDT_BIN = 0.35
W_LGB_MSE = 0.15
W_CB_RMSE = 0.20
W_MLP_MSE = 0.30

W_LGB_BIN, W_CB_BIN, W_XGB_BIN = 0.20, 0.72, 0.08
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

class CatEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(max_emb_dim, max(2, int(card ** 0.25 * emb_dim))))
            for card in cat_cardinalities
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        if len(self.embs) == 0:
            return torch.zeros(x_cat.shape[0], 0, device=x_cat.device)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

class SimpleMLP_MSE(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

data_dir = os.path.join(SCRIPT_DIR, "data")
if not os.path.exists(data_dir):
    data_dir = "data"
output_dir = os.path.join(SCRIPT_DIR, "output")
if not os.path.exists(output_dir):
    output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(SCRIPT_DIR, "model")

test_path = os.path.join(data_dir, "test.csv")
if not os.path.exists(test_path):
    test_path = "data/test.csv"

print(f"Loading test data from: {test_path}")
df_test = pd.read_csv(test_path)
print(f"Test data shape: {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder

tkm_builder = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
prep_obj = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
if isinstance(prep_obj, PitchPreprocessor):
    prep = prep_obj
    if hasattr(tkm_builder, 'transform'):
        prep.trackman_builder = tkm_builder
else:
    prep = PitchPreprocessor()
    prep.artifacts = prep_obj
    if hasattr(tkm_builder, 'transform'):
        prep.trackman_builder = tkm_builder
    prep.is_fitted = True

X_test_base = prep.transform(df_test)

base_str = ((df_test['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_test['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_test_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D Tunneling Features
v0 = X_test_base['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_test_base['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_test_base['tkm_rel_side_mean']
rel_height = X_test_base['tkm_rel_height_mean']
ivb = X_test_base['tkm_induced_vert_break_mean'] / 12.0
hb = X_test_base['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight

d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_test_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_test_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_test_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_test = dec.transform(df_test)
A_test.index = X_test_base.index
X_test_base = pd.concat([X_test_base, A_test], axis=1)

# 4 Sabermetric Physics Features
v_rel = X_test_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_test_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_test_136 = X_test_base.copy()
X_test_136['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_test_136['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_test_136['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_test_136['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

# 10 Domain Interaction Features
b = df_test['balls_before'].fillna(0).values
s = df_test['strikes_before'].fillna(0).values
li = df_test['li'].fillna(1.0).values
r1 = (df_test['runner_on_1b'].fillna(0) > 0).astype(float).values
r2 = (df_test['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_test['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_test['score_diff_pitcher_team'].fillna(0).values
inning = df_test['inning'].fillna(1).values

fb_rate = df_test['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_test['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df_test['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df_test['pitcher_hand'].astype(str) == df_test['batter_hand'].astype(str)).astype(float).values

X_test_136['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_test_136['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_test_136['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_test_136['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_test_136['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_test_136['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_test_136['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_test_136['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_test_136['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_test_136['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

# 3 Pitch Tunneling Differentials
X_test_136['tunnel_delta_speed'] = (v_rel - 92.0).astype(np.float32)
X_test_136['tunnel_delta_ivb'] = (ivb - 1.2).astype(np.float32)
X_test_136['tunnel_delta_hb'] = np.abs(hb).astype(np.float32)

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']

# GBDT Binary Inference (15 seeds on 119 features)
X_test_cb_bin = X_test_base.copy()
for c in cat_cols:
    X_test_cb_bin[c] = pd.to_numeric(X_test_cb_bin[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_test_cb_bin.columns if col not in cat_cols]:
    X_test_cb_bin[c] = pd.to_numeric(X_test_cb_bin[c], errors='coerce').fillna(0.0).astype(np.float32)

X_test_xgb = X_test_base.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_test_xgb[c] = X_test_xgb[c].astype(np.float32)
    else:
        X_test_xgb[c] = (X_test_xgb[c] - 1).astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

# CatBoost RMSE format
X_test_cb_rmse = X_test_136.copy()
for c in cat_cols:
    X_test_cb_rmse[c] = X_test_cb_rmse[c].astype(str)

print("Predicting with Grand 30-Model Hyper Ensemble...")
p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))
p_lgb_mse_sum = np.zeros(len(df_test))
p_cb_rmse_sum = np.zeros(len(df_test))

X_test_136_mat = X_test_136.values.astype(np.float32)

for seed in SEEDS:
    # LGB Binary
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test_base)
    # CB Binary
    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_test_cb_bin)[:, 1]
    # XGB Binary
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]
    # LGB MSE (136 features)
    m_lgb_mse = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_mse_model_seed{seed}.txt'))
    p_lgb_mse_sum += m_lgb_mse.predict(X_test_136_mat)
    # CB RMSE (136 features)
    m_cb_rmse = CatBoostRegressor()
    m_cb_rmse.load_model(os.path.join(model_dir, f'catboost_rmse_model_seed{seed}.cbm'))
    p_cb_rmse_sum += m_cb_rmse.predict(X_test_cb_rmse)

n_seeds = len(SEEDS)
p_lgb_bin = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb_bin = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb_bin = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)
p_gbdt_bin = np.clip(W_LGB_BIN * p_lgb_bin + W_CB_BIN * p_cb_bin + W_XGB_BIN * p_xgb_bin, 1e-6, 1 - 1e-6)

p_gbdt_mse = np.clip(p_lgb_mse_sum / n_seeds, 1e-6, 1 - 1e-6)
p_cb_rmse = np.clip(p_cb_rmse_sum / n_seeds, 1e-6, 1 - 1e-6)

# SimpleMLP MSE Inference (5 seeds on 136 features)
art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
num_cols_mlp, cat_cols_mlp = art['num_cols'], art['cat_cols']
mean_mlp, std_mlp = art['mean'], art['std']
cat_vocabs = art['cat_vocabs']
cat_cardinalities = art['cat_cardinalities']
num_dim = art['num_dim']

num_raw = X_test_136[num_cols_mlp].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)
num_t = torch.tensor(num_z, dtype=torch.float32).to(DEVICE)

cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_test_136[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_test_136), 0), dtype=np.int64)
cat_t = torch.tensor(cat_arr, dtype=torch.long).to(DEVICE)

p_mlp_sum = np.zeros(len(df_test), dtype=np.float64)

for seed in SEEDS:
    mlp_net = SimpleMLP_MSE(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12).to(DEVICE)
    mlp_net.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt'), map_location=DEVICE))
    mlp_net.eval()
    with torch.no_grad():
        p_mlp_sum += mlp_net(num_t, cat_t).cpu().numpy()

p_mlp_mse = p_mlp_sum / len(SEEDS)

# Grand Quad-Blend
p_raw = W_GBDT_BIN * p_gbdt_bin + W_LGB_MSE * p_gbdt_mse + W_CB_RMSE * p_cb_rmse + W_MLP_MSE * p_mlp_mse

# Count-Conditional Micro-Calibration
count_shifts = joblib.load(os.path.join(model_dir, 'count_shifts_artifact.pkl'))
counts_test = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_test['strikes_before'].fillna(0).astype(int).astype(str)).values

p_cond = p_raw.copy()
for cc, s_val in count_shifts.items():
    p_cond[counts_test == cc] += s_val

# Final Affine Calibration
CALIBRATION_SCALE = 1.10
CALIBRATION_SHIFT = -0.0045192086
p_calibrated = np.clip(0.5 + CALIBRATION_SCALE * (p_cond - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)

# Output submission
df_sub = pd.DataFrame({
    'row_id': df_test['row_id'],
    'control_success': p_calibrated
})

out_path = os.path.join(output_dir, 'submission.csv')
df_sub.to_csv(out_path, index=False)
print(f"Submission successfully saved to: {out_path}")
print(f"Summary stats: Mean={p_calibrated.mean():.6f}, Min={p_calibrated.min():.6f}, Max={p_calibrated.max():.6f}")
print(f"Total pipeline elapsed time: {time.time() - t0:.2f}s")
"""

with open(os.path.join(submit_v41_dir, 'script.py'), 'w') as f:
    f.write(script_py_v41)
log("Updated work/submit_v41/script.py")

# Step 7: Local Rehearsal
log("Running local test execution rehearsal on test.csv...")
mkdir_p = os.path.join(submit_v41_dir, 'data')
os.makedirs(mkdir_p, exist_ok=True)
shutil.copy(os.path.join(data_dir, 'test.csv'), os.path.join(mkdir_p, 'test.csv'))

res = subprocess.run([sys.executable, os.path.join(submit_v41_dir, 'script.py')], cwd=submit_v41_dir, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    raise RuntimeError("Submission rehearsal failed!")
log("Rehearsal passed 100%!")

# Step 8: Package submit_v41.zip
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v41.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(submit_v41_dir):
        if '__pycache__' in root or os.path.basename(root) in ['data', 'output', 'catboost_info']:
            continue
        for file in files:
            if file.startswith('.') or file.endswith('.pyc') or file == 'submission.csv' or 'factor_scores' in file or 'gt_features' in file:
                continue
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, submit_v41_dir)
            zf.write(full_p, rel_p)

log(f"Successfully packaged submit_v41.zip! Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
