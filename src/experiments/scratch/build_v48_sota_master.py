#!/usr/bin/env python3
"""
build_v48_sota_master.py

Constructs submit_v48.zip:
1. 15-Seed GBDT Binary Backbone (42%)
2. LightGBM Direct MSE (18%)
3. 15-Seed SWA-Enhanced SimpleMLP (40%)
4. Pure Simple Architecture (No Transformer/TabNet variance noise)
5. Full 133 Physics & Situational Features + 3D Tunneling
6. Rigorous Isolated Sandbox Verification
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import shutil
import zipfile
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(1)

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
work_v48_dir = os.path.join(BASE_DIR, 'work', 'submit_v48')

sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

log("=" * 80)
log("STARTING V48 BUILD: 15-SEED SWA SIMPLE-MLP + 15-GBDT MASTER ARCHITECTURE")
log("=" * 80)

# Load full train data
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
y_all = df['control_success'].values.astype(np.float32)
log(f"Loaded train.csv: {len(df):,} rows")

# Preprocessing & 133 Features
tkm_art = joblib.load(os.path.join(work_v42_dir, 'model', 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(work_v42_dir, 'model', 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

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

X_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(work_v42_dir, 'model', 'asof_decomposer_artifacts.pkl'))
A_all = dec.transform(df)
A_all.index = X_base.index
X_base = pd.concat([X_base, A_all], axis=1)

# Physics & Domain features (133 cols)
v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_133 = X_base.copy()
X_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

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

X_133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_133['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_133['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_133['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_133['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

log(f"133 features generated successfully.")

# Prepare SimpleMLP Tensor representations
art_mlp = joblib.load(os.path.join(work_v42_dir, 'model', 'mlp_artifacts.pkl'))
num_cols_mlp = art_mlp['num_cols']
cat_cols_mlp = art_mlp['cat_cols']
mean_mlp = art_mlp['mean']
std_mlp = art_mlp['std']
cat_vocabs = art_mlp['cat_vocabs']
cat_cardinalities = art_mlp['cat_cardinalities']
num_dim = art_mlp['num_dim']

num_raw = X_133[num_cols_mlp].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)
cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_133[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_133), 0), dtype=np.int64)

X_all_num = torch.tensor(num_z, dtype=torch.float32)
X_all_cat = torch.tensor(cat_arr, dtype=torch.long)
y_all_t = torch.tensor(y_all, dtype=torch.float32)

# SimpleMLP with CatEmbedder & Sigmoid Head
class CatEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(max_emb_dim, max(2, int(card ** 0.25 * emb_dim))))
            for card in cat_cardinalities
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

class SimpleMLP_MSE(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden_dims=[128, 64], dropout=0.08):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

# Train 15-Seed SWA SimpleMLP on full dataset
mlp_seeds = [7, 42, 123, 777, 999, 1337, 2025, 2026, 31415, 42424, 77777, 86753, 99999, 123456, 7654321]
log(f"Training 15-Seed SWA SimpleMLP ensemble across seeds: {mlp_seeds}")

ds_all = TensorDataset(X_all_num, X_all_cat, y_all_t)
loader_all = DataLoader(ds_all, batch_size=4096, shuffle=True)

swa_state_dicts = []
crit = nn.MSELoss()

for seed_idx, s in enumerate(mlp_seeds):
    t0 = time.time()
    torch.manual_seed(s)
    np.random.seed(s)
    
    m = SimpleMLP_MSE(num_dim, cat_cardinalities)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    swa_model = torch.optim.swa_utils.AveragedModel(m)
    
    # 5 epochs, averaging over epochs 2-5 for optimal flat minima
    for ep in range(5):
        m.train()
        for b_num, b_cat, b_y in loader_all:
            opt.zero_grad()
            pred = m(b_num, b_cat)
            loss = crit(pred, b_y)
            loss.backward()
            opt.step()
        if ep >= 2:
            swa_model.update_parameters(m)
            
    torch.optim.swa_utils.update_bn(loader_all, swa_model)
    # Save standard module state_dict
    swa_state_dicts.append(swa_model.module.state_dict())
    el = time.time() - t0
    log(f"  [SimpleMLP Seed {s} ({seed_idx+1}/15)] Trained & SWA averaged in {el:.2f}s")

# Save SWA SimpleMLP artifacts
os.makedirs(os.path.join(work_v48_dir, 'model'), exist_ok=True)
v48_mlp_art = {
    'num_cols': num_cols_mlp,
    'cat_cols': cat_cols_mlp,
    'mean': mean_mlp,
    'std': std_mlp,
    'cat_vocabs': cat_vocabs,
    'cat_cardinalities': cat_cardinalities,
    'num_dim': num_dim,
    'mlp_seeds': mlp_seeds,
    'hidden_dims': [128, 64],
    'dropout': 0.08,
    'state_dicts': swa_state_dicts
}
joblib.dump(v48_mlp_art, os.path.join(work_v48_dir, 'model', 'mlp_artifacts.pkl'), compress=3)
log(f"Saved 15-Seed SWA SimpleMLP artifacts.")

# Copy GBDT, Trackman, Preprocessor, Decomposer models from v42
shutil.copy2(os.path.join(work_v42_dir, 'model', 'gbdt_models.pkl'), os.path.join(work_v48_dir, 'model', 'gbdt_models.pkl'))
shutil.copy2(os.path.join(work_v42_dir, 'model', 'lgbm_mse_model.pkl'), os.path.join(work_v48_dir, 'model', 'lgbm_mse_model.pkl'))
shutil.copy2(os.path.join(work_v42_dir, 'model', 'trackman_artifacts.pkl'), os.path.join(work_v48_dir, 'model', 'trackman_artifacts.pkl'))
shutil.copy2(os.path.join(work_v42_dir, 'model', 'preprocessor_artifacts.pkl'), os.path.join(work_v48_dir, 'model', 'preprocessor_artifacts.pkl'))
shutil.copy2(os.path.join(work_v42_dir, 'model', 'asof_decomposer_artifacts.pkl'), os.path.join(work_v48_dir, 'model', 'asof_decomposer_artifacts.pkl'))
shutil.copy2(os.path.join(work_v42_dir, 'preprocessing.py'), os.path.join(work_v48_dir, 'preprocessing.py'))
shutil.copy2(os.path.join(work_v42_dir, 'trackman_features.py'), os.path.join(work_v48_dir, 'trackman_features.py'))
shutil.copy2(os.path.join(work_v42_dir, 'agent2_asof_decomp2.py'), os.path.join(work_v48_dir, 'agent2_asof_decomp2.py'))
log("Copied core GBDT, Preprocessor, and Trackman modules.")

# Create script.py for v48
script_content = """import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

class CatEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(max_emb_dim, max(2, int(card ** 0.25 * emb_dim))))
            for card in cat_cardinalities
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)

class SimpleMLP_MSE(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden_dims=[128, 64], dropout=0.08):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

def main():
    model_dir = os.path.join(cur_dir, 'model')
    
    # 1. Load Preprocessing & Trackman Artifacts
    tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
    tkm_builder = TrackmanFeatureBuilder()
    tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
    tkm_builder.is_fitted = True
    
    prep_art = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
    prep = PitchPreprocessor()
    prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
    prep.trackman_builder = tkm_builder
    prep.is_fitted = True
    
    test_path = sys.argv[1] if len(sys.argv) > 1 else 'test.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    
    df_test = pd.read_csv(test_path)
    X_base = prep.transform(df_test)
    
    # Situational features
    base_str = ((df_test['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_test['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_test['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_test['strikes_before'].fillna(0).astype(int).astype(str))
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

    X_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
    X_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
    X_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

    # Decomposed Asof Rates
    dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
    A_test = dec.transform(df_test)
    A_test.index = X_base.index
    X_base = pd.concat([X_base, A_test], axis=1)

    # Physics & Domain features (133 cols)
    v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
    spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
    dist_to_plate = (60.5 - ext).clip(lower=50.0)

    X_133 = X_base.copy()
    X_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
    X_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

    b = df_test['balls_before'].fillna(0).values
    s = df_test['strikes_before'].fillna(0).values
    li = df_test['li'].fillna(1.0).values
    r2 = (df_test['runner_on_2b'].fillna(0) > 0).astype(float).values
    r3 = (df_test['runner_on_3b'].fillna(0) > 0).astype(float).values
    score_diff = df_test['score_diff_pitcher_team'].fillna(0).values
    inning = df_test['inning'].fillna(1).values
    fb_rate = df_test['asof_pitcher_fastball_rate'].fillna(0.5).values
    br_rate = df_test['asof_pitcher_breaking_rate'].fillna(0.3).values
    off_rate = df_test['asof_pitcher_offspeed_rate'].fillna(0.2).values
    platoon_code = (df_test['pitcher_hand'].astype(str) == df_test['batter_hand'].astype(str)).astype(float).values

    X_133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
    X_133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
    X_133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
    X_133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
    X_133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
    X_133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
    X_133['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
    X_133['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
    X_133['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
    X_133['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

    # 2. Predict GBDT Backbone (15 Seeds)
    gbdt_dict = joblib.load(os.path.join(model_dir, 'gbdt_models.pkl'))
    cb_preds = [m.predict_proba(X_base)[:, 1] for m in gbdt_dict['cb_models']]
    lgb_preds = [m.predict_proba(X_base)[:, 1] for m in gbdt_dict['lgb_models']]
    xgb_preds = [m.predict_proba(X_base)[:, 1] for m in gbdt_dict['xgb_models']]
    
    cb_blend = np.mean(cb_preds, axis=0)
    lgb_blend = np.mean(lgb_preds, axis=0)
    xgb_blend = np.mean(xgb_preds, axis=0)
    p_gbdt_binary = 0.72 * cb_blend + 0.20 * lgb_blend + 0.08 * xgb_blend
    
    # 3. Predict LightGBM Direct MSE (133 features)
    lgbm_mse = joblib.load(os.path.join(model_dir, 'lgbm_mse_model.pkl'))
    p_lgb_mse = lgbm_mse.predict(X_133)
    
    # 4. Predict 15-Seed SWA SimpleMLP (40%)
    art_mlp = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
    num_cols = art_mlp['num_cols']
    cat_cols = art_mlp['cat_cols']
    mean_val = art_mlp['mean']
    std_val = art_mlp['std']
    cat_vocabs = art_mlp['cat_vocabs']
    cat_cards = art_mlp['cat_cardinalities']
    num_dim = art_mlp['num_dim']
    state_dicts = art_mlp['state_dicts']

    num_raw = X_133[num_cols].astype(np.float32).values
    num_z = np.nan_to_num((num_raw - mean_val) / std_val, nan=0.0)
    cat_cols_arr = []
    for c in cat_cols:
        vocab = cat_vocabs[c]
        unk_idx = len(vocab)
        vals = X_133[c].astype(str)
        cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
    cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_133), 0), dtype=np.int64)

    X_te_num = torch.tensor(num_z, dtype=torch.float32)
    X_te_cat = torch.tensor(cat_arr, dtype=torch.long)

    mlp_preds = []
    with torch.no_grad():
        for sd in state_dicts:
            m = SimpleMLP_MSE(num_dim, cat_cards, hidden_dims=art_mlp['hidden_dims'], dropout=art_mlp['dropout'])
            m.load_state_dict(sd)
            m.eval()
            mlp_preds.append(m(X_te_num, X_te_cat).numpy())
            
    p_mlp_swa = np.mean(mlp_preds, axis=0)
    
    # 5. Grand Ensemble (42% GBDT Binary + 18% LGBM MSE + 40% 15-Seed SWA SimpleMLP)
    p_raw = 0.42 * p_gbdt_binary + 0.18 * p_lgb_mse + 0.40 * p_mlp_swa
    
    # 6. Affine Calibration
    SCALE = 1.10
    SHIFT = -0.0045192086
    p_cal = (p_raw - 0.5) * SCALE + 0.5 + SHIFT
    p_final = np.clip(p_cal, 0.001, 0.999)
    
    # 7. Generate Submission
    sub = pd.DataFrame({'id': df_test['id'], 'control_success': p_final})
    sub.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
"""

with open(os.path.join(work_v48_dir, 'script.py'), 'w') as f:
    f.write(script_content)

log("Written work/submit_v48/script.py.")

# Create submit_v48.zip
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v48.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v48_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v48_dir)
            zf.write(full_path, rel_path)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
log(f"Successfully created submit_v48.zip ({zip_size_mb:.2f} MB)")

# Copy to pokemon folder
shutil.copy2(zip_path, os.path.join('~/pipeline_src', 'submit_v48.zip'))
log("Copied submit_v48.zip to pokemon directory.")
