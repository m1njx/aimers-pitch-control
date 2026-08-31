import os
import sys
import shutil
import zipfile
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v50_dir = os.path.join(BASE_DIR, 'work', 'submit_v50')
work_v53_dir = os.path.join(BASE_DIR, 'work', 'submit_v53')
model_dir = os.path.join(work_v53_dir, 'model')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v53.zip')
pokemon_zip = '~/pipeline_src/submit_v53.zip'

os.makedirs(model_dir, exist_ok=True)

# 1. Copy base scripts and artifacts from v50
shutil.copy2(os.path.join(work_v50_dir, 'preprocessing.py'), os.path.join(work_v53_dir, 'preprocessing.py'))
shutil.copy2(os.path.join(work_v50_dir, 'trackman_features.py'), os.path.join(work_v53_dir, 'trackman_features.py'))
shutil.copy2(os.path.join(work_v50_dir, 'agent2_asof_decomp2.py'), os.path.join(work_v53_dir, 'agent2_asof_decomp2.py'))
shutil.copy2(os.path.join(work_v50_dir, 'config.py'), os.path.join(work_v53_dir, 'config.py'))
shutil.copy2(os.path.join(work_v50_dir, 'requirements.txt'), os.path.join(work_v53_dir, 'requirements.txt'))

for f in os.listdir(os.path.join(work_v50_dir, 'model')):
    if f.endswith('.cbm') or f.endswith('.txt') or f.endswith('.json') or f.endswith('.pt') or f.endswith('.pkl'):
        shutil.copy2(os.path.join(work_v50_dir, 'model', f), os.path.join(model_dir, f))

print("Base v50 25 models copied successfully.")

# 2. Train 5 Seeds Tabular Transformer & 5 Seeds Focal ResMLP
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
is_val24 = (df['season'] == 2024)
is_val23 = (df['season'] == 2023)
is_train = (df['season'] < 2024)

y = df['control_success'].values.astype(np.float32)

sys.path.insert(0, work_v53_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

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

A_all = dec.transform(df)
A_all.index = X_base.index
X_base = pd.concat([X_base, A_all], axis=1)

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

X_133 = X_base.copy()
X_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)
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

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
X_norm = X_133.copy()
for c in cat_cols:
    X_norm[c] = pd.to_numeric(X_norm[c], errors='coerce').fillna(0).astype(np.float32)
X_norm_mat = X_norm.values.astype(np.float32)
mean_all = np.nanmean(X_norm_mat[is_train], axis=0)
std_all = np.nanstd(X_norm_mat[is_train], axis=0)
std_all[std_all < 1e-6] = 1.0
X_z = np.nan_to_num((X_norm_mat - mean_all) / std_all, nan=0.0)

joblib.dump({'mean': mean_all, 'std': std_all, 'cols': list(X_norm.columns)}, os.path.join(model_dir, 'neural_suite_artifacts.pkl'))

train_idx = np.where(is_train)[0]
ds_tr = TensorDataset(torch.tensor(X_z[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32))
loader = DataLoader(ds_tr, batch_size=4096, shuffle=True)

class TabularTransformer(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.SiLU(), nn.Linear(32, 1))
    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        h = self.transformer(h).squeeze(1)
        return self.head(h).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.2):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        return loss.mean()

SEEDS = [7, 123, 2025, 31415, 8675309]

print("Training 5 Seeds Tabular Transformer...")
for s_idx, seed in enumerate(SEEDS):
    torch.manual_seed(seed)
    net_tf = TabularTransformer(X_z.shape[1], d_model=64, nhead=4, num_layers=2, dropout=0.1)
    opt = torch.optim.AdamW(net_tf.parameters(), lr=3e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    net_tf.train()
    for ep in range(6):
        for bx, by in loader:
            opt.zero_grad()
            out = net_tf(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
    torch.save(net_tf.state_dict(), os.path.join(model_dir, f'transformer_model_seed{seed}.pt'))

print("Training 5 Seeds Focal ResMLP...")
for s_idx, seed in enumerate(SEEDS):
    torch.manual_seed(seed)
    net_res = nn.Sequential(
        nn.Linear(X_z.shape[1], 128),
        nn.BatchNorm1d(128),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1)
    )
    opt_res = torch.optim.AdamW(net_res.parameters(), lr=3e-3, weight_decay=1e-4)
    crit_focal = FocalLoss(gamma=1.2)
    net_res.train()
    for ep in range(6):
        for bx, by in loader:
            opt_res.zero_grad()
            out = net_res(bx).squeeze(-1)
            loss = crit_focal(out, by)
            loss.backward()
            opt_res.step()
    torch.save(net_res.state_dict(), os.path.join(model_dir, f'focal_res_model_seed{seed}.pt'))

print("All 35 neural & tree models prepared successfully.")
