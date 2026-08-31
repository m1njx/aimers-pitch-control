import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

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

torch.set_num_threads(4)

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v48_dir = os.path.join(BASE_DIR, 'work', 'submit_v48')
model_dir = os.path.join(work_v48_dir, 'model')

sys.path.insert(0, work_v48_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

t0 = time.time()
print("Starting In-Process 15-Seed SWA Training with PyTorch 4-thread acceleration...")

df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
y_all = df['control_success'].values.astype(np.float32)

tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
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

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
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

art_mlp = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
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

SEEDS_15 = [7, 42, 123, 777, 999, 1337, 2025, 2026, 31415, 42424, 77777, 86753, 99999, 123456, 7654321]
ds_all = TensorDataset(X_all_num, X_all_cat, y_all_t)
loader = DataLoader(ds_all, batch_size=8192, shuffle=True, num_workers=0)
crit = nn.MSELoss()

# Remove old mlp models
for f in os.listdir(model_dir):
    if f.startswith('mlp_model_seed'):
        os.remove(os.path.join(model_dir, f))

for idx, s in enumerate(SEEDS_15):
    s_t0 = time.time()
    torch.manual_seed(s)
    np.random.seed(s)
    
    m = SimpleMLP_MSE(num_dim, cat_cardinalities)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    swa_m = torch.optim.swa_utils.AveragedModel(m)
    
    for ep in range(5):
        m.train()
        for b_n, b_c, b_y in loader:
            opt.zero_grad()
            pred = m(b_n, b_c)
            loss = crit(pred, b_y)
            loss.backward()
            opt.step()
        if ep >= 2:
            swa_m.update_parameters(m)
            
    torch.optim.swa_utils.update_bn(loader, swa_m)
    save_path = os.path.join(model_dir, f'mlp_model_seed{s}.pt')
    torch.save(swa_m.module.state_dict(), save_path)
    print(f"  [Seed {s:8d} ({idx+1}/15)] SWA trained & saved to {os.path.basename(save_path)} ({time.time() - s_t0:.1f}s)")

# Update script.py in submit_v48 with SEEDS_15 and SWA configuration
script_path = os.path.join(work_v48_dir, 'script.py')
with open(script_path, 'r') as f:
    code = f.read()

# Cleanly configure SEEDS_GBDT and SEEDS_MLP in script.py
code = code.replace("SEEDS = [7, 123, 2025, 31415, 8675309]", "SEEDS_GBDT = [7, 123, 2025, 31415, 8675309]\nSEEDS_MLP = [7, 42, 123, 777, 999, 1337, 2025, 2026, 31415, 42424, 77777, 86753, 99999, 123456, 7654321]")
code = code.replace("for seed in SEEDS:", "for seed in SEEDS_GBDT:")
code = code.replace("n_seeds = len(SEEDS)", "n_seeds = len(SEEDS_GBDT)")
code = code.replace("for seed in SEEDS:\n    mlp_net = SimpleMLP_MSE", "for seed in SEEDS_MLP:\n    mlp_net = SimpleMLP_MSE")
code = code.replace("p_mlp_mse = p_mlp_sum / len(SEEDS)", "p_mlp_mse = p_mlp_sum / len(SEEDS_MLP)")

with open(script_path, 'w') as f:
    f.write(code)

print("Updated work/submit_v48/script.py with 15 SWA seeds.")

# Zip submit_v48.zip
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v48.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v48_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v48_dir)
            zf.write(full_path, rel_path)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Created submit_v48.zip: {zip_size_mb:.2f} MB in {time.time() - t0:.2f}s total!")

# Copy to pokemon
pokemon_zip = '~/pipeline_src/submit_v48.zip'
shutil.copy2(zip_path, pokemon_zip)
print(f"Copied submit_v48.zip to pokemon directory.")
