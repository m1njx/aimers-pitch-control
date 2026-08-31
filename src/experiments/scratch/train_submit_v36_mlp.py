import os, sys, time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v36_dir = os.path.join(BASE_DIR, 'work', 'submit_v36')
model_dir = os.path.join(submit_v36_dir, 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

sys.path.insert(0, submit_v36_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

print("Loading train.csv and building 133 features...")
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

y_all = df_all['control_success'].values.astype(np.float32)
cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
num_cols = [c for c in X_all_f.columns if c not in cat_cols]

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

SEEDS = [7, 123, 2025, 31415, 8675309]

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
print("Saved new mlp_artifacts.pkl")

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
    t_s = time.time()
    print(f"Training SimpleMLP_MSE Seed {seed} ({s_idx+1}/5) on 1.47M rows...")
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
    model_save_path = os.path.join(model_dir, f'mlp_model_seed{seed}.pt')
    torch.save(m.state_dict(), model_save_path)
    print(f"  Saved {model_save_path} in {time.time() - t_s:.1f}s")

print("All 5-seed SimpleMLP models successfully trained and saved!")
