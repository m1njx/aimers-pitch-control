import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

print("Starting SimpleMLP 5-Seed Training ...")
t0 = time.time()

model_dir = '~/LG_data/work/submit_v16/model'
data_dir = '~/LG_data/open/data'

sys.path.insert(0, '~/LG_data/work/submit_v16')
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

# 1. Load train.csv
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print(f"Loaded train data: {len(df_train):,} rows")

# 2. Transform with fitted prep and dec
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder().load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
prep.trackman_builder = tkm_builder
X_train = prep.transform(df_train)

base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_train['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D tunneling features
v0 = X_train['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_train['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_train['tkm_rel_side_mean']
rel_height = X_train['tkm_rel_height_mean']
ivb = X_train['tkm_induced_vert_break_mean'] / 12.0
hb = X_train['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_train['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_train['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_train['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_train = dec.transform(df_train)
A_train.index = X_train.index
X_train = pd.concat([X_train, A_train], axis=1)

y_train = df_train['control_success'].values.astype(np.float32)

cat_cols = [c for c in X_train.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]
num_cols = [c for c in X_train.columns if c not in cat_cols]

print(f"X_train total columns: {len(X_train.columns)} ({len(num_cols)} num, {len(cat_cols)} cat)")

# Build MLP artifacts
mean = X_train[num_cols].mean(axis=0).values.astype(np.float32)
std = X_train[num_cols].std(axis=0).values.astype(np.float32)
std[std < 1e-6] = 1.0

cat_vocabs = {}
cat_cardinalities = []
for c in cat_cols:
    vals = X_train[c].astype(str).unique()
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
print("Saved mlp_artifacts.pkl")

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

class SimpleMLP(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

num_z = np.nan_to_num((X_train[num_cols].values.astype(np.float32) - mean) / std, nan=0.0)
cat_cols_arr = [X_train[c].astype(str).map(cat_vocabs[c]).fillna(len(cat_vocabs[c])).values for c in cat_cols]
cat_arr = np.stack(cat_cols_arr, axis=1).astype(np.int64)

X_num_t = torch.tensor(num_z, dtype=torch.float32)
X_cat_t = torch.tensor(cat_arr, dtype=torch.int64)
y_t = torch.tensor(y_train, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_num_t, X_cat_t, y_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)

for i, seed in enumerate(SEEDS, 1):
    print(f"[{i}/5] Training SimpleMLP seed {seed} ...")
    torch.manual_seed(seed)
    model = SimpleMLP(len(num_cols), cat_cardinalities, hidden=(128, 64), dropout=0.15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(5):
        for bx_num, bx_cat, by in loader:
            opt.zero_grad()
            out = model(bx_num, bx_cat)
            loss = crit(out, by)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), os.path.join(model_dir, f'mlp_model_seed{seed}.pt'))
    print(f"Saved mlp_model_seed{seed}.pt ({time.time()-t0:.1f}s elapsed)")

print(f"SimpleMLP training completed successfully in {time.time()-t0:.1f} seconds!")
