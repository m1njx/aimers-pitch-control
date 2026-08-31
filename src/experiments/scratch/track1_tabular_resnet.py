import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("=" * 70)
print("TRACK 1: Real Tabular ResNet vs SimpleMLP Evaluation on 2024 Val Fold")
print("=" * 70)
t0 = time.time()

BASE_DIR = os.path.expanduser('~/LG_data')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v33', 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

sys.path.insert(0, os.path.join(BASE_DIR, 'work', 'submit_v33'))
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

# 1. Load train.csv
df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print(f"Loaded all data: {len(df_all):,} rows")

# Preprocess features
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder().load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
prep.trackman_builder = tkm_builder
X_all = prep.transform(df_all)

base_str = ((df_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_all['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_all['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D tunneling features
v0 = X_all['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_all['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_all['tkm_rel_side_mean']
rel_height = X_all['tkm_rel_height_mean']
ivb = X_all['tkm_induced_vert_break_mean'] / 12.0
hb = X_all['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_all['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_all['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_all['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_all = dec.transform(df_all)
A_all.index = X_all.index
X_all = pd.concat([X_all, A_all], axis=1)

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

# Split Train (2019-2023) and Val (2024)
train_mask = (seasons <= 2023)
val_mask = (seasons == 2024)

X_train, y_train = X_all[train_mask].copy(), y_all[train_mask]
X_val, y_val = X_all[val_mask].copy(), y_all[val_mask]
print(f"Train set (2019-2023): {len(X_train):,} rows | Val set (2024): {len(X_val):,} rows")
assert len(X_val) == 253507, f"Expected 253507 rows for 2024 val, got {len(X_val)}"

cat_cols = [c for c in X_all.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]
num_cols = [c for c in X_all.columns if c not in cat_cols]

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

def encode_dataset(df_x):
    x_num = ((df_x[num_cols].values - mean) / std).astype(np.float32)
    x_num = np.nan_to_num(x_num, nan=0.0)
    x_cat_list = []
    for i, c in enumerate(cat_cols):
        v_map = cat_vocabs[c]
        default_idx = len(v_map)
        col_encoded = df_x[c].astype(str).map(lambda v: v_map.get(v, default_idx)).values
        x_cat_list.append(col_encoded)
    x_cat = np.column_stack(x_cat_list).astype(np.int64)
    return torch.tensor(x_num), torch.tensor(x_cat)

t_num_tr, t_cat_tr = encode_dataset(X_train)
t_num_val, t_cat_val = encode_dataset(X_val)
t_y_tr = torch.tensor(y_train)

# Architecture 1: SimpleMLP
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

# Architecture 2: Tabular ResNet
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class TabularResNet(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden_dim=128, num_blocks=2, dropout=0.15):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

# Training loop
dataset = TensorDataset(t_num_tr, t_cat_tr, t_y_tr)
loader = DataLoader(dataset, batch_size=4096, shuffle=True)

criterion = nn.BCEWithLogitsLoss()

# 1. Train SimpleMLP
print("\n--- Training SimpleMLP (Baseline) ---")
torch.manual_seed(42)
mlp_model = SimpleMLP(len(num_cols), cat_cardinalities)
optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)

mlp_model.train()
for epoch in range(5):
    ep_loss = 0.0
    for b_num, b_cat, b_y in loader:
        optimizer.zero_grad()
        out = mlp_model(b_num, b_cat)
        loss = criterion(out, b_y)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(b_y)
    print(f"Epoch {epoch+1}/5 - Loss: {ep_loss / len(X_train):.4f}")

mlp_model.eval()
with torch.no_grad():
    p_mlp_val = torch.sigmoid(mlp_model(t_num_val, t_cat_val)).numpy()
score_mlp, brier_mlp = calc_brier_skill_score(y_val, p_mlp_val)
print(f"SimpleMLP 2024 Val Skill Score: {score_mlp:.2f} (Brier: {brier_mlp:.6f})")

# 2. Train Tabular ResNet
print("\n--- Training Tabular ResNet ---")
torch.manual_seed(42)
resnet_model = TabularResNet(len(num_cols), cat_cardinalities, hidden_dim=128, num_blocks=2, dropout=0.15)
optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=1e-3, weight_decay=1e-4)

resnet_model.train()
for epoch in range(5):
    ep_loss = 0.0
    for b_num, b_cat, b_y in loader:
        optimizer.zero_grad()
        out = resnet_model(b_num, b_cat)
        loss = criterion(out, b_y)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(b_y)
    print(f"Epoch {epoch+1}/5 - Loss: {ep_loss / len(X_train):.4f}")

resnet_model.eval()
with torch.no_grad():
    p_resnet_val = torch.sigmoid(resnet_model(t_num_val, t_cat_val)).numpy()
score_resnet, brier_resnet = calc_brier_skill_score(y_val, p_resnet_val)
print(f"Tabular ResNet 2024 Val Skill Score: {score_resnet:.2f} (Brier: {brier_resnet:.6f})")

# Ensemble Comparison
p_blend = 0.5 * p_mlp_val + 0.5 * p_resnet_val
score_blend, brier_blend = calc_brier_skill_score(y_val, p_blend)
print(f"\n[SUMMARY] ResNet vs MLP Comparison on 2024 Val:")
print(f"  SimpleMLP Skill Score:       {score_mlp:.2f}")
print(f"  Tabular ResNet Skill Score:  {score_resnet:.2f}")
print(f"  MLP + ResNet 50:50 Blend:    {score_blend:.2f} (Gain vs MLP: {score_blend - score_mlp:+.2f} pts)")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
