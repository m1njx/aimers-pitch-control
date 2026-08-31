#!/usr/bin/env python3
"""
frontier3_transformer_tabular_net.py — Frontier 3: Tabular Feature Tokenizer Transformer (FT-Transformer)

Architecture:
- Feature Tokenizer: Each numeric feature is projected to d_token=16 via linear weight, each cat feature via embedding table
- 2-Layer TransformerEncoder (nhead=4, dim_feedforward=64, dropout=0.10)
- CLS token pooling -> Linear(16, 1) -> Sigmoid
- Direct MSELoss optimization
- 2024 Val Fold evaluation (N=253,507)
"""

import os
import sys
import time

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v40_dir = os.path.join(BASE_DIR, 'work', 'submit_v40')
if submit_v40_dir not in sys.path:
    sys.path.insert(0, submit_v40_dir)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

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
log("STARTING FRONTIER 3: TABULAR FEATURE TOKENIZER TRANSFORMER (FT-TRANSFORMER)")
log("=" * 80)

# Load data and 133 features
t_start = time.time()
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
seasons = df_all['season'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
num_cols = [c for c in X_all_f.columns if c not in cat_cols]

tr_2024 = (seasons <= 2023)
val_2024 = (seasons == 2024)

mean_num = X_all_f.loc[tr_2024, num_cols].mean(axis=0).values.astype(np.float32)
std_num = X_all_f.loc[tr_2024, num_cols].std(axis=0).values.astype(np.float32)
std_num[std_num < 1e-6] = 1.0

cat_vocabs = {}
cat_cardinalities = []
for c in cat_cols:
    vals = X_all_f.loc[tr_2024, c].astype(str).unique()
    vocab = {v: i for i, v in enumerate(sorted(vals))}
    cat_vocabs[c] = vocab
    cat_cardinalities.append(len(vocab) + 1)

def encode_df(df_x):
    x_num = ((df_x[num_cols].values - mean_num) / std_num).astype(np.float32)
    x_num = np.nan_to_num(x_num, nan=0.0)
    x_cat_list = []
    for i, c in enumerate(cat_cols):
        v_map = cat_vocabs[c]
        def_idx = len(v_map)
        col_enc = df_x[c].astype(str).map(lambda v: v_map.get(v, def_idx)).values
        x_cat_list.append(col_enc)
    x_cat = np.column_stack(x_cat_list).astype(np.int64)
    return torch.tensor(x_num), torch.tensor(x_cat)

t_num_tr, t_cat_tr = encode_df(X_all_f[tr_2024])
t_num_val, t_cat_val = encode_df(X_all_f[val_2024])
t_y_tr = torch.tensor(y_all[tr_2024])

class TabularTransformer(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, d_token=16, nhead=2, num_layers=2):
        super().__init__()
        self.num_dim = num_dim
        self.cat_dim = len(cat_cardinalities)
        self.d_token = d_token
        
        # Numeric tokenizers: each numeric feature gets its own W_i * x_i + b_i
        self.num_weights = nn.Parameter(torch.randn(num_dim, d_token) * 0.02)
        self.num_biases = nn.Parameter(torch.zeros(num_dim, d_token))
        
        # Categorical tokenizers
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=nhead, dim_feedforward=d_token*4, dropout=0.10, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        # Tokenize numeric: (B, N_num, d_token)
        x_num_tok = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)
        # Tokenize categorical: (B, N_cat, d_token)
        x_cat_tok = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
        # Concatenate tokens
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, x_num_tok, x_cat_tok], dim=1)
        # Pass transformer
        trans_out = self.transformer(tokens)
        cls_out = trans_out[:, 0, :]
        return self.head(cls_out).squeeze(-1)

log("Training Tabular Transformer on 1.22M train rows...")
torch.manual_seed(42)
m = TabularTransformer(len(num_cols), cat_cardinalities, d_token=16, nhead=2, num_layers=2)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

ds = TensorDataset(t_num_tr, t_cat_tr, t_y_tr)
loader = DataLoader(ds, batch_size=4096, shuffle=True)

for ep in range(3):
    m.train()
    t_ep = time.time()
    l_sum = 0.0
    for b_num, b_cat, b_y in loader:
        opt.zero_grad()
        p = m(b_num, b_cat)
        loss = criterion(p, b_y)
        loss.backward()
        opt.step()
        l_sum += loss.item()
    log(f"  Epoch {ep+1}/3 finished in {time.time() - t_ep:.1f}s (Avg Loss: {l_sum/len(loader):.5f})")

m.eval()
with torch.no_grad():
    p_trans = m(t_num_val, t_cat_val).numpy()

sc_trans, brier_trans = calc_brier_skill_score(y_all[val_2024], p_trans)
log(f"Tabular Transformer Solo 2024 Val Score: {sc_trans:.2f} pts (Brier: {brier_trans:.6f})")

# Write Report 315
rep315_path = os.path.join(report_dir, '315_tabular_transformer_cross_attention_results.md')
with open(rep315_path, 'w') as f:
    f.write(f"""# 📊 [실측 보고서] Exp 315: Tabular Transformer (FT-Transformer) 교차 주의집중 실측

- **실행 시간**: {time.time() - t_start:.1f}초
- **아키텍처**: Feature Tokenizer + 2-Layer TransformerEncoder (nhead=2, d_token=16, GELU)
- **손실 함수**: Direct MSELoss
- **검증 데이터**: 2024 Val Fold (N = 253,507)

## 실측 결과
- Tabular Transformer 단독 2024 Val Score: **{sc_trans:.2f}점**
""")
os.system(f"cp {rep315_path} {os.path.join(output_dir, '315_tabular_transformer_cross_attention_results.md')}")
log("Saved Report 315!")
