#!/usr/bin/env python3
"""
autonomous_phase5_neural_brier.py — Phase 5: Neural Networks Trained Directly on MSE / Brier Loss

1. Exp 303: PyTorch SimpleMLP & TabularResNet trained on MSELoss()
2. Exp 304: Grand Multi-Modal Optimization (GBDT + MSE Neural Networks)
"""

import os
import sys
import time

BASE_DIR = os.path.expanduser('~/LG_data')
submit_v33_dir = os.path.join(BASE_DIR, 'work', 'submit_v33')
if submit_v33_dir not in sys.path:
    sys.path.insert(0, submit_v33_dir)

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import minimize

from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

model_dir = os.path.join(submit_v33_dir, 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
report_dir = os.path.join(BASE_DIR, 'gemini_reports_for_ai')
output_dir = os.path.join(BASE_DIR, 'outputs')
cache_dir = os.path.join(BASE_DIR, 'scratch', 'cache_final')

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
log("STARTING PHASE 5: NEURAL NETWORKS TRAINED DIRECTLY ON MSE / BRIER LOSS")
log("=" * 80)

# Load data
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

y_all = df_all['control_success'].values.astype(np.float32)
seasons = df_all['season'].values

tr_m = (seasons <= 2023)
val_m = (seasons == 2024)

cat_cols = [c for c in X_base.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]
num_cols = [c for c in X_base.columns if c not in cat_cols]

mean = X_base.loc[tr_m, num_cols].mean(axis=0).values.astype(np.float32)
std = X_base.loc[tr_m, num_cols].std(axis=0).values.astype(np.float32)
std[std < 1e-6] = 1.0

cat_vocabs = {}
cat_cardinalities = []
for c in cat_cols:
    vals = X_base.loc[tr_m, c].astype(str).unique()
    vocab = {v: i for i, v in enumerate(sorted(vals))}
    cat_vocabs[c] = vocab
    cat_cardinalities.append(len(vocab) + 1)

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

t_num_tr, t_cat_tr = encode_df(X_base[tr_m])
t_num_val, t_cat_val = encode_df(X_base[val_m])
t_y_tr = torch.tensor(y_all[tr_m])

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
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

# Training with MSE Loss
log("Training SimpleMLP with MSE Loss directly...")
t0_nn = time.time()
ds = TensorDataset(t_num_tr, t_cat_tr, t_y_tr)
loader = DataLoader(ds, batch_size=4096, shuffle=True)

torch.manual_seed(42)
mlp_mse = SimpleMLP_MSE(len(num_cols), cat_cardinalities)
opt = torch.optim.AdamW(mlp_mse.parameters(), lr=2e-3, weight_decay=1e-4)
criterion_mse = nn.MSELoss()

for ep in range(5):
    mlp_mse.train()
    ep_loss = 0.0
    for b_num, b_cat, b_y in loader:
        opt.zero_grad()
        p = mlp_mse(b_num, b_cat)
        loss = criterion_mse(p, b_y)
        loss.backward()
        opt.step()
        ep_loss += loss.item() * len(b_y)
    log(f"  Epoch {ep+1}/5 - MSE Loss: {ep_loss / len(X_base[tr_m]):.6f}")

mlp_mse.eval()
with torch.no_grad():
    p_mlp_mse_val = mlp_mse(t_num_val, t_cat_val).numpy()

sc_mlp_mse, brier_mlp_mse = calc_brier_skill_score(y_all[val_m], p_mlp_mse_val)
log(f"SimpleMLP MSE Loss Score (2024 Val): {sc_mlp_mse:.2f} (Brier: {brier_mlp_mse:.6f})")

# Load cached GBDT predictions on 2024
val_2024 = np.load(os.path.join(cache_dir, 'final_val2024.npz'))
p_gbdt = np.clip(0.15 * (val_2024['p_lgb'] - 0.007) + 0.75 * (val_2024['p_cb'] - 0.008) + 0.10 * (val_2024['p_xgb'] - 0.006), 1e-6, 1 - 1e-6)

# Blend GBDT + MLP_MSE
p_grand = 0.65 * p_gbdt + 0.35 * p_mlp_mse_val
p_grand_cal = np.clip(0.5 + 1.10 * (p_grand - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
sc_grand, brier_grand = calc_brier_skill_score(y_all[val_m], p_grand_cal)

log(f"\n[GRAND MULTI-MODAL SYNTHESIS RESULTS]:")
log(f"  v33 Baseline Score: {826.86:.2f}")
log(f"  GBDT + MLP_MSE Calibrated Score: {sc_grand:.2f} pts (Gain: {sc_grand - 826.86:+.2f} pts)")
log(f"  Estimated Public LB: {1017.8593 + 0.45 * (sc_grand - 826.86):.4f} pts")

# Write Report 303
rep303_path = os.path.join(report_dir, '303_direct_mse_neural_network_optimization.md')
with open(rep303_path, 'w') as f:
    f.write(f"""# 📊 [실측 보고서] Exp 303: 직접 MSE 손실 기반 신경망 학습 및 그랜드 블렌딩 실측

- **실행 시간**: {time.time() - t0_nn:.1f}초
- **환경**: PyTorch 2.x (`venv311`)
- **모델 구성**: SimpleMLP with MSELoss() + GBDT 15-Seed + v33 Calibration

## 실측 결과
- v33 Baseline 2024 Val Score: **826.86점**
- SimpleMLP Direct MSE 단독: **{sc_mlp_mse:.2f}점**
- **GBDT + SimpleMLP Direct MSE Grand Blend**: **{sc_grand:.2f}점** (**`+{sc_grand - 826.86:.2f} pts` 상승**)
- **🎯 예상 Public LB 점수**: **`{1017.8593 + 0.45 * (sc_grand - 826.86):.4f}점`** 👑
""")
os.system(f"cp {rep303_path} {os.path.join(output_dir, '303_direct_mse_neural_network_optimization.md')}")
log("Saved Report 303!")

# Update 00_README_FOR_CLAUDE_GPT.md
with open(os.path.join(report_dir, '00_README_FOR_CLAUDE_GPT.md'), 'a') as f:
    f.write(f"""
---

## 5. 실측 브레이크스루 보고서 (Reports 301, 302, 303)

- [`301_5seed_bagged_gbdt_direct_mse.md`](file://~/LG_data/gemini_reports_for_ai/301_5seed_bagged_gbdt_direct_mse.md): 5-Seed 배깅 GBDT Direct MSE 실측 (**2024 Val: 791.75점**, `+78.81 pts` gain vs Single LGB)
- [`302_multimodal_constrained_meta_stacking.md`](file://~/LG_data/gemini_reports_for_ai/302_multimodal_constrained_meta_stacking.md): 멀티모달 제약 메타 스태킹 분석
- [`303_direct_mse_neural_network_optimization.md`](file://~/LG_data/gemini_reports_for_ai/303_direct_mse_neural_network_optimization.md): 직접 MSE 손실 신경망 학습 및 블렌딩 (**2024 Val: {sc_grand:.2f}점**, 예상 Public LB: **`{1017.8593 + 0.45 * (sc_grand - 826.86):.4f}점`**)
""")
log("README updated successfully!")
