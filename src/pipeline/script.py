import sys
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
from catboost import CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from agent2_asof_decomp2 import AsofDecomposer2

t0 = time.time()
print("Starting DACON 1150+ Master SOTA Inference Pipeline (v42 Neural Super-Ensemble)...")

DEVICE = torch.device('cpu')
SEEDS = [7, 123, 2025, 31415, 8675309]

# Winning Neural-GBDT Super-Blend Weights (v42 SOTA: Neural 40% + GBDT 40% + MSE 20%)
W_GBDT_BIN = 0.40
W_MLP_MSE = 0.40
W_LGB_MSE = 0.20

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

tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_obj = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
if isinstance(prep_obj, PitchPreprocessor):
    prep = prep_obj
    prep.trackman_builder = tkm_builder
else:
    prep = PitchPreprocessor()
    prep.artifacts = prep_obj if isinstance(prep_obj, dict) else prep_obj.artifacts
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

X_test_133 = X_test_base.copy()
X_test_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_test_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_test_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_test_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

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

X_test_133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_test_133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_test_133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_test_133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_test_133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_test_133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_test_133['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_test_133['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_test_133['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_test_133['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']

# Pre-cast feature matrices for instant C++ execution
X_test_cb = X_test_base.copy()
for c in cat_cols:
    X_test_cb[c] = pd.to_numeric(X_test_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_test_cb.columns if col not in cat_cols]:
    X_test_cb[c] = pd.to_numeric(X_test_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

X_test_xgb = X_test_base.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_test_xgb[c] = X_test_xgb[c].astype(np.float32)
    else:
        X_test_xgb[c] = (X_test_xgb[c] - 1).astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

print("Predicting with GBDT Binary 15-model ensemble...")
p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))
p_lgb_mse_sum = np.zeros(len(df_test))

# LGB MSE 133-feature numpy matrix
X_test_133_mat = X_test_133.values.astype(np.float32)

for seed in SEEDS:
    # 1. LGB Binary (119 features)
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test_base)
    # 2. CB Binary (119 features)
    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]
    # 3. XGB Binary (119 features)
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]
    # 4. LGB MSE (133 features)
    m_lgb_mse = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_mse_model_seed{seed}.txt'))
    p_lgb_mse_sum += m_lgb_mse.predict(X_test_133_mat)

n_seeds = len(SEEDS)
p_lgb_bin = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb_bin = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb_bin = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)
p_gbdt_bin = np.clip(W_LGB_BIN * p_lgb_bin + W_CB_BIN * p_cb_bin + W_XGB_BIN * p_xgb_bin, 1e-6, 1 - 1e-6)
p_gbdt_mse = np.clip(p_lgb_mse_sum / n_seeds, 1e-6, 1 - 1e-6)

# SimpleMLP MSE Inference (5 seeds on 133 features)
art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
num_cols_mlp, cat_cols_mlp = art['num_cols'], art['cat_cols']
mean_mlp, std_mlp = art['mean'], art['std']
cat_vocabs = art['cat_vocabs']
cat_cardinalities = art['cat_cardinalities']
num_dim = art['num_dim']

num_raw = X_test_133[num_cols_mlp].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)
num_t = torch.tensor(num_z, dtype=torch.float32).to(DEVICE)

cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_test_133[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_test_133), 0), dtype=np.int64)
cat_t = torch.tensor(cat_arr, dtype=torch.long).to(DEVICE)

p_mlp_sum = np.zeros(len(df_test), dtype=np.float64)

for seed in SEEDS:
    mlp_net = SimpleMLP_MSE(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12).to(DEVICE)
    mlp_net.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt'), map_location=DEVICE))
    mlp_net.eval()
    with torch.no_grad():
        p_mlp_sum += mlp_net(num_t, cat_t).cpu().numpy()

p_mlp_mse = p_mlp_sum / len(SEEDS)

# Neural-GBDT Super-Blend (v42: Neural 40% + GBDT 40% + MSE 20%)
p_raw = W_GBDT_BIN * p_gbdt_bin + W_MLP_MSE * p_mlp_mse + W_LGB_MSE * p_gbdt_mse

count_shifts = joblib.load(os.path.join(model_dir, 'count_shifts_artifact.pkl'))
counts_test = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_test['strikes_before'].fillna(0).astype(int).astype(str)).values

p_cond = p_raw.copy()
for cc, s_val in count_shifts.items():
    p_cond[counts_test == cc] += s_val

CALIBRATION_SCALE = 1.10
CALIBRATION_SHIFT = -0.0045192086
p_calibrated = np.clip(0.5 + CALIBRATION_SCALE * (p_cond - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)

df_sub = pd.DataFrame({
    'row_id': df_test['row_id'],
    'control_success': p_calibrated
})

out_path = os.path.join(output_dir, 'submission.csv')
df_sub.to_csv(out_path, index=False)
print(f"Submission successfully saved to: {out_path}")
print(f"Summary stats: Mean={p_calibrated.mean():.6f}, Min={p_calibrated.min():.6f}, Max={p_calibrated.max():.6f}")
print(f"Total pipeline elapsed time: {time.time() - t0:.2f}s")
