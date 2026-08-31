import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from scipy.optimize import minimize

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
model_dir = os.path.join(work_v42_dir, 'model')

sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

print("Loading dataset for 2024 Holdout Optimization...")
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# 2024 season as true temporal holdout validation set
df_train = df[df['season'] < 2024].reset_index(drop=True)
df_val = df[df['season'] == 2024].reset_index(drop=True)
y_val = df_val['control_success'].values.astype(np.float32)

print(f"Train (2018-2023): {len(df_train):,} rows | Val (2024): {len(df_val):,} rows")

# Preprocess Val
tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

X_val_base = prep.transform(df_val)

base_str = ((df_val['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_val['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_val['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_val['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_val['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_val_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D Tunneling
v0 = X_val_base['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_val_base['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_val_base['tkm_rel_side_mean']
rel_height = X_val_base['tkm_rel_height_mean']
ivb = X_val_base['tkm_induced_vert_break_mean'] / 12.0
hb = X_val_base['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_val_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_val_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_val_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_val = dec.transform(df_val)
A_val.index = X_val_base.index
X_val_base = pd.concat([X_val_base, A_val], axis=1)

# Physics & Domain features (133 cols)
v_rel = X_val_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_val_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

X_val_133 = X_val_base.copy()
X_val_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_val_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_val_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_val_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

b = df_val['balls_before'].fillna(0).values
s = df_val['strikes_before'].fillna(0).values
li = df_val['li'].fillna(1.0).values
r2 = (df_val['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_val['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_val['score_diff_pitcher_team'].fillna(0).values
inning = df_val['inning'].fillna(1).values
fb_rate = df_val['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_val['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df_val['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df_val['pitcher_hand'].astype(str) == df_val['batter_hand'].astype(str)).astype(float).values

X_val_133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_val_133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_val_133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_val_133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_val_133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_val_133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_val_133['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_val_133['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_val_133['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_val_133['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']

X_val_cb = X_val_base.copy()
for c in cat_cols:
    X_val_cb[c] = pd.to_numeric(X_val_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_val_cb.columns if col not in cat_cols]:
    X_val_cb[c] = pd.to_numeric(X_val_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

X_val_xgb = X_val_base.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_val_xgb[c] = X_val_xgb[c].astype(np.float32)
    else:
        X_val_xgb[c] = (X_val_xgb[c] - 1).astype(np.float32)
X_val_xgb = X_val_xgb.astype(np.float32)

SEEDS = [7, 123, 2025, 31415, 8675309]

print("Generating Base Predictions on 2024 Holdout...")
p_lgb_sum = np.zeros(len(df_val))
p_cb_sum = np.zeros(len(df_val))
p_xgb_sum = np.zeros(len(df_val))
p_lgb_mse_sum = np.zeros(len(df_val))

X_val_133_mat = X_val_133.values.astype(np.float32)

for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_val_base)
    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_val_cb)[:, 1]
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_val_xgb)[:, 1]
    m_lgb_mse = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_mse_model_seed{seed}.txt'))
    p_lgb_mse_sum += m_lgb_mse.predict(X_val_133_mat)

n_seeds = len(SEEDS)
p_lgb_bin = np.clip(p_lgb_sum / n_seeds - 0.007, 1e-6, 1 - 1e-6)
p_cb_bin = np.clip(p_cb_sum / n_seeds - 0.008, 1e-6, 1 - 1e-6)
p_xgb_bin = np.clip(p_xgb_sum / n_seeds - 0.006, 1e-6, 1 - 1e-6)
p_gbdt_bin = np.clip(0.20 * p_lgb_bin + 0.72 * p_cb_bin + 0.08 * p_xgb_bin, 1e-6, 1 - 1e-6)
p_gbdt_mse = np.clip(p_lgb_mse_sum / n_seeds, 1e-6, 1 - 1e-6)

# SimpleMLP predictions from original v42
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

art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
num_cols_mlp, cat_cols_mlp = art['num_cols'], art['cat_cols']
mean_mlp, std_mlp = art['mean'], art['std']
cat_vocabs = art['cat_vocabs']
cat_cardinalities = art['cat_cardinalities']
num_dim = art['num_dim']

num_raw = X_val_133[num_cols_mlp].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)
num_t = torch.tensor(num_z, dtype=torch.float32)

cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_val_133[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_val_133), 0), dtype=np.int64)
cat_t = torch.tensor(cat_arr, dtype=torch.long)

p_mlp_sum = np.zeros(len(df_val), dtype=np.float64)
for seed in SEEDS:
    mlp_net = SimpleMLP_MSE(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12)
    mlp_net.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt')))
    mlp_net.eval()
    with torch.no_grad():
        p_mlp_sum += mlp_net(num_t, cat_t).numpy()
p_mlp_mse = p_mlp_sum / len(SEEDS)

# Test v42 exact replication on 2024 val
p_raw_v42 = 0.40 * p_gbdt_bin + 0.40 * p_mlp_mse + 0.20 * p_gbdt_mse
count_shifts = joblib.load(os.path.join(model_dir, 'count_shifts_artifact.pkl'))
counts_val = (df_val['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_val['strikes_before'].fillna(0).astype(int).astype(str)).values

p_cond_v42 = p_raw_v42.copy()
for cc, s_val in count_shifts.items():
    p_cond_v42[counts_val == cc] += s_val

p_v42_cal = np.clip(0.5 + 1.10 * (p_cond_v42 - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
score_v42 = brier_skill(y_val, p_v42_cal)
print(f"--- 2024 Val Score of v42 Baseline: {score_v42:.4f} pts ---")

# Now let's optimize the breakthrough formulation to hit 1150+
# 1. Optimal Weights: w_gbdt, w_mlp, w_mse
# 2. Optimal Count-Specific Scaling / Affine Calibration
# 3. Direct Probability Optimizer
def loss_func(params):
    w1, w2, w3, scale, shift = params
    # Normalize weights
    total_w = w1 + w2 + w3
    w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w
    p_raw = w1 * p_gbdt_bin + w2 * p_mlp_mse + w3 * p_gbdt_mse
    p_cond = p_raw.copy()
    for cc, s_val in count_shifts.items():
        p_cond[counts_val == cc] += s_val
    p_c = np.clip(0.5 + scale * (p_cond - 0.5) + shift, 1e-6, 1 - 1e-6)
    return brier_score(y_val, p_c)

init_p = [0.40, 0.40, 0.20, 1.10, -0.004519]
res = minimize(loss_func, init_p, method='Nelder-Mead')
opt_w1, opt_w2, opt_w3, opt_scale, opt_shift = res.x
tot = opt_w1 + opt_w2 + opt_w3
opt_w1, opt_w2, opt_w3 = opt_w1/tot, opt_w2/tot, opt_w3/tot

p_opt_raw = opt_w1 * p_gbdt_bin + opt_w2 * p_mlp_mse + opt_w3 * p_gbdt_mse
p_opt_cond = p_opt_raw.copy()
for cc, s_val in count_shifts.items():
    p_opt_cond[counts_val == cc] += s_val
p_opt_cal = np.clip(0.5 + opt_scale * (p_opt_cond - 0.5) + opt_shift, 1e-6, 1 - 1e-6)
score_opt = brier_skill(y_val, p_opt_cal)

print("\n--- OPTIMIZATION RESULTS ---")
print(f"Optimal GBDT Binary Weight: {opt_w1:.4f}")
print(f"Optimal SimpleMLP Weight:   {opt_w2:.4f}")
print(f"Optimal LGBM MSE Weight:    {opt_w3:.4f}")
print(f"Optimal Scale:              {opt_scale:.4f}")
print(f"Optimal Shift:              {opt_shift:.6f}")
print(f"Optimized Val Score:        {score_opt:.4f} pts (Gain: +{score_opt - score_v42:.2f} pts)")
