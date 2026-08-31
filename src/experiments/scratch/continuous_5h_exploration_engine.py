import os
import sys
import time
import shutil
import joblib
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from scipy.optimize import minimize
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
save_dir = os.path.join(BASE_DIR, 'scratch', 'continuous_5h_sota')
log_file = os.path.join(BASE_DIR, 'scratch', 'continuous_5h_search.log')
os.makedirs(save_dir, exist_ok=True)

def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

log("=" * 80)
log("🚀 5-HOUR NON-STOP MULTI-APPROACH AUTONOMOUS EXPLORATION ENGINE STARTED")
log("=" * 80)

# Load data safely
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
is_val24 = (df['season'] == 2024)
is_val23 = (df['season'] == 2023)
is_train = (df['season'] < 2024)

y = df['control_success'].values.astype(np.float32)
y_train = y[is_train]
y_val24 = y[is_val24]
y_val23 = y[is_val23]

log(f"Dataset loaded: Total {len(df):,} rows (Train: {is_train.sum():,}, Val24: {is_val24.sum():,}, Val23: {is_val23.sum():,})")

sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

tkm_art = joblib.load(os.path.join(work_v42_dir, 'model', 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(work_v42_dir, 'model', 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

dec = joblib.load(os.path.join(work_v42_dir, 'model', 'asof_decomposer_artifacts.pkl'))

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

log(f"Features prepared cleanly: {X_z.shape[1]} features.")

# Master continuous loop running for 5 hours (18,000 seconds)
end_time = time.time() + 5 * 3600
round_num = 0
best_2yr_score = 1791.29

while time.time() < end_time:
    round_num += 1
    log(f"\n{'='*40} [ITERATION ROUND {round_num}] {'='*40}")
    
    try:
        # Approach 1: Self-Attention Tabular Transformer
        log("[Approach 1] Training Multi-Head Self-Attention Tabular Transformer...")
        class TabularTransformer(nn.Module):
            def __init__(self, in_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
                super().__init__()
                self.proj = nn.Linear(in_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.head = nn.Sequential(nn.Linear(d_model, 32), nn.SiLU(), nn.Linear(32, 1))
            def forward(self, x):
                h = self.proj(x).unsqueeze(1) # (B, 1, d_model)
                h = self.transformer(h).squeeze(1)
                return self.head(h).squeeze(-1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_idx = np.where(is_train)[0]
        val24_idx = np.where(is_val24)[0]
        val23_idx = np.where(is_val23)[0]
        
        ds_tr = TensorDataset(torch.tensor(X_z[train_idx], dtype=torch.float32), torch.tensor(y[train_idx], dtype=torch.float32))
        loader = DataLoader(ds_tr, batch_size=4096, shuffle=True)
        
        net_tf = TabularTransformer(X_z.shape[1], d_model=64, nhead=4, num_layers=2, dropout=0.1).to(device)
        opt = torch.optim.AdamW(net_tf.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        
        net_tf.train()
        for ep in range(6):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = net_tf(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
        
        net_tf.eval()
        with torch.no_grad():
            p24_tf = 1.0 / (1.0 + np.exp(-net_tf(torch.tensor(X_z[val24_idx], dtype=torch.float32).to(device)).cpu().numpy()))
            p23_tf = 1.0 / (1.0 + np.exp(-net_tf(torch.tensor(X_z[val23_idx], dtype=torch.float32).to(device)).cpu().numpy()))
        
        p24_tf_cal = np.clip(0.5 + 1.10 * (p24_tf - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        p23_tf_cal = np.clip(0.5 + 1.10 * (p23_tf - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        s24_tf = brier_skill(y_val24, p24_tf_cal)
        s23_tf = brier_skill(y_val23, p23_tf_cal)
        mean_tf = 0.5 * (s24_tf + s23_tf)
        log(f"Tabular Transformer -> 2024: {s24_tf:.2f} pts | 2023: {s23_tf:.2f} pts | 2-Yr: {mean_tf:.2f} pts")
        
        # Approach 2: Deep Residual Swish TabNet with Focal Loss
        log("[Approach 2] Training Deep Residual Tabular MLP with Smooth Focal Loss...")
        class FocalLoss(nn.Module):
            def __init__(self, gamma=1.5):
                super().__init__()
                self.gamma = gamma
            def forward(self, logits, targets):
                p = torch.sigmoid(logits)
                ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                p_t = p * targets + (1 - p) * (1 - targets)
                loss = ce * ((1 - p_t) ** self.gamma)
                return loss.mean()
        
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
        ).to(device)
        
        opt_res = torch.optim.AdamW(net_res.parameters(), lr=3e-3, weight_decay=1e-4)
        crit_focal = FocalLoss(gamma=1.2)
        
        net_res.train()
        for ep in range(6):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt_res.zero_grad()
                out = net_res(bx).squeeze(-1)
                loss = crit_focal(out, by)
                loss.backward()
                opt_res.step()
        
        net_res.eval()
        with torch.no_grad():
            p24_res = 1.0 / (1.0 + np.exp(-net_res(torch.tensor(X_z[val24_idx], dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()))
            p23_res = 1.0 / (1.0 + np.exp(-net_res(torch.tensor(X_z[val23_idx], dtype=torch.float32).to(device)).squeeze(-1).cpu().numpy()))
        
        p24_res_cal = np.clip(0.5 + 1.10 * (p24_res - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        p23_res_cal = np.clip(0.5 + 1.10 * (p23_res - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        s24_res = brier_skill(y_val24, p24_res_cal)
        s23_res = brier_skill(y_val23, p23_res_cal)
        mean_res = 0.5 * (s24_res + s23_res)
        log(f"Focal ResMLP -> 2024: {s24_res:.2f} pts | 2023: {s23_res:.2f} pts | 2-Yr: {mean_res:.2f} pts")

        # Approach 3: Dynamic Non-Linear Meta-Ensembling
        log("[Approach 3] Evaluating Non-Linear Stacking Blend...")
        # Check ensemble with best past models
        p24_combo = 0.40 * p24_tf + 0.30 * p24_res + 0.30 * 0.4861
        p23_combo = 0.40 * p23_tf + 0.30 * p23_res + 0.30 * 0.4861
        p24_combo_cal = np.clip(0.5 + 1.10 * (p24_combo - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        p23_combo_cal = np.clip(0.5 + 1.10 * (p23_combo - 0.5) - 0.0035, 1e-6, 1 - 1e-6)
        s24_c = brier_skill(y_val24, p24_combo_cal)
        s23_c = brier_skill(y_val23, p23_combo_cal)
        log(f"Ensemble Combo -> 2024: {s24_c:.2f} pts | 2023: {s23_c:.2f} pts | 2-Yr: {0.5*(s24_c+s23_c):.2f} pts")

        log(f"Round {round_num} completed. Sleeping 30s before next exploration wave...")
        time.sleep(30)
        
    except Exception as e:
        log(f"Error in round {round_num}: {e}")
        log(traceback.format_exc())
        time.sleep(30)

log("5-Hour Continuous Exploration Completed.")
