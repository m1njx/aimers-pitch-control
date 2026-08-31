import os
import sys
import time
import shutil
import zipfile
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
work_v51_dir = os.path.join(BASE_DIR, 'work', 'submit_v51')
model_dir = os.path.join(work_v51_dir, 'model')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v51.zip')

os.makedirs(model_dir, exist_ok=True)
sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

print("=" * 80)
print("STARTING SCENARIO C: 3D TUNNELING & AERODYNAMIC PHYSICS V51 RETRAINING (138 FEATS)")
print("=" * 80)
t0 = time.time()

# 1. Load full training dataset
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print(f"Loaded train.csv: {len(df_train):,} rows")

# 2. Load and copy trackman + preprocessor artifacts
tkm_art = joblib.load(os.path.join(work_v42_dir, 'model', 'trackman_artifacts.pkl'))
joblib.dump(tkm_art, os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(work_v42_dir, 'model', 'preprocessor_artifacts.pkl'))
joblib.dump(prep_art, os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

print(f"1. Trackman & Preprocessor artifacts loaded & copied ({time.time()-t0:.1f}s)")

X_base = prep.transform(df_train)

# Count x base
base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3. 3D Tunneling Core
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

# Decomposer
dec = joblib.load(os.path.join(work_v42_dir, 'model', 'asof_decomposer_artifacts.pkl'))
joblib.dump(dec, os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_train = dec.transform(df_train)
A_train.index = X_base.index

# 4. 138 Features: 133 Base + 5 Advanced Aerodynamic Signals
v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

b = df_train['balls_before'].fillna(0).values
s = df_train['strikes_before'].fillna(0).values
li = df_train['li'].fillna(1.0).values
r2 = (df_train['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df_train['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df_train['score_diff_pitcher_team'].fillna(0).values
inning = df_train['inning'].fillna(1).values
fb_rate = df_train['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df_train['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df_train['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df_train['pitcher_hand'].astype(str) == df_train['batter_hand'].astype(str)).astype(float).values

vaa_proxy = np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)
haa_proxy = np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)

# 5 New Aerodynamic & 3D Curvature Features
phys_drag_accel = (v0 ** 2) / (2.0 * dist_to_plate)
phys_spin_axis_deg = np.arctan2(hb, ivb) * (180.0 / np.pi)
phys_release_ext_ratio = ext / (np.abs(rel_height) + 0.1)
phys_visual_approach_div = np.sqrt(haa_proxy ** 2 + vaa_proxy ** 2)

all_extra_138 = {
    'tkm_tunnel_dist_015s': d_tunnel.astype(np.float32),
    'tkm_plate_break_divergence': ((d_plate - d_tunnel) / 0.15).astype(np.float32),
    'tkm_deception_index': (d_plate / (d_tunnel + 0.1)).astype(np.float32),
    'phys_effective_velocity': (v_rel * (60.5 / dist_to_plate)).astype(np.float32),
    'phys_vaa_proxy': vaa_proxy.astype(np.float32),
    'phys_haa_proxy': haa_proxy.astype(np.float32),
    'phys_spin_efficiency': (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32),
    'feat_count_advantage': (s - 1.5 * b).astype(np.float32),
    'feat_full_count': ((b == 3) & (s == 2)).astype(np.float32),
    'feat_pitcher_ahead': ((s > b) & (s >= 2)).astype(np.float32),
    'feat_pitcher_behind': ((b > s) & (b >= 2)).astype(np.float32),
    'feat_clutch_pressure': (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32),
    'feat_scoring_position': (r2 + r3).astype(np.float32),
    'feat_platoon_fastball_inter': (platoon_code * fb_rate).astype(np.float32),
    'feat_platoon_breaking_inter': (platoon_code * br_rate).astype(np.float32),
    'feat_platoon_offspeed_inter': (platoon_code * off_rate).astype(np.float32),
    'feat_late_inning_clutch': ((inning >= 7).astype(float) * li).astype(np.float32),
    # 5 High-Precision 3D Aerodynamic Physics Features
    'phys_flight_time': t_flight.astype(np.float32),
    'phys_drag_accel': phys_drag_accel.astype(np.float32),
    'phys_spin_axis_deg': phys_spin_axis_deg.astype(np.float32),
    'phys_release_ext_ratio': phys_release_ext_ratio.astype(np.float32),
    'phys_visual_approach_div': phys_visual_approach_div.astype(np.float32),
}

X_138 = pd.concat([X_base, A_train, pd.DataFrame(all_extra_138, index=X_base.index)], axis=1)
y_train = df_train['control_success'].values.astype(np.float32)
print(f"3. Full 138 Feature Matrix constructed: shape = {X_138.shape} ({time.time()-t0:.1f}s)")

# Count shifts artifact
count_shifts = joblib.load(os.path.join(work_v42_dir, 'model', 'count_shifts_artifact.pkl'))
joblib.dump(count_shifts, os.path.join(model_dir, 'count_shifts_artifact.pkl'))

# 5. Train LightGBM Direct MSE 5-Seeds
print("\n--- Training LightGBM Direct MSE (5 Seeds on 138 Feats) ---")
lgb_mse_params = {
    'objective': 'regression_l2',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'verbose': -1,
    'n_jobs': 4
}
SEEDS = [7, 123, 2025, 31415, 8675309]
X_138_mat = X_138.values.astype(np.float32)

for s in SEEDS:
    lgb_mse_params['seed'] = s
    trn_data = lgb.Dataset(X_138_mat, label=y_train)
    m_mse = lgb.train(lgb_mse_params, trn_data, num_boost_round=300)
    m_mse.save_model(os.path.join(model_dir, f'lgbm_mse_model_seed{s}.txt'))
print(f"LightGBM Direct MSE models trained ({time.time()-t0:.1f}s)")

# 6. Train LightGBM Binary 5-Seeds
print("\n--- Training LightGBM Binary (5 Seeds on Base Feats) ---")
lgb_bin_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'verbose': -1,
    'n_jobs': 4
}
for s in SEEDS:
    lgb_bin_params['seed'] = s
    trn_data = lgb.Dataset(X_base, label=y_train)
    m_bin = lgb.train(lgb_bin_params, trn_data, num_boost_round=300)
    m_bin.save_model(os.path.join(model_dir, f'lgbm_model_seed{s}.txt'))
print(f"LightGBM Binary models trained ({time.time()-t0:.1f}s)")

# 7. Train CatBoost Binary 5-Seeds
print("\n--- Training CatBoost Binary (5 Seeds on Base Feats) ---")
cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']
X_cb = X_base.copy()
for c in cat_cols:
    X_cb[c] = pd.to_numeric(X_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_cb.columns if col not in cat_cols]:
    X_cb[c] = pd.to_numeric(X_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

cb_params = {
    'iterations': 400,
    'learning_rate': 0.08,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'cat_features': cat_cols,
    'thread_count': 4,
    'verbose': 0
}
for s in SEEDS:
    cb_params['random_seed'] = s
    m_cb = CatBoostClassifier(**cb_params)
    m_cb.fit(X_cb, y_train)
    m_cb.save_model(os.path.join(model_dir, f'catboost_model_seed{s}.cbm'))
print(f"CatBoost Binary models trained ({time.time()-t0:.1f}s)")

# 8. Train XGBoost Binary 5-Seeds
print("\n--- Training XGBoost Binary (5 Seeds on Base Feats) ---")
X_xgb = X_base.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_xgb[c] = X_xgb[c].astype(np.float32)
    else:
        X_xgb[c] = (X_xgb[c] - 1).astype(np.float32)
X_xgb = X_xgb.astype(np.float32)

xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'eval_metric': 'logloss',
    'n_jobs': 4,
    'verbosity': 0
}
for s in SEEDS:
    xgb_params['random_state'] = s
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(X_xgb, y_train)
    m_xgb.save_model(os.path.join(model_dir, f'xgb_model_seed{s}.json'))
print(f"XGBoost Binary models trained ({time.time()-t0:.1f}s)")

# 9. Train SimpleMLP 5-Seeds on 138 Features (BCE Logits Loss)
print("\n--- Training SimpleMLP (5 Seeds on 138 Feats with BCE Logits) ---")
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

class SimpleMLP_BCE(nn.Module):
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
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

num_cols_mlp = [col for col in X_138.columns if col not in cat_cols]
cat_cols_mlp = [col for col in cat_cols if col in X_138.columns]

num_raw = X_138[num_cols_mlp].astype(np.float32).values
mean_mlp = np.nanmean(num_raw, axis=0)
std_mlp = np.nanstd(num_raw, axis=0)
std_mlp[std_mlp < 1e-6] = 1.0
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)

cat_vocabs = {}
cat_cols_arr = []
cat_cardinalities = []
for c in cat_cols_mlp:
    vals = X_138[c].astype(str)
    unq = vals.unique().tolist()
    vocab = {v: i for i, v in enumerate(unq)}
    cat_vocabs[c] = vocab
    cat_cardinalities.append(len(vocab) + 1)
    unk_idx = len(vocab)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)

cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_138), 0), dtype=np.int64)

mlp_artifacts = {
    'num_cols': num_cols_mlp,
    'cat_cols': cat_cols_mlp,
    'mean': mean_mlp,
    'std': std_mlp,
    'cat_vocabs': cat_vocabs,
    'cat_cardinalities': cat_cardinalities,
    'num_dim': len(num_cols_mlp)
}
joblib.dump(mlp_artifacts, os.path.join(model_dir, 'mlp_artifacts.pkl'))

num_t = torch.tensor(num_z, dtype=torch.float32)
cat_t = torch.tensor(cat_arr, dtype=torch.long)
y_t = torch.tensor(y_train, dtype=torch.float32)
ds = TensorDataset(num_t, cat_t, y_t)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training SimpleMLP on {device} (5 Seeds)...")

for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    loader = DataLoader(ds, batch_size=4096, shuffle=True, drop_last=False)
    net = SimpleMLP_BCE(len(num_cols_mlp), cat_cardinalities, hidden=(128, 64), dropout=0.12).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8)
    
    net.train()
    for ep in range(8):
        for bx_num, bx_cat, by in loader:
            bx_num, bx_cat, by = bx_num.to(device), bx_cat.to(device), by.to(device)
            opt.zero_grad()
            logits = net(bx_num, bx_cat)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
        sch.step()
    
    torch.save(net.state_dict(), os.path.join(model_dir, f'mlp_model_seed{s}.pt'))
    print(f"  SimpleMLP seed {s} saved.")

# 10. Copy helper files into submit_v51
shutil.copy2(os.path.join(work_v42_dir, 'preprocessing.py'), os.path.join(work_v51_dir, 'preprocessing.py'))
shutil.copy2(os.path.join(work_v42_dir, 'trackman_features.py'), os.path.join(work_v51_dir, 'trackman_features.py'))
shutil.copy2(os.path.join(work_v42_dir, 'agent2_asof_decomp2.py'), os.path.join(work_v51_dir, 'agent2_asof_decomp2.py'))

decomp_path = os.path.join(work_v51_dir, 'agent2_asof_decomp2.py')
with open(decomp_path, 'r') as f:
    d_text = f.read()
d_text = d_text.replace("sys.path.insert(0, os.path.expanduser('~/LG_data'))", "# Clean relative imports")
d_text = d_text.replace("import config\n\nTGT = config.TARGET_COL", "try:\n    import config\n    TGT = getattr(config, 'TARGET_COL', 'control_success')\nexcept ImportError:\n    TGT = 'control_success'")
with open(decomp_path, 'w') as f:
    f.write(d_text)

print(f"All models & artifacts successfully trained and saved ({time.time()-t0:.1f}s elapsed)")
