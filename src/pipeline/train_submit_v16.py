import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LG_DATA_DIR = os.path.expanduser("~/LG_data")
if LG_DATA_DIR not in sys.path:
    sys.path.insert(0, LG_DATA_DIR)

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from agent2_asof_decomp2 import AsofDecomposer2
from preprocessing import PitchPreprocessor

t0 = time.time()
print("Starting submit_v16 Model Training Pipeline (119 features with 3D Pitch Tunneling)...")

SEEDS = [7, 123, 2025, 31415, 8675309]
model_dir = os.path.join(SCRIPT_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# 1. Load train data
print("Loading train.csv ...")
df_train = pd.read_csv(os.path.join(LG_DATA_DIR, "open/data/train.csv"))
print(f"Train data shape: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns")

# 2. Base feature prep
print("Fitting PitchPreprocessor ...")
prep = PitchPreprocessor()
prep.fit(df_train, is_final=True)
joblib.dump(prep.artifacts, os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
joblib.dump(prep.trackman_builder.artifacts, os.path.join(model_dir, 'trackman_artifacts.pkl'))

X_train = prep.transform(df_train)

# Derive count_x_base
base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = {val: idx for idx, val in enumerate(count_x_base_raw.unique())}
prep.count_x_base_map = cat_map
joblib.dump(prep, os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
X_train['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# --- 3D Pitch Tunneling Feature Calculations (Module 264) ---
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

# --- asof_dec 분해 피처 ---
print("Fitting AsofDecomposer2 ...")
dec = AsofDecomposer2()
dec.fit(df_train, val_season=2025)
joblib.dump(dec, os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_train = dec.transform(df_train)
A_train.index = X_train.index
X_train = pd.concat([X_train, A_train], axis=1)

y_train = df_train['control_success'].values

cat_cols = [c for c in X_train.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]

print(f"Final feature matrix shape: {X_train.shape[1]} columns across {X_train.shape[0]:,} rows")

# Prepare datasets for GBDT
X_train_cb = X_train.copy()
for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].astype(int).astype(str)
for c in [col for col in X_train_cb.columns if col not in cat_cols]:
    X_train_cb[c] = X_train_cb[c].astype(np.float32)

X_train_xgb = X_train.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_train_xgb[c] = X_train_xgb[c].astype(np.float32)
    else:
        X_train_xgb[c] = (X_train_xgb[c] - 1).astype(np.float32)
X_train_xgb = X_train_xgb.astype(np.float32)

# Train 5-Seed GBDTs
print("Training LightGBM 5 seeds ...")
for seed in SEEDS:
    params_lgb = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': seed,
        'verbose': -1,
        'n_estimators': 300,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    m_lgb = lgb.train(params_lgb, dtrain)
    m_lgb.save_model(os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))

print("Training CatBoost 5 seeds ...")
for seed in SEEDS:
    m_cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.06,
        depth=6,
        cat_features=cat_cols,
        random_seed=seed,
        verbose=0
    )
    m_cb.fit(X_train_cb, y_train)
    m_cb.save_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))

print("Training XGBoost 5 seeds ...")
for seed in SEEDS:
    m_xgb = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        random_state=seed,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    m_xgb.fit(X_train_xgb, y_train)
    m_xgb.save_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))

# Train 5-Seed SimpleMLP
print("Training SimpleMLP 5 seeds ...")
num_cols = [c for c in X_train.columns if c not in cat_cols]
cat_cols_mlp = cat_cols.copy()

mean = X_train[num_cols].values.mean(axis=0).astype(np.float32)
std = X_train[num_cols].values.std(axis=0).astype(np.float32)
std[std < 1e-6] = 1.0

cat_vocabs = {}
cat_cardinalities = []
for c in cat_cols_mlp:
    vals = X_train[c].astype(str).unique()
    vocab = {v: i for i, v in enumerate(sorted(vals))}
    cat_vocabs[c] = vocab
    cat_cardinalities.append(len(vocab) + 1)

mlp_artifacts = {
    'num_cols': num_cols,
    'cat_cols': cat_cols_mlp,
    'mean': mean,
    'std': std,
    'cat_vocabs': cat_vocabs,
    'cat_cardinalities': cat_cardinalities,
    'num_dim': len(num_cols),
    'mlp_shifts': {s: 0.0 for s in SEEDS}
}
joblib.dump(mlp_artifacts, os.path.join(model_dir, 'mlp_artifacts.pkl'))

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
cat_cols_arr = [X_train[c].astype(str).map(cat_vocabs[c]).fillna(len(cat_vocabs[c])).values for c in cat_cols_mlp]
cat_arr = np.stack(cat_cols_arr, axis=1).astype(np.int64)

X_num_t = torch.tensor(num_z, dtype=torch.float32)
X_cat_t = torch.tensor(cat_arr, dtype=torch.int64)
y_t = torch.tensor(y_train, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X_num_t, X_cat_t, y_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)

for seed in SEEDS:
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

print(f"Training completed successfully in {time.time()-t0:.1f} seconds!")
