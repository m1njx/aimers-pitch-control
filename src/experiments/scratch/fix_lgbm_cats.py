import os, sys, joblib
import pandas as pd, numpy as np
import lightgbm as lgb
v55_dir = '~/LG_data/work/submit_v55'
print("Retraining LGBM properly with exact integer codes...")
X = pd.read_pickle(os.path.join(v55_dir, 'model', 'X_train.pkl'))
y_df = pd.read_pickle(os.path.join(v55_dir, 'model', 'y_train.pkl'))
y = y_df['control_success'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match']

# Ensure integer types for categorical columns
for c in cat_cols:
    if c in X:
        X[c] = X[c].astype(int)

# Retrain
for s in [7, 123, 2025]:
    ds = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 31, 'seed': s, 'verbose': -1, 'num_threads': 4}
    m = lgb.train(params, ds, num_boost_round=300)
    m.save_model(os.path.join(v55_dir, 'model', f'lgbm_seed{s}.txt'))
    print(f"LGBM Seed {s} fixed.")

# Also fix the MLP model by training it with StandardScaler!
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
print("Retraining MLP with StandardScaler...")
mlp_cols = joblib.load(os.path.join(v55_dir, 'model', 'mlp_cols.pkl'))['num_cols']
X_num_raw = X[mlp_cols].fillna(0).values
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_raw)
joblib.dump(scaler, os.path.join(v55_dir, 'model', 'mlp_scaler.pkl'))

X_t = torch.tensor(X_num_scaled, dtype=torch.float32)
y_main_t = torch.tensor(y, dtype=torch.float32)
aux_cols = ["lab_reverse", "lab_middle", "lab_ball", "lab_strike", "lab_fastball", "lab_breaking", "lab_offspeed"]
y_aux_t = torch.tensor(y_df[aux_cols].fillna(0).values, dtype=torch.float32)

class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_main = nn.Linear(64, 1)
        self.head_aux = nn.Linear(64, 7)
    def forward(self, x):
        feat = self.net(x)
        return torch.sigmoid(self.head_main(feat)).squeeze(), torch.sigmoid(self.head_aux(feat))

for s in [7, 123, 2025]:
    torch.manual_seed(s)
    model = MultiTaskMLP(X_num_scaled.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = nn.BCELoss()
    for epoch in range(2):
        opt.zero_grad()
        p_main, p_aux = model(X_t)
        loss = crit(p_main, y_main_t) + 0.5 * crit(p_aux, y_aux_t)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), os.path.join(v55_dir, 'model', f'mlp_seed{s}.pt'))
    print(f"MLP Seed {s} fixed.")
