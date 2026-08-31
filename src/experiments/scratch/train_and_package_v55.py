import sys, os, time, shutil, zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss
import joblib

BASE_DIR = os.path.expanduser('~/LG_data')
sys.path[:0] = [os.path.join(BASE_DIR, 'scratch'), BASE_DIR]

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from agent2_recover_labels import recover

v55_dir = os.path.join(BASE_DIR, 'work', 'submit_v55')
if os.path.exists(v55_dir): shutil.rmtree(v55_dir)
os.makedirs(os.path.join(v55_dir, 'model'))

print("1. Loading Data & Recovering Labels...")
df = pd.read_csv(config.TRAIN_PATH)
L = recover(df)
df = pd.concat([df, L], axis=1)

print("2. Building EB Matchup Dictionary...")
global_rate = df[config.TARGET_COL].mean()
p_rates = df.groupby('pitcher_id')[config.TARGET_COL].agg(['mean', 'count'])
b_rates = df.groupby('batter_id')[config.TARGET_COL].agg(['mean', 'count'])

p_dict = p_rates.to_dict('index')
b_dict = b_rates.to_dict('index')
joblib.dump({'global': global_rate, 'p': p_dict, 'b': b_dict}, os.path.join(v55_dir, 'model', 'eb_matchup.pkl'))

def get_eb(p_id, b_id):
    pr = p_dict.get(p_id, {'mean': global_rate, 'count': 0})['mean']
    br = b_dict.get(b_id, {'mean': global_rate, 'count': 0})['mean']
    return (pr * br) / global_rate if global_rate > 0 else global_rate

df['eb_matchup_prior'] = [get_eb(p, b) for p, b in zip(df['pitcher_id'], df['batter_id'])]

print("3. Fitting Preprocessor...")
prep = PitchPreprocessor().fit(df, is_final=True)
X = prep.transform(df)
y = df[config.TARGET_COL].values
X['eb_matchup_prior'] = df['eb_matchup_prior'].values
joblib.dump(prep, os.path.join(v55_dir, 'model', 'preprocessor_artifacts.pkl'))
joblib.dump(prep.trackman_builder, os.path.join(v55_dir, 'model', 'trackman_artifacts.pkl'))

print("4. Training LGBM Bin Models (3 Seeds)...")
cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match']
X_lgb = X.copy()
for c in cat_cols:
    if c in X_lgb: X_lgb[c] = X_lgb[c].astype('category')

SEEDS = [7, 123, 2025]
for s in SEEDS:
    ds = lgb.Dataset(X_lgb, label=y)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 31, 'seed': s, 'verbose': -1}
    m = lgb.train(params, ds, num_boost_round=300)
    m.save_model(os.path.join(v55_dir, 'model', f'lgbm_seed{s}.txt'))

print("5. Training Multi-Task MLP Models (3 Seeds)...")
class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_main = nn.Linear(64, 1)
        self.head_aux = nn.Linear(64, 7)
    def forward(self, x):
        feat = self.net(x)
        return torch.sigmoid(self.head_main(feat)).squeeze(), torch.sigmoid(self.head_aux(feat))

X_num = X_lgb.select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values
y_main_t = torch.tensor(y, dtype=torch.float32)
aux_cols = ["lab_reverse", "lab_middle", "lab_ball", "lab_strike", "lab_fastball", "lab_breaking", "lab_offspeed"]
y_aux_t = torch.tensor(df[aux_cols].fillna(0).values, dtype=torch.float32)
X_t = torch.tensor(X_num, dtype=torch.float32)

joblib.dump({'num_cols': list(X_lgb.select_dtypes(include=[np.number]).columns)}, os.path.join(v55_dir, 'model', 'mlp_cols.pkl'))

for s in SEEDS:
    torch.manual_seed(s)
    model = MultiTaskMLP(X_num.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    crit_main = nn.BCELoss()
    crit_aux = nn.BCELoss()
    
    # Just 2 epochs for speed on full data
    for epoch in range(2):
        opt.zero_grad()
        p_main, p_aux = model(X_t)
        loss = crit_main(p_main, y_main_t) + 0.5 * crit_aux(p_aux, y_aux_t)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), os.path.join(v55_dir, 'model', f'mlp_seed{s}.pt'))

print("6. Writing Inference Script...")
script_code = """import sys, os, torch, joblib
import pandas as pd, numpy as np, lightgbm as lgb
import torch.nn as nn
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
model_dir = os.path.join(SCRIPT_DIR, "model")

df_test = pd.read_csv("data/test.csv" if os.path.exists("data/test.csv") else os.path.join(SCRIPT_DIR, "data/test.csv"))

eb = joblib.load(os.path.join(model_dir, 'eb_matchup.pkl'))
def get_eb(p_id, b_id):
    pr = eb['p'].get(p_id, {'mean': eb['global']})['mean']
    br = eb['b'].get(b_id, {'mean': eb['global']})['mean']
    return (pr * br) / eb['global'] if eb['global'] > 0 else eb['global']
df_test['eb_matchup_prior'] = [get_eb(p, b) for p, b in zip(df_test['pitcher_id'], df_test['batter_id'])]

tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_obj = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
if isinstance(prep_obj, PitchPreprocessor):
    prep = prep_obj; prep.trackman_builder = tkm_builder
else:
    prep = PitchPreprocessor(); prep.artifacts = prep_obj.artifacts if not isinstance(prep_obj, dict) else prep_obj; prep.trackman_builder = tkm_builder; prep.is_fitted = True

X_test = prep.transform(df_test)
X_test['eb_matchup_prior'] = df_test['eb_matchup_prior'].values
cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match']
for c in cat_cols:
    if c in X_test: X_test[c] = X_test[c].astype('category')

preds = []
SEEDS = [7, 123, 2025]
for s in SEEDS:
    m = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_seed{s}.txt'))
    preds.append(m.predict(X_test))

class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_main = nn.Linear(64, 1)
        self.head_aux = nn.Linear(64, 7)
    def forward(self, x):
        feat = self.net(x)
        return torch.sigmoid(self.head_main(feat)).squeeze(), torch.sigmoid(self.head_aux(feat))

mlp_cols = joblib.load(os.path.join(model_dir, 'mlp_cols.pkl'))['num_cols']
X_num = torch.tensor(X_test[mlp_cols].fillna(0).astype(np.float32).values)

for s in SEEDS:
    mlp = MultiTaskMLP(X_num.shape[1])
    mlp.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_seed{s}.pt')))
    mlp.eval()
    with torch.no_grad():
        p_main, _ = mlp(X_num)
        preds.append(p_main.numpy())

# Uncertainty-Aware Adaptive Calibration
preds_mat = np.column_stack(preds)
pred_mean = np.mean(preds_mat, axis=1)
pred_std = np.std(preds_mat, axis=1)

# Scale shrinks from 1.10 down to 0.5 when models disagree
scale_adaptive = np.maximum(1.10 - 2.0 * pred_std, 0.5)
p_calib = np.clip(0.5 + scale_adaptive * (pred_mean - 0.5) - 0.0035, 1e-6, 1 - 1e-6)

df_sub = pd.DataFrame({'row_id': df_test['row_id'], 'control_success': p_calib})
os.makedirs('output', exist_ok=True)
df_sub.to_csv('output/submission.csv', index=False)
print("v55 predictions saved.")
"""
with open(os.path.join(v55_dir, 'script.py'), 'w') as f:
    f.write(script_code)

print("Done building v55.")
