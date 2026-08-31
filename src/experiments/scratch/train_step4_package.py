import sys, os, zipfile
print("6. Writing Inference Script...", flush=True)
v55_dir = '~/LG_data/work/submit_v55'
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

preds_mat = np.column_stack(preds)
pred_mean = np.mean(preds_mat, axis=1)
pred_std = np.std(preds_mat, axis=1)

scale_adaptive = np.maximum(1.10 - 2.0 * pred_std, 0.5)
p_calib = np.clip(0.5 + scale_adaptive * (pred_mean - 0.5) - 0.0035, 1e-6, 1 - 1e-6)

df_sub = pd.DataFrame({'row_id': df_test['row_id'], 'control_success': p_calib})
os.makedirs('output', exist_ok=True)
df_sub.to_csv('output/submission.csv', index=False)
print("v55 predictions saved.")
"""
with open(os.path.join(v55_dir, 'script.py'), 'w') as f:
    f.write(script_code)

print("7. Zipping submit_v55.zip...", flush=True)
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v55.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(v55_dir):
        for file in files:
            file_path = os.path.join(root, file)
            zf.write(file_path, os.path.relpath(file_path, v55_dir))
print("v55 Package Complete!", flush=True)
