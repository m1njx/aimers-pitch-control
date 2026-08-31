import sys, os
import pandas as pd
import lightgbm as lgb
print("4. Training LGBM Bin Models (3 Seeds)...", flush=True)
v55_dir = '~/LG_data/work/submit_v55'
X = pd.read_pickle(os.path.join(v55_dir, 'model', 'X_train.pkl'))
y_df = pd.read_pickle(os.path.join(v55_dir, 'model', 'y_train.pkl'))
y = y_df['control_success'].values

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match']
for c in cat_cols:
    if c in X: X[c] = X[c].astype('category')

for s in [7, 123, 2025]:
    ds = lgb.Dataset(X, label=y)
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05, 'num_leaves': 31, 'seed': s, 'verbose': -1, 'num_threads': 4}
    m = lgb.train(params, ds, num_boost_round=300)
    m.save_model(os.path.join(v55_dir, 'model', f'lgbm_seed{s}.txt'))
    print(f"LGBM Seed {s} done.", flush=True)
