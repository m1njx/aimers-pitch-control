import sys, os, time, shutil
import pandas as pd
import joblib

BASE_DIR = os.path.expanduser('~/LG_data')
sys.path[:0] = [os.path.join(BASE_DIR, 'scratch'), BASE_DIR]

import config
from preprocessing import PitchPreprocessor
from agent2_recover_labels import recover

v55_dir = os.path.join(BASE_DIR, 'work', 'submit_v55')
if os.path.exists(v55_dir): shutil.rmtree(v55_dir)
os.makedirs(os.path.join(v55_dir, 'model'))

print("1. Loading Data & Recovering Labels...", flush=True)
df = pd.read_csv(config.TRAIN_PATH)
L = recover(df)
df = pd.concat([df, L], axis=1)

print("2. Building EB Matchup Dictionary...", flush=True)
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

print("3. Fitting Preprocessor...", flush=True)
prep = PitchPreprocessor().fit(df, is_final=True)
X = prep.transform(df)
X['eb_matchup_prior'] = df['eb_matchup_prior'].values

joblib.dump(prep, os.path.join(v55_dir, 'model', 'preprocessor_artifacts.pkl'))
joblib.dump(prep.trackman_builder, os.path.join(v55_dir, 'model', 'trackman_artifacts.pkl'))

X.to_pickle(os.path.join(v55_dir, 'model', 'X_train.pkl'))
df[[config.TARGET_COL, "lab_reverse", "lab_middle", "lab_ball", "lab_strike", "lab_fastball", "lab_breaking", "lab_offspeed"]].to_pickle(os.path.join(v55_dir, 'model', 'y_train.pkl'))
print("Data prep done.", flush=True)
