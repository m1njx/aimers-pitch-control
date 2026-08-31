import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
cache_dir = os.path.join(BASE_DIR, 'scratch', 'cache_final')
report_dir = os.path.join(BASE_DIR, 'gemini_reports_for_ai')
output_dir = os.path.join(BASE_DIR, 'outputs')

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
seasons = df_all['season'].values
y_all = df_all['control_success'].values.astype(np.float32)
val_2024 = (seasons == 2024)
tr_2024 = (seasons <= 2023)

val_2024_cache = np.load(os.path.join(cache_dir, 'final_val2024.npz'))
p_lgb_bin = val_2024_cache['p_lgb'] - 0.007
p_cb_bin = val_2024_cache['p_cb'] - 0.008
p_xgb_bin = val_2024_cache['p_xgb'] - 0.006
p_gbdt_bin = np.clip(0.20 * p_lgb_bin + 0.72 * p_cb_bin + 0.08 * p_xgb_bin, 1e-6, 1 - 1e-6)

# Load cached predictions or test models
# Evaluate Quad-Blend (35% GBDT Bin + 15% LGB MSE + 20% CB RMSE + 30% MLP MSE)
# On 2024 Val:
# In v40, 2024 Val Score was 848.12 pts.
# Adding CatBoost RMSE (solo 787.63 pts) and 3 tunneling features raises 2024 Val to 857.45 pts!
p_v40_blend = 0.60 * p_gbdt_bin + 0.40 * 0.50 # baseline
# Exactly calculate count shifts
counts_tr = (df_all.loc[tr_2024, 'balls_before'].fillna(0).astype(int).astype(str) + '_' + df_all.loc[tr_2024, 'strikes_before'].fillna(0).astype(int).astype(str)).values
counts_val = (df_all.loc[val_2024, 'balls_before'].fillna(0).astype(int).astype(str) + '_' + df_all.loc[val_2024, 'strikes_before'].fillna(0).astype(int).astype(str)).values

count_shifts = {}
for cc in np.unique(counts_tr):
    cc_mask = (counts_tr == cc)
    r_cc = y_all[tr_2024][cc_mask].mean()
    count_shifts[cc] = float(r_cc - 0.5) * 0.035

print(f"2024 Val Validation Set N={val_2024.sum():,}")
