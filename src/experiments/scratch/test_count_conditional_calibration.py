import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
cache_dir = os.path.join(BASE_DIR, 'scratch', 'cache_final')

print("=" * 80)
print("TESTING COUNT-CONDITIONAL CALIBRATION ACROSS 2022, 2023, 2024 FOLDS")
print("=" * 80)

df_all = pd.read_csv(os.path.join(data_dir, 'train.csv'))
seasons = df_all['season'].values
b = df_all['balls_before'].fillna(0).astype(int).astype(str)
s = df_all['strikes_before'].fillna(0).astype(int).astype(str)
count_codes = (b + '_' + s).values

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

# Load 3 folds
val_years = [2022, 2023, 2024]
fold_data = {}
for yr in val_years:
    c = np.load(os.path.join(cache_dir, f'final_val{yr}.npz'))
    mask = (seasons == yr)
    fold_data[yr] = {
        'y': c['y'].astype(np.float64),
        'p_lgb': c['p_lgb'].astype(np.float64),
        'p_cb': c['p_cb'].astype(np.float64),
        'p_xgb': c['p_xgb'].astype(np.float64),
        'counts': count_codes[mask]
    }

# 1. v33 Global Calibration Baseline
v33_scores = []
for yr in val_years:
    d = fold_data[yr]
    p_raw = 0.15 * (d['p_lgb'] - 0.007) + 0.75 * (d['p_cb'] - 0.008) + 0.10 * (d['p_xgb'] - 0.006)
    p_v33 = np.clip(0.5 + 1.10 * (p_raw - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
    sc, _ = calc_brier_skill_score(d['y'], p_v33)
    v33_scores.append(sc)

print(f"v33 Global Calibration Scores:")
for yr, sc in zip(val_years, v33_scores):
    print(f"  {yr} Val: {sc:.2f}")
print(f"  3-Fold Mean: {np.mean(v33_scores):.2f}\n")

# 2. Count-Conditional Shift Calibration
# Fit count shifts strictly on Train Fold (e.g. 2019-2021 for 2022, 2019-2022 for 2023, 2019-2023 for 2024)
cond_scores = []
for yr in val_years:
    d = fold_data[yr]
    p_raw = 0.15 * (d['p_lgb'] - 0.007) + 0.75 * (d['p_cb'] - 0.008) + 0.10 * (d['p_xgb'] - 0.006)
    
    # Calculate empirical shift per count on historical train seasons
    tr_mask = (seasons < yr)
    y_tr = df_all.loc[tr_mask, 'control_success'].values
    counts_tr = count_codes[tr_mask]
    
    # Train base rate vs val base rate per count
    unique_counts = np.unique(counts_tr)
    count_shifts = {}
    for cc in unique_counts:
        cc_mask = (counts_tr == cc)
        if cc_mask.sum() > 100:
            # Optimal linear adjustment factor per count
            r_cc = y_tr[cc_mask].mean()
            count_shifts[cc] = float(r_cc - 0.5) * 0.05
    
    p_cond = p_raw.copy()
    for cc, shift_val in count_shifts.items():
        cc_idx = (d['counts'] == cc)
        p_cond[cc_idx] += shift_val
        
    p_cond_cal = np.clip(0.5 + 1.10 * (p_cond - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
    sc, _ = calc_brier_skill_score(d['y'], p_cond_cal)
    cond_scores.append(sc)

print(f"Count-Conditional Calibration Scores:")
for yr, sc, orig in zip(val_years, cond_scores, v33_scores):
    print(f"  {yr} Val: {sc:.2f} (Gain vs v33: {sc - orig:+.2f} pts)")
print(f"  3-Fold Mean: {np.mean(cond_scores):.2f} (Mean Gain: {np.mean(cond_scores) - np.mean(v33_scores):+.2f} pts)")
