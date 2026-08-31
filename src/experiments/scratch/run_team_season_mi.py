"""
run_team_season_mi.py — Calculates Mutual Information and Chi-Square independence test
between (pitcher_team_id, season) and (batter_team_id, season) on train.csv.

Compares against MI(game_type, season) = 0.000444.
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
import config

print("Loading train.csv ...")
df = pd.read_csv(config.TRAIN_PATH, usecols=['season', 'game_type', 'pitcher_team_id', 'batter_team_id'])
print(f"Loaded {len(df):,} rows.\n")

# 1. MI Calculations (using natural log / nats, consistent with sklearn.metrics.mutual_info_score)
mi_gt = mutual_info_score(df['game_type'].astype(str), df['season'].astype(str))
mi_pitcher_team = mutual_info_score(df['pitcher_team_id'].astype(str), df['season'].astype(str))
mi_batter_team = mutual_info_score(df['batter_team_id'].astype(str), df['season'].astype(str))

# Normalized MI (NMI) and Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

v_gt = cramers_v(df['game_type'], df['season'])
v_pitcher_team = cramers_v(df['pitcher_team_id'], df['season'])
v_batter_team = cramers_v(df['batter_team_id'], df['season'])

chi2_p, p_p, dof_p, _ = chi2_contingency(pd.crosstab(df['pitcher_team_id'], df['season']))
chi2_b, p_b, dof_b, _ = chi2_contingency(pd.crosstab(df['batter_team_id'], df['season']))

print("=== Mutual Information & Independence Test Results ===")
print(f"MI(game_type, season):        {mi_gt:.6f}  | Cramér's V: {v_gt:.4f}")
print(f"MI(pitcher_team_id, season): {mi_pitcher_team:.6f}  | Cramér's V: {v_pitcher_team:.4f}  | Chi2: {chi2_p:.1f} (p={p_p:.4e})")
print(f"MI(batter_team_id, season):  {mi_batter_team:.6f}  | Cramér's V: {v_batter_team:.4f}  | Chi2: {chi2_b:.1f} (p={p_b:.4e})")

# Check team row distribution across seasons
print("\nPitcher Team Row Share (%) per Season:")
ct_p = pd.crosstab(df['pitcher_team_id'], df['season'], normalize='columns') * 100
print(ct_p.round(2))

print("\nBatter Team Row Share (%) per Season:")
ct_b = pd.crosstab(df['batter_team_id'], df['season'], normalize='columns') * 100
print(ct_b.round(2))

# Save results for markdown report
res_df = pd.DataFrame([
    {'feature': 'game_type', 'MI': mi_gt, 'Cramers_V': v_gt},
    {'feature': 'pitcher_team_id', 'MI': mi_pitcher_team, 'Cramers_V': v_pitcher_team},
    {'feature': 'batter_team_id', 'MI': mi_batter_team, 'Cramers_V': v_batter_team},
])
res_df.to_csv('~/LG_data/outputs/23_mi_results.csv', index=False)
ct_p.to_csv('~/LG_data/outputs/23_pitcher_team_crosstab.csv')
ct_b.to_csv('~/LG_data/outputs/23_batter_team_crosstab.csv')

print("\nSaved CSVs to outputs/23_*.csv")
