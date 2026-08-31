"""EXP02a: recover team_id <-> trackman team string mapping via schedule fingerprints."""
import pandas as pd, numpy as np
from scipy.optimize import linear_sum_assignment

MAIN = '~/LG_data/open/data/train.csv'
TM = '~/LG_data/open/data/trackman_history.csv'

df = pd.read_csv(MAIN, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                                'pitcher_team_id', 'batter_team_id', 'game_type', 'pitcher_hand'])
tm = pd.read_csv(TM, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                              'pitcher_team', 'batter_team', 'pitcher_hand', 'game_date'])

print("main team_id row counts by season:")
print(df.pivot_table(index='pitcher_team_id', columns='season', values='inning', aggfunc='size'))
print()
print("main team_id x game_type:")
print(df.pivot_table(index='pitcher_team_id', columns='game_type', values='inning', aggfunc='size'))
print()
tm['is_min'] = tm.pitcher_team.str.startswith('MIN_') | tm.batter_team.str.startswith('MIN_')
print("trackman pitcher_team row counts by season (all):")
print(tm.pivot_table(index='pitcher_team', columns='season', values='inning', aggfunc='size').fillna(0).astype(int))
print()
print("hand distribution main:", df.pitcher_hand.value_counts(normalize=True).to_dict())
print("hand distribution tm:", tm.pitcher_hand.value_counts(normalize=True).to_dict())

# --- schedule fingerprint: rows per (season, month, dayofweek) ---
def fp(d, teamcol, seasons):
    g = d.groupby([teamcol, 'season', 'game_month', 'game_dayofweek']).size().rename('n').reset_index()
    piv = g.pivot_table(index=teamcol, columns=['season', 'game_month', 'game_dayofweek'], values='n', fill_value=0)
    return piv

A = fp(df, 'pitcher_team_id', None)
B = fp(tm, 'pitcher_team', None)
cols = sorted(set(A.columns) & set(B.columns))
A2 = A.reindex(columns=cols, fill_value=0).astype(float)
B2 = B.reindex(columns=cols, fill_value=0).astype(float)
An = A2.div(np.linalg.norm(A2.values, axis=1), axis=0)
Bn = B2.div(np.linalg.norm(B2.values, axis=1), axis=0)
S = An.values @ Bn.values.T
sim = pd.DataFrame(S, index=A2.index, columns=B2.index)
print("\ntop-3 trackman team match per main team_id (cosine on (season,month,dow) pitch counts):")
for t in sim.index:
    row = sim.loc[t].sort_values(ascending=False)
    print(f"  team_id {t}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.head(3).items()))
