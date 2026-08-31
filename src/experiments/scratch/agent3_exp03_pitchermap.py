"""EXP03: recover pitcher_id <-> pitcher_trackman_id mapping.

Strategy: main and trackman describe the same KBO pitches (trackman ~85% coverage).
Key = (season, month, dayofweek, pitcher_team, batter_team, inning, top_bottom)
identifies (almost) a specific half-inning of a specific game.
Accumulate co-occurrence between main pitcher_id and trackman pitcher id over keys,
then solve assignment.
"""
import pandas as pd, numpy as np, json, sys
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

MAIN = '~/LG_data/open/data/train.csv'
TM = '~/LG_data/open/data/trackman_history.csv'
OUT = '~/LG_data/scratch/agent3_cache'

df = pd.read_csv(MAIN, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                                'pitcher_team_id', 'batter_team_id', 'pitcher_id', 'pitcher_hand',
                                'balls_before', 'strikes_before', 'outs_before'])
tm = pd.read_csv(TM, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                              'pitcher_team', 'batter_team', 'pitcher_trackman_id', 'pitcher_hand',
                              'balls_before', 'strikes_before', 'outs_before'])

tm['top_bottom'] = tm['top_bottom'].map({'Top': 'T', 'Bottom': 'B'})
tm['hand'] = tm['pitcher_hand'].map({'Right': 2, 'Left': 1})
df['hand'] = df['pitcher_hand']

# franchise merge (same club renamed)
tm['team_f'] = tm['pitcher_team'].replace({'SK_WYV': 'SSG', 'SSG_LAN': 'SSG'})
tm['bteam_f'] = tm['batter_team'].replace({'SK_WYV': 'SSG', 'SSG_LAN': 'SSG'})

# ---------- step 1: team mapping via (season,month,dow,inning) schedule cosine ----------
def cos_map(a_keys, b_keys, dfa, dfb, acol, bcol):
    ga = dfa.groupby([acol] + a_keys).size().rename('n').reset_index()
    gb = dfb.groupby([bcol] + b_keys).size().rename('n').reset_index()
    pa = ga.pivot_table(index=acol, columns=a_keys, values='n', fill_value=0).astype(float)
    pb = gb.pivot_table(index=bcol, columns=b_keys, values='n', fill_value=0).astype(float)
    cols = sorted(set(pa.columns) & set(pb.columns))
    pa = pa.reindex(columns=cols, fill_value=0); pb = pb.reindex(columns=cols, fill_value=0)
    na = pa.div(np.linalg.norm(pa.values, axis=1) + 1e-9, axis=0)
    nb = pb.div(np.linalg.norm(pb.values, axis=1) + 1e-9, axis=0)
    return pd.DataFrame(na.values @ nb.values.T, index=pa.index, columns=pb.index)

main_teams = [t for t in sorted(df.pitcher_team_id.unique()) if 12 <= t <= 21]
tm_teams = [t for t in sorted(tm.team_f.unique()) if not t.startswith('MIN_') and not t.startswith('KBO') and not t.startswith('ACE')]
dfa = df[df.pitcher_team_id.isin(main_teams)]
dfb = tm[tm.team_f.isin(tm_teams)]
S = cos_map(['season', 'game_month', 'game_dayofweek'], ['season', 'game_month', 'game_dayofweek'],
            dfa, dfb, 'pitcher_team_id', 'team_f')
S = S.loc[main_teams, tm_teams]
r, c = linear_sum_assignment(-S.values)
team_map = {}
print("=== TEAM ASSIGNMENT (Hungarian) ===")
for i, j in zip(r, c):
    team_map[S.index[i]] = S.columns[j]
    runner = sorted(S.iloc[i].values)[-2]
    print(f"  team_id {S.index[i]:>3} -> {S.columns[j]:<9} cos={S.iloc[i, j]:.4f} (2nd best {runner:.4f})")
json.dump({str(k): v for k, v in team_map.items()}, open(f'{OUT}/team_map.json', 'w'), indent=1)

# ---------- step 2: pitcher co-occurrence ----------
df['team_f'] = df.pitcher_team_id.map(team_map)
df['bteam_f'] = df.batter_team_id.map(team_map)
d = df.dropna(subset=['team_f', 'bteam_f']).copy()
t = tm[tm.team_f.isin(tm_teams) & tm.bteam_f.isin(tm_teams)].copy()
print(f"\nusable main rows {len(d):,} / trackman rows {len(t):,}")

KEY = ['season', 'game_month', 'game_dayofweek', 'team_f', 'bteam_f', 'inning', 'top_bottom']
d['key'] = list(map(tuple, d[KEY].values))
t['key'] = list(map(tuple, t[KEY].values))
keys = pd.Index(sorted(set(d['key']) & set(t['key'])))
kidx = {k: i for i, k in enumerate(keys)}
print(f"shared keys: {len(keys):,}")

d = d[d['key'].isin(kidx)]
t = t[t['key'].isin(kidx)]
pmain = pd.Index(sorted(d.pitcher_id.unique()))
ptm = pd.Index(sorted(t.pitcher_trackman_id.unique()))
pm_i = {v: i for i, v in enumerate(pmain)}
pt_i = {v: i for i, v in enumerate(ptm)}

A = csr_matrix((np.ones(len(d)), (d.pitcher_id.map(pm_i).values, d['key'].map(kidx).values)),
               shape=(len(pmain), len(keys)))
B = csr_matrix((np.ones(len(t)), (t.pitcher_trackman_id.map(pt_i).values, t['key'].map(kidx).values)),
               shape=(len(ptm), len(keys)))
# normalize per key so a shared half-inning contributes ~1
Ab = A.copy(); Ab.data = np.ones_like(Ab.data)
Bb = B.copy(); Bb.data = np.ones_like(Bb.data)
kcount = np.asarray(Ab.sum(0)).ravel() * np.asarray(Bb.sum(0)).ravel()
w = 1.0 / np.sqrt(np.maximum(kcount, 1))
from scipy.sparse import diags
M = (Ab @ diags(w) @ Bb.T).toarray()
print("co-occurrence matrix", M.shape)

# hand constraint
hand_main = d.groupby('pitcher_id')['hand'].agg(lambda s: s.mode().iloc[0]).reindex(pmain).values
hand_tm = t.groupby('pitcher_trackman_id')['hand'].agg(lambda s: s.mode().iloc[0]).reindex(ptm).values
hand_ok = (hand_main[:, None] == hand_tm[None, :])
Mh = np.where(hand_ok, M, -1e6)

# normalize: cosine-like
na = np.sqrt(np.asarray(Ab.multiply(Ab).sum(1)).ravel())
nb = np.sqrt(np.asarray(Bb.multiply(Bb).sum(1)).ravel())
C = M / (na[:, None] * nb[None, :] + 1e-9)
Ch = np.where(hand_ok, C, -1e6)

ri, ci = linear_sum_assignment(-Ch)
rows = []
for i, j in zip(ri, ci):
    srt = np.sort(Ch[i])[::-1]
    best, second = srt[0], srt[1]
    rows.append(dict(pitcher_id=int(pmain[i]), tm_id=int(ptm[j]), score=float(Ch[i, j]),
                     best=float(best), second=float(second),
                     margin=float(Ch[i, j] - second), n_main=int(Ab[i].sum()),
                     raw=float(M[i, j])))
res = pd.DataFrame(rows).sort_values('score', ascending=False)
res.to_csv(f'{OUT}/pitcher_map_raw.csv', index=False)
print(res.describe())
print("\nscore distribution:")
print(res.score.describe(percentiles=[.05, .1, .25, .5, .75, .9]))
print("\nmargin (best - 2nd) distribution:")
print(res.margin.describe(percentiles=[.05, .1, .25, .5, .75, .9]))
print("\nhead:"); print(res.head(15).to_string())
print("\ntail:"); print(res.tail(15).to_string())
