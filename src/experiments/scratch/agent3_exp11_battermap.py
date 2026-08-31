"""EXP11a: recover batter_id <-> batter_trackman_id mapping (same technique as EXP03)."""
import pandas as pd, numpy as np, json
from scipy.sparse import csr_matrix, diags
from scipy.optimize import linear_sum_assignment

MAIN = '~/LG_data/open/data/train.csv'
TM = '~/LG_data/open/data/trackman_history.csv'
OUT = '~/LG_data/scratch/agent3_cache'

team_map = {int(k): v for k, v in json.load(open(f'{OUT}/team_map.json')).items()}

df = pd.read_csv(MAIN, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                                'pitcher_team_id', 'batter_team_id', 'batter_id', 'batter_hand',
                                'pitcher_id', 'balls_before', 'strikes_before', 'outs_before'])
tm = pd.read_csv(TM, usecols=['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                              'pitcher_team', 'batter_team', 'batter_trackman_id', 'batter_hand',
                              'pitcher_trackman_id', 'balls_before', 'strikes_before', 'outs_before'])
_pm = pd.read_csv(f'{OUT}/pitcher_map_raw.csv'); _pm = _pm[_pm.margin >= 0.3]
tm['pitcher_id'] = tm.pitcher_trackman_id.map(dict(zip(_pm.tm_id, _pm.pitcher_id)))
tm['top_bottom'] = tm['top_bottom'].map({'Top': 'T', 'Bottom': 'B'})
tm['hand'] = tm['batter_hand'].map({'Right': 2, 'Left': 1})
tm['team_f'] = tm['pitcher_team'].replace({'SK_WYV': 'SSG', 'SSG_LAN': 'SSG'})
tm['bteam_f'] = tm['batter_team'].replace({'SK_WYV': 'SSG', 'SSG_LAN': 'SSG'})
df['team_f'] = df.pitcher_team_id.map(team_map)
df['bteam_f'] = df.batter_team_id.map(team_map)
df['hand'] = df.batter_hand

good = set(team_map.values())
d = df.dropna(subset=['team_f', 'bteam_f'])
t = tm[tm.team_f.isin(good) & tm.bteam_f.isin(good)]

KEY = ['season', 'game_month', 'game_dayofweek', 'team_f', 'bteam_f', 'inning', 'top_bottom',
       'pitcher_id', 'outs_before', 'balls_before', 'strikes_before']
d = d.dropna(subset=['pitcher_id']); t = t.dropna(subset=['pitcher_id'])
d['pitcher_id'] = d['pitcher_id'].astype(int); t = t.assign(pitcher_id=t['pitcher_id'].astype(int))
d = d.copy(); t = t.copy()
d['key'] = list(map(tuple, d[KEY].values))
t['key'] = list(map(tuple, t[KEY].values))
keys = pd.Index(sorted(set(d['key']) & set(t['key'])))
kidx = {k: i for i, k in enumerate(keys)}
d = d[d['key'].isin(kidx)]; t = t[t['key'].isin(kidx)]
print(f'main {len(d):,} tm {len(t):,} shared keys {len(keys):,}')

bmain = pd.Index(sorted(d.batter_id.unique())); btm = pd.Index(sorted(t.batter_trackman_id.unique()))
mi = {v: i for i, v in enumerate(bmain)}; ti = {v: i for i, v in enumerate(btm)}
A = csr_matrix((np.ones(len(d)), (d.batter_id.map(mi).values, d['key'].map(kidx).values)),
               shape=(len(bmain), len(keys)))
B = csr_matrix((np.ones(len(t)), (t.batter_trackman_id.map(ti).values, t['key'].map(kidx).values)),
               shape=(len(btm), len(keys)))
A.data = np.ones_like(A.data); B.data = np.ones_like(B.data)
kc = np.asarray(A.sum(0)).ravel() * np.asarray(B.sum(0)).ravel()
M = (A @ diags(1.0 / np.sqrt(np.maximum(kc, 1))) @ B.T).toarray()
na = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
nb = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
C = M / (na[:, None] * nb[None, :] + 1e-9)
hm = d.groupby('batter_id')['hand'].agg(lambda s: s.mode().iloc[0]).reindex(bmain).values
ht = t.groupby('batter_trackman_id')['hand'].agg(lambda s: s.mode().iloc[0]).reindex(btm).values
Ch = np.where(hm[:, None] == ht[None, :], C, -1e6)
ri, ci = linear_sum_assignment(-Ch)
rows = []
for i, j in zip(ri, ci):
    srt = np.sort(Ch[i])[::-1]
    rows.append(dict(batter_id=int(bmain[i]), tm_id=int(btm[j]), score=float(Ch[i, j]),
                     margin=float(Ch[i, j] - srt[1]), n_main=int(A[i].sum())))
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/batter_map_raw.csv', index=False)
print(res.score.describe(percentiles=[.1, .25, .5, .75]))
print(res.margin.describe(percentiles=[.1, .25, .5, .75]))
n_main = df.groupby('batter_id').size()
for thr in [0.0, 0.3, 0.5]:
    ok = res[res.margin >= thr]
    print(f'margin>={thr}: {len(ok)} batters, row coverage {n_main.reindex(ok.batter_id).sum()/len(df):.4f}')
