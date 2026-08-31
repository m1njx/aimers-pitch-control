"""agent3_tkm_sit.py — SITUATIONAL pitch-mix / stuff expectation features from trackman.

New use of the recovered pitcher-ID mapping (EXP03): instead of static pitcher averages
(EXP04, rejected), predict *what pitch is about to be thrown* in THIS situation:
    P(pitch_type | pitcher, balls, strikes, batter_hand)  and  E[velo/break | same]
with hierarchical shrinkage (situation-cell -> pitcher -> league situation -> global).

This is legal: uses only trackman seasons <= as_of_season, keyed on columns available
for every test row (pitcher_id, count, batter hand). No current-pitch information.
"""
import numpy as np, pandas as pd, functools
from pathlib import Path

CACHE = Path('~/LG_data/scratch/agent3_cache')
TM_PATH = '~/LG_data/open/data/trackman_history.csv'

TARGETS = ['is_fb', 'is_br', 'is_os', 'rel_speed', 'induced_vert_break', 'horz_break',
           'rel_height', 'rel_side', 'extension', 'spin_rate']


@functools.lru_cache(maxsize=1)
def _tm():
    tm = pd.read_csv(TM_PATH, usecols=['season', 'pitcher_trackman_id', 'pitch_type_group',
                                       'balls_before', 'strikes_before', 'batter_hand',
                                       'rel_speed', 'induced_vert_break', 'horz_break',
                                       'rel_height', 'rel_side', 'extension', 'spin_rate'])
    res = pd.read_csv(CACHE / 'pitcher_map_raw.csv')
    res = res[res.margin >= 0.3]
    tm['pid'] = tm.pitcher_trackman_id.map(dict(zip(res.tm_id, res.pitcher_id)))
    tm = tm.dropna(subset=['pid'])
    tm['pid'] = tm['pid'].astype(np.int32)
    tm['bh'] = (tm.batter_hand == 'Right').astype(np.int8) + 1   # 2=Right matches main coding
    tm['is_fb'] = (tm.pitch_type_group == 'fastball').astype(np.float32)
    tm['is_br'] = (tm.pitch_type_group == 'breaking').astype(np.float32)
    tm['is_os'] = (tm.pitch_type_group == 'offspeed').astype(np.float32)
    tm['b'] = tm.balls_before.clip(0, 3).astype(np.int8)
    tm['s'] = tm.strikes_before.clip(0, 2).astype(np.int8)
    return tm


def build_situational(as_of_season: int, k_sit: float = 40.0, k_pit: float = 80.0) -> pd.DataFrame:
    """Returns table indexed by (pid, b, s, bh) with shrunken situational expectations."""
    t = _tm()
    t = t[t.season <= as_of_season]
    glob = t[TARGETS].mean()                                   # global
    L0 = t.groupby(['b', 's', 'bh'])[TARGETS].mean()           # league situational
    L1n = t.groupby('pid').size()
    L1 = t.groupby('pid')[TARGETS].mean()
    # shrink pitcher mean toward global
    lam1 = (L1n / (L1n + k_pit)).values[:, None]
    L1s = pd.DataFrame(lam1 * L1.values + (1 - lam1) * glob.values[None, :],
                       index=L1.index, columns=TARGETS)
    L2n = t.groupby(['pid', 'b', 's', 'bh']).size()
    L2 = t.groupby(['pid', 'b', 's', 'bh'])[TARGETS].mean()

    idx = L2.index
    pid = idx.get_level_values('pid')
    bsh = pd.MultiIndex.from_arrays([idx.get_level_values('b'), idx.get_level_values('s'),
                                     idx.get_level_values('bh')])
    # expected value if the pitcher behaved like league in this situation
    base = L1s.reindex(pid).values + (L0.reindex(bsh).values - glob.values[None, :])
    lam2 = (L2n / (L2n + k_sit)).values[:, None]
    est = lam2 * L2.values + (1 - lam2) * base
    F = pd.DataFrame(est, index=idx, columns=[f'sit_{c}' for c in TARGETS]).astype(np.float32)
    F['sit_n'] = np.log1p(L2n.values).astype(np.float32)
    # how much the pitcher deviates from his own baseline in this count
    F['sit_fb_dev'] = (F['sit_is_fb'].values - L1s['is_fb'].reindex(pid).values).astype(np.float32)
    F['sit_velo_dev'] = (F['sit_rel_speed'].values - L1s['rel_speed'].reindex(pid).values).astype(np.float32)
    # entropy of the (fb, br, os) mix = pitcher unpredictability in this situation
    q = F[['sit_is_fb', 'sit_is_br', 'sit_is_os']].values
    q = np.clip(q, 1e-6, 1); q = q / q.sum(1, keepdims=True)
    F['sit_entropy'] = (-(q * np.log(q)).sum(1)).astype(np.float32)
    return F


def attach(F, sd, X, prefix_cols=None):
    key = pd.MultiIndex.from_arrays([sd['pitcher_id'].values,
                                     sd['balls_before'].clip(0, 3).values,
                                     sd['strikes_before'].clip(0, 2).values,
                                     sd['batter_hand'].values])
    add = F.reindex(key)
    cols = prefix_cols if prefix_cols else list(F.columns)
    for c in cols:
        X[c] = add[c].values
    return X


if __name__ == '__main__':
    for s in [2021, 2022, 2023, 2024]:
        F = build_situational(s)
        F.to_parquet(CACHE / f'tkm_sit_{s}.parquet')
        print(s, F.shape)
    print(list(F.columns))
