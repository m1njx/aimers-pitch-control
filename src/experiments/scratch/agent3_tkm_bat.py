"""agent3_tkm_bat.py — batter-side trackman features via the recovered batter-ID mapping.

Idea: `asof_batter_success_rate` (an OUTCOME statistic) died after 2022. But how pitchers
*approach* a batter is a behavioural signal that may survive the regime break:
pitchers nibble (fewer fastballs, more breaking balls, more edge-of-zone) against dangerous
hitters -> lower control-success probability.
"""
import numpy as np, pandas as pd, functools
from pathlib import Path

CACHE = Path('~/LG_data/scratch/agent3_cache')
TM_PATH = '~/LG_data/open/data/trackman_history.csv'
T = ['is_fb', 'is_br', 'is_os', 'rel_speed', 'induced_vert_break', 'horz_break', 'spin_rate']


@functools.lru_cache(maxsize=1)
def _tm():
    tm = pd.read_csv(TM_PATH, usecols=['season', 'batter_trackman_id', 'pitcher_trackman_id',
                                       'pitch_type_group', 'balls_before', 'strikes_before',
                                       'pitcher_hand', 'pitch_of_pa',
                                       'rel_speed', 'induced_vert_break', 'horz_break', 'spin_rate'])
    bm = pd.read_csv(CACHE / 'batter_map_raw.csv'); bm = bm[bm.margin >= 0.5]
    pm = pd.read_csv(CACHE / 'pitcher_map_raw.csv'); pm = pm[pm.margin >= 0.3]
    tm['bid'] = tm.batter_trackman_id.map(dict(zip(bm.tm_id, bm.batter_id)))
    tm['pid'] = tm.pitcher_trackman_id.map(dict(zip(pm.tm_id, pm.pitcher_id)))
    tm = tm.dropna(subset=['bid'])
    tm['bid'] = tm['bid'].astype(np.int32)
    tm['ph'] = (tm.pitcher_hand == 'Right').astype(np.int8) + 1
    tm['is_fb'] = (tm.pitch_type_group == 'fastball').astype(np.float32)
    tm['is_br'] = (tm.pitch_type_group == 'breaking').astype(np.float32)
    tm['is_os'] = (tm.pitch_type_group == 'offspeed').astype(np.float32)
    tm['b'] = tm.balls_before.clip(0, 3).astype(np.int8)
    tm['s'] = tm.strikes_before.clip(0, 2).astype(np.int8)
    return tm


def build_batter(as_of_season, k_sit=40.0, k_bat=80.0):
    t = _tm()
    t = t[t.season <= as_of_season]
    glob = t[T].mean()
    L0 = t.groupby(['b', 's', 'ph'])[T].mean()
    n1 = t.groupby('bid').size()
    L1 = t.groupby('bid')[T].mean()
    lam1 = (n1 / (n1 + k_bat)).values[:, None]
    L1s = pd.DataFrame(lam1 * L1.values + (1 - lam1) * glob.values[None, :], index=L1.index, columns=T)
    n2 = t.groupby(['bid', 'b', 's', 'ph']).size()
    L2 = t.groupby(['bid', 'b', 's', 'ph'])[T].mean()
    idx = L2.index
    bid = idx.get_level_values('bid')
    bsh = pd.MultiIndex.from_arrays([idx.get_level_values('b'), idx.get_level_values('s'),
                                     idx.get_level_values('ph')])
    base = L1s.reindex(bid).values + (L0.reindex(bsh).values - glob.values[None, :])
    lam2 = (n2 / (n2 + k_sit)).values[:, None]
    F = pd.DataFrame(lam2 * L2.values + (1 - lam2) * base, index=idx,
                     columns=[f'bsit_{c}' for c in T]).astype(np.float32)
    F['bsit_n'] = np.log1p(n2.values).astype(np.float32)
    F['bsit_fb_dev'] = (F['bsit_is_fb'].values - L1s['is_fb'].reindex(bid).values).astype(np.float32)
    # batter-career-level "respect" indices (constant per batter, still informative)
    F['bat_fb_career'] = L1s['is_fb'].reindex(bid).values.astype(np.float32)
    F['bat_velo_career'] = L1s['rel_speed'].reindex(bid).values.astype(np.float32)
    # how deep into PAs this batter goes (patience / difficulty to put away)
    pa = t.groupby('bid')['pitch_of_pa'].mean()
    F['bat_pitch_of_pa'] = pa.reindex(bid).values.astype(np.float32)
    return F


def build_matchup(as_of_season, k=25.0):
    """pitcher x batter historical pitch-mix, shrunk toward the pitcher's own mix."""
    t = _tm()
    t = t[(t.season <= as_of_season)].dropna(subset=['pid'])
    t['pid'] = t['pid'].astype(np.int32)
    pmix = t.groupby('pid')[['is_fb', 'is_br']].mean()
    n = t.groupby(['pid', 'bid']).size()
    mm = t.groupby(['pid', 'bid'])[['is_fb', 'is_br']].mean()
    lam = (n / (n + k)).values[:, None]
    base = pmix.reindex(mm.index.get_level_values('pid')).values
    F = pd.DataFrame(lam * mm.values + (1 - lam) * base, index=mm.index,
                     columns=['mu_is_fb', 'mu_is_br']).astype(np.float32)
    F['mu_n'] = np.log1p(n.values).astype(np.float32)
    F['mu_fb_dev'] = (F['mu_is_fb'].values - base[:, 0]).astype(np.float32)
    return F


def attach_bat(F, sd, X):
    key = pd.MultiIndex.from_arrays([sd['batter_id'].values,
                                     sd['balls_before'].clip(0, 3).values,
                                     sd['strikes_before'].clip(0, 2).values,
                                     sd['pitcher_hand'].values])
    add = F.reindex(key)
    for c in F.columns:
        X[c] = add[c].values
    return X


def attach_matchup(F, sd, X):
    key = pd.MultiIndex.from_arrays([sd['pitcher_id'].values, sd['batter_id'].values])
    add = F.reindex(key)
    for c in F.columns:
        X[c] = add[c].values
    return X


if __name__ == '__main__':
    for s in [2021, 2022, 2023, 2024]:
        build_batter(s).to_parquet(CACHE / f'tkm_bsit_{s}.parquet')
        build_matchup(s).to_parquet(CACHE / f'tkm_mu_{s}.parquet')
        print('done', s)
