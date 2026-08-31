"""agent3_tkm_fatigue.py — in-game workload / fatigue expectations from trackman.

The main table has NO within-game pitch counter (and test rows may not use row order),
but trackman has full games. So we can learn, per pitcher, the *expected* in-game
workload state at a given inning:  E[pitches thrown so far | pitcher, inning],
E[times through the order | pitcher, inning], P(starter), pitches per appearance.
These are pure train-side artifacts keyed on (pitcher_id, inning) -> legal for test rows.
"""
import numpy as np, pandas as pd, functools
from pathlib import Path

CACHE = Path('~/LG_data/scratch/agent3_cache')
TM_PATH = '~/LG_data/open/data/trackman_history.csv'


@functools.lru_cache(maxsize=1)
def _tm():
    tm = pd.read_csv(TM_PATH, usecols=['season', 'trackman_game_id', 'pitch_no', 'inning',
                                       'top_bottom', 'pitcher_trackman_id', 'batter_trackman_id',
                                       'pitch_of_pa', 'rel_speed', 'pitch_type_group'])
    pm = pd.read_csv(CACHE / 'pitcher_map_raw.csv'); pm = pm[pm.margin >= 0.3]
    tm['pid'] = tm.pitcher_trackman_id.map(dict(zip(pm.tm_id, pm.pitcher_id)))
    tm = tm.dropna(subset=['pid'])
    tm['pid'] = tm['pid'].astype(np.int32)
    tm = tm.sort_values(['trackman_game_id', 'pitcher_trackman_id', 'pitch_no'])
    g = tm.groupby(['trackman_game_id', 'pitcher_trackman_id'], sort=False)
    tm['n_so_far'] = g.cumcount()
    # new plate appearance whenever pitch_of_pa == 1
    tm['new_pa'] = (tm.pitch_of_pa == 1).astype(np.int32)
    tm['pa_so_far'] = g['new_pa'].cumsum()
    tm['app_pitches'] = g['pitch_no'].transform('size')
    tm['is_start'] = (g['inning'].transform('min') == 1).astype(np.int8)
    return tm


def build_fatigue(as_of_season, k=60.0):
    t = _tm()
    t = t[t.season <= as_of_season]
    t = t.assign(inn=t.inning.clip(1, 10))
    glob = t.groupby('inn')[['n_so_far', 'pa_so_far', 'rel_speed']].mean()
    n = t.groupby(['pid', 'inn']).size()
    cell = t.groupby(['pid', 'inn'])[['n_so_far', 'pa_so_far', 'rel_speed']].mean()
    base = glob.reindex(cell.index.get_level_values('inn')).values
    lam = (n / (n + k)).values[:, None]
    F = pd.DataFrame(lam * cell.values + (1 - lam) * base, index=cell.index,
                     columns=['fat_pitches_so_far', 'fat_pa_so_far', 'fat_velo_at_inn']).astype(np.float32)
    pit = t.groupby('pid')
    app = pit['app_pitches'].mean()
    st = pit['is_start'].mean()
    F['fat_app_pitches'] = app.reindex(F.index.get_level_values('pid')).values.astype(np.float32)
    F['fat_p_starter'] = st.reindex(F.index.get_level_values('pid')).values.astype(np.float32)
    F['fat_frac_of_app'] = (F['fat_pitches_so_far'] / F['fat_app_pitches'].clip(lower=1)).astype(np.float32)
    F['fat_times_thru'] = (F['fat_pa_so_far'] / 9.0).astype(np.float32)
    # in-appearance velocity decay: mean velo at this inning minus pitcher's overall velo
    ov = pit['rel_speed'].mean()
    F['fat_velo_dev'] = (F['fat_velo_at_inn'] - ov.reindex(F.index.get_level_values('pid')).values).astype(np.float32)
    return F


def attach_fat(F, sd, X):
    key = pd.MultiIndex.from_arrays([sd['pitcher_id'].values, sd['inning'].clip(1, 10).values])
    add = F.reindex(key)
    for c in F.columns:
        X[c] = add[c].values
    return X


if __name__ == '__main__':
    for s in [2021, 2022, 2023, 2024]:
        F = build_fatigue(s)
        F.to_parquet(CACHE / f'tkm_fat_{s}.parquet')
        print(s, F.shape)
    print(list(F.columns))
