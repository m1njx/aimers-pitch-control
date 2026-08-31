"""agent3_tkm.py — trackman pitcher-level feature builder using the recovered ID map.

Features are computed only from trackman seasons <= as_of_season (no leakage).
"""
import pandas as pd, numpy as np, functools
from pathlib import Path

CACHE = Path('~/LG_data/scratch/agent3_cache')
TM_PATH = '~/LG_data/open/data/trackman_history.csv'

PHYS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']


@functools.lru_cache(maxsize=1)
def load_tm():
    tm = pd.read_csv(TM_PATH, usecols=['season', 'pitcher_trackman_id', 'pitch_type_group',
                                       'auto_pitch_type', 'balls_before', 'strikes_before',
                                       'pitch_of_pa', 'batter_hand', 'pitcher_hand'] + PHYS)
    return tm


@functools.lru_cache(maxsize=1)
def load_map(margin_thr=0.3):
    res = pd.read_csv(CACHE / 'pitcher_map_raw.csv')
    res = res[res.margin >= margin_thr]
    return dict(zip(res.tm_id, res.pitcher_id))


def _agg_block(g, prefix):
    """Aggregate physical stats for a grouped frame."""
    out = g[PHYS].agg(['mean', 'std'])
    out.columns = [f'{prefix}{a}_{b}' for a, b in out.columns]
    out[f'{prefix}n'] = g.size()
    return out


def build_pitcher_features(as_of_season: int, margin_thr: float = 0.3) -> pd.DataFrame:
    tm = load_tm()
    m = load_map(margin_thr)
    t = tm[tm.season <= as_of_season].copy()
    t['pid'] = t.pitcher_trackman_id.map(m)
    t = t.dropna(subset=['pid'])
    t['pid'] = t['pid'].astype(int)

    feats = []
    # --- career-level, all pitches ---
    g = t.groupby('pid')
    feats.append(_agg_block(g, 'tk_all_'))

    # --- fastball only (release consistency = command proxy) ---
    tf = t[t.pitch_type_group == 'fastball']
    feats.append(_agg_block(tf.groupby('pid'), 'tk_fb_'))

    # --- breaking only ---
    tb = t[t.pitch_type_group == 'breaking']
    feats.append(_agg_block(tb.groupby('pid'), 'tk_br_'))

    # --- last season only (recent form) ---
    last = t[t.season == as_of_season]
    feats.append(_agg_block(last.groupby('pid'), 'tk_last_'))

    F = pd.concat(feats, axis=1)

    # --- pitch mix ---
    mix = pd.crosstab(t.pid, t.pitch_type_group, normalize='index')
    mix.columns = [f'tk_mix_{c}' for c in mix.columns]
    F = F.join(mix)
    mix_l = pd.crosstab(last.pid, last.pitch_type_group, normalize='index')
    mix_l.columns = [f'tk_lastmix_{c}' for c in mix_l.columns]
    F = F.join(mix_l)

    # --- repertoire size ---
    F['tk_n_types'] = t.groupby('pid')['auto_pitch_type'].nunique()
    F['tk_n_types_last'] = last.groupby('pid')['auto_pitch_type'].nunique()

    # --- derived ---
    F['tk_velo_drop'] = F['tk_all_rel_speed_mean'] - F['tk_all_zone_speed_mean']
    F['tk_fb_velo_delta_last'] = F['tk_last_rel_speed_mean'] - F['tk_all_rel_speed_mean']
    F['tk_break_total'] = np.hypot(F['tk_all_induced_vert_break_mean'], F['tk_all_horz_break_mean'])
    F['tk_release_scatter'] = np.hypot(F['tk_all_rel_height_std'], F['tk_all_rel_side_std'])
    F['tk_fb_release_scatter'] = np.hypot(F['tk_fb_rel_height_std'], F['tk_fb_rel_side_std'])
    # fastball vs offspeed velocity separation
    to = t[t.pitch_type_group == 'offspeed'].groupby('pid')['rel_speed'].mean()
    F['tk_velo_sep_fb_os'] = F['tk_fb_rel_speed_mean'] - to.reindex(F.index)
    tbr = t[t.pitch_type_group == 'breaking'].groupby('pid')['rel_speed'].mean()
    F['tk_velo_sep_fb_br'] = F['tk_fb_rel_speed_mean'] - tbr.reindex(F.index)

    # --- count-dependent approach ---
    t['is_fb'] = (t.pitch_type_group == 'fastball').astype(np.float32)
    beh = t[t.balls_before >= 2].groupby('pid')['is_fb'].mean().rename('tk_fb_rate_behind')
    ahd = t[t.strikes_before >= 2].groupby('pid')['is_fb'].mean().rename('tk_fb_rate_ahead')
    F = F.join(beh).join(ahd)
    F['tk_fb_rate_gap'] = F['tk_fb_rate_behind'] - F['tk_fb_rate_ahead']
    # first-pitch vs later
    fp = t[t.pitch_of_pa == 1].groupby('pid')['is_fb'].mean().rename('tk_fb_rate_p1')
    F = F.join(fp)

    # --- platoon split: velo/movement vs LHB/RHB ---
    for hnd, tag in [('Left', 'vL'), ('Right', 'vR')]:
        sub = t[t.batter_hand == hnd].groupby('pid')
        F[f'tk_{tag}_fb_rate'] = sub['is_fb'].mean()
        F[f'tk_{tag}_relside'] = sub['rel_side'].mean()

    F.index.name = 'pitcher_id'
    return F.astype(np.float32)


if __name__ == '__main__':
    import sys
    for s in [2021, 2022, 2023, 2024]:
        F = build_pitcher_features(s)
        F.to_parquet(CACHE / f'tkm_pfeat_{s}.parquet')
        print(f'as_of={s}: {F.shape[0]} pitchers x {F.shape[1]} features')
    print(list(F.columns))
