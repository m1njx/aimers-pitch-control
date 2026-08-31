"""agent3_enc.py — era-adjusted, recency-weighted, shrunk group encodings.

All artifacts are built from the FOLD'S TRAIN SPLIT ONLY (seasons <= fold_max_season).
Core idea: raw asof_* rates are career-expanding means contaminated by the league
base-rate era drift (0.5647 -> 0.4861). Subtracting the league rate of the season in
which each pitch happened yields an era-neutral pitcher/batter skill estimate.
"""
import numpy as np, pandas as pd


def _league_rates(season, y):
    d = pd.DataFrame({'s': season, 'y': y})
    return d.groupby('s')['y'].mean()


def era_skill_table(keys, season, y, as_of, decay=0.8, k_shrink=200.0, rs=None):
    """keys: array-like group id per row. Returns Series index=key -> era-adjusted skill."""
    if rs is None:
        rs = _league_rates(season, y)
    d = pd.DataFrame({'g': keys, 's': season, 'y': y})
    d['r'] = d['s'].map(rs).values
    d['w'] = decay ** (as_of - d['s'].values)
    d['num'] = d['w'] * (d['y'] - d['r'])
    d['den'] = d['w']
    agg = d.groupby('g')[['num', 'den']].sum()
    return (agg['num'] / (agg['den'] + k_shrink)), agg['den']


def build_encodings(sd_tr, y_tr, s_tr, as_of, cfg=None):
    """Returns dict of {name: (mapping_series, key_builder_fn)}"""
    cfg = cfg or {}
    rs = _league_rates(s_tr, y_tr)
    out = {}

    def add(name, keys_tr, keyfn, decay=0.8, k=200.0):
        sk, den = era_skill_table(keys_tr, s_tr, y_tr, as_of, decay, k, rs)
        out[name] = (sk, keyfn, den)

    add('pit_skill', sd_tr['pitcher_id'].values, lambda sd: sd['pitcher_id'].values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_pit', 200.0))
    add('pit_skill_flat', sd_tr['pitcher_id'].values, lambda sd: sd['pitcher_id'].values,
        decay=1.0, k=cfg.get('k_pit', 200.0))
    add('bat_skill', sd_tr['batter_id'].values, lambda sd: sd['batter_id'].values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_bat', 200.0))

    # pitcher x count-state
    ktr = (sd_tr['pitcher_id'].astype(str) + '|' + sd_tr['balls_before'].astype(str) + '_' +
           sd_tr['strikes_before'].astype(str)).values
    add('pit_count_skill', ktr,
        lambda sd: (sd['pitcher_id'].astype(str) + '|' + sd['balls_before'].astype(str) + '_' +
                    sd['strikes_before'].astype(str)).values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_pc', 100.0))

    # pitcher x batter hand (platoon)
    ktr = (sd_tr['pitcher_id'].astype(str) + '|' + sd_tr['batter_hand'].astype(str)).values
    add('pit_hand_skill', ktr,
        lambda sd: (sd['pitcher_id'].astype(str) + '|' + sd['batter_hand'].astype(str)).values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_ph', 150.0))

    # park (home team) effect: home team pitches when top_bottom == 'T'
    def park(sd):
        return np.where(sd['top_bottom'].values == 'T',
                        sd['pitcher_team_id'].values, sd['batter_team_id'].values)
    add('park_skill', park(sd_tr), park, decay=cfg.get('decay', 0.8), k=cfg.get('k_park', 500.0))

    # pitcher team (catcher framing / staff) effect
    add('pteam_skill', sd_tr['pitcher_team_id'].values, lambda sd: sd['pitcher_team_id'].values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_park', 500.0))

    # batter team
    add('bteam_skill', sd_tr['batter_team_id'].values, lambda sd: sd['batter_team_id'].values,
        decay=cfg.get('decay', 0.8), k=cfg.get('k_park', 500.0))
    return out


def apply_encodings(enc, sd, X, with_n=True):
    for name, (sk, keyfn, den) in enc.items():
        keys = keyfn(sd)
        X[name] = pd.Series(keys).map(sk).fillna(0.0).values.astype(np.float32)
        if with_n and name in ('pit_skill', 'bat_skill', 'pit_count_skill'):
            X[name + '_n'] = np.log1p(pd.Series(keys).map(den).fillna(0.0).values).astype(np.float32)
    return X
