"""
agent2_tkm_profile.py — Build PER-PITCHER trackman physical/arsenal profiles,
made possible by the pitcher_id <-> pitcher_trackman_id entity resolution
(agent2_link_pitchers.py).

The existing pipeline joins trackman only on a 7-key SITUATION signature
(month/dow/inning/top_bottom/count/outs), i.e. it never uses *who* threw the
pitch. That throws away essentially all of the value in trackman: release
mechanics, velocity, spin, and pitch-mix are pitcher-level properties.

Fold-safety: trackman rows are filtered to season <= as_of_season by caller.
The id mapping itself uses NO labels (pure identity resolution), and would be
computed identically at submission time from train.csv + trackman_history.csv.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config

MAP_P = '~/LG_data/scratch/agent2_map_pitcher.csv'
MAP_B = '~/LG_data/scratch/agent2_map_batter.csv'

PHYS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']


def load_pitcher_map(min_margin=0.15, min_n=20):
    m = pd.read_csv(MAP_P)
    ok = (m.margin >= min_margin) & (m.n_a >= min_n) & m.mutual
    hand_ok = ((m.hand_a == 1) & (m.hand_b == 'Left')) | ((m.hand_a == 2) & (m.hand_b == 'Right'))
    m = m[ok & hand_ok]
    return dict(zip(m.a_id.astype(int), m.b_id.astype(int)))


def _entropy(p):
    p = np.clip(p, 1e-9, 1)
    return -(p * np.log(p)).sum(axis=1)


class PitcherTrackmanProfile:
    """Per-pitcher aggregates + per-(pitcher,count) arsenal mix."""

    def __init__(self, pmap=None, eb_m=40.0):
        self.pmap = pmap if pmap is not None else load_pitcher_map()
        self.eb_m = eb_m

    def fit(self, tm: pd.DataFrame):
        tm = tm.copy()
        g = tm.groupby('pitcher_trackman_id')

        agg = {}
        for c in PHYS:
            agg[f'p_{c}_mean'] = (c, 'mean')
            agg[f'p_{c}_std'] = (c, 'std')
        prof = g.agg(**agg)
        prof['p_n'] = g.size()
        prof['p_speed_drop'] = prof['p_rel_speed_mean'] - prof['p_zone_speed_mean']

        # arsenal mix over pitch_type_group
        mix = pd.crosstab(tm['pitcher_trackman_id'], tm['pitch_type_group'])
        mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0)
        for c in ['fastball', 'breaking', 'offspeed', 'other']:
            if c not in mix.columns:
                mix[c] = 0.0
        mix = mix[['fastball', 'breaking', 'offspeed', 'other']].fillna(0.0)
        prof['p_mix_entropy'] = _entropy(mix.values)
        prof['p_arsenal_size'] = tm.groupby('pitcher_trackman_id')['tagged_pitch_type'].nunique()

        # fastball-only mechanics (release repeatability without pitch-type mixing)
        fb = tm[tm['pitch_type_group'] == 'fastball']
        gfb = fb.groupby('pitcher_trackman_id')
        fbagg = gfb.agg(p_fb_speed_mean=('rel_speed', 'mean'),
                        p_fb_speed_std=('rel_speed', 'std'),
                        p_fb_spin_mean=('spin_rate', 'mean'),
                        p_fb_ivb_mean=('induced_vert_break', 'mean'),
                        p_fb_hb_mean=('horz_break', 'mean'),
                        p_fb_relh_std=('rel_height', 'std'),
                        p_fb_rels_std=('rel_side', 'std'),
                        p_fb_ext_std=('extension', 'std'))
        br = tm[tm['pitch_type_group'] == 'breaking']
        gbr = br.groupby('pitcher_trackman_id')
        bragg = gbr.agg(p_br_spin_mean=('spin_rate', 'mean'),
                        p_br_hb_mean=('horz_break', 'mean'),
                        p_br_ivb_mean=('induced_vert_break', 'mean'),
                        p_br_speed_mean=('rel_speed', 'mean'))
        prof = prof.join(fbagg).join(bragg)
        prof['p_fb_br_speed_gap'] = prof['p_fb_speed_mean'] - prof['p_br_speed_mean']
        self.prof_ = prof

        # ---- per-(pitcher, balls, strikes) arsenal mix, EB-smoothed ----
        tm['_fb'] = (tm['pitch_type_group'] == 'fastball').astype(np.float32)
        tm['_br'] = (tm['pitch_type_group'] == 'breaking').astype(np.float32)
        tm['_os'] = (tm['pitch_type_group'] == 'offspeed').astype(np.float32)
        gc = tm.groupby(['pitcher_trackman_id', 'balls_before', 'strikes_before'])
        cm = gc.agg(n=('_fb', 'size'), fb=('_fb', 'mean'), br=('_br', 'mean'), os=('_os', 'mean'))
        # pitcher-level priors
        pri = tm.groupby('pitcher_trackman_id').agg(fb0=('_fb', 'mean'), br0=('_br', 'mean'),
                                                    os0=('_os', 'mean'))
        cm = cm.join(pri, on='pitcher_trackman_id')
        m = self.eb_m
        for k in ['fb', 'br', 'os']:
            cm[f'pc_{k}_rate'] = (cm['n'] * cm[k] + m * cm[f'{k}0']) / (cm['n'] + m)
        cm['pc_n'] = cm['n']
        self.count_mix_ = cm[['pc_fb_rate', 'pc_br_rate', 'pc_os_rate', 'pc_n']]

        # ---- per-(pitcher, batter_hand) fastball rate ----
        gh = tm.groupby(['pitcher_trackman_id', 'batter_hand'])
        hm = gh.agg(n=('_fb', 'size'), fb=('_fb', 'mean'), br=('_br', 'mean'))
        hm = hm.join(pri, on='pitcher_trackman_id')
        hm['ph_fb_rate'] = (hm['n'] * hm['fb'] + m * hm['fb0']) / (hm['n'] + m)
        hm['ph_br_rate'] = (hm['n'] * hm['br'] + m * hm['br0']) / (hm['n'] + m)
        self.hand_mix_ = hm[['ph_fb_rate', 'ph_br_rate']]
        return self

    def transform(self, df: pd.DataFrame, groups=('prof', 'count', 'hand')) -> pd.DataFrame:
        tid = df['pitcher_id'].map(self.pmap)
        out = pd.DataFrame(index=df.index)
        if 'prof' in groups:
            P = self.prof_.reindex(tid.values)
            P.index = df.index
            out = pd.concat([out, P], axis=1)
        if 'count' in groups:
            key = pd.MultiIndex.from_arrays([tid.values,
                                             df['balls_before'].fillna(0).astype(int).values,
                                             df['strikes_before'].fillna(0).astype(int).values])
            C = self.count_mix_.reindex(key)
            C.index = df.index
            out = pd.concat([out, C], axis=1)
        if 'hand' in groups:
            bh = df['batter_hand'].map({1: 'Left', 2: 'Right'})
            key = pd.MultiIndex.from_arrays([tid.values, bh.values])
            H = self.hand_mix_.reindex(key)
            H.index = df.index
            out = pd.concat([out, H], axis=1)
        out['p_mapped'] = tid.notna().astype(int).values
        return out.astype(np.float32)


def load_trackman_upto(as_of_season):
    tm = pd.read_csv(config.TRACKMAN_PATH)
    return tm[tm['season'] <= as_of_season].copy()


if __name__ == '__main__':
    pmap = load_pitcher_map()
    print(f"mapped pitchers: {len(pmap)}")
    df = pd.read_csv(config.TRAIN_PATH, usecols=['season', 'pitcher_id', 'batter_hand',
                                                 'balls_before', 'strikes_before'])
    print("coverage of train rows by season:")
    df['m'] = df['pitcher_id'].isin(pmap).astype(int)
    print(df.groupby('season')['m'].mean())
    tm = load_trackman_upto(2021)
    p = PitcherTrackmanProfile(pmap).fit(tm)
    X = p.transform(df[df.season == 2022])
    print(X.shape)
    print(X.describe().T.to_string())
    print("null frac:\n", X.isna().mean().sort_values(ascending=False).head(10))
