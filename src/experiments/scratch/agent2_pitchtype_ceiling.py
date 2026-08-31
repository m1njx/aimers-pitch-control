"""
agent2_pitchtype_ceiling.py — DIAGNOSTIC: how much of control_success is
explained by WHICH PITCH was thrown?

Uses the resolved pitcher/batter id mapping to match individual train rows to
individual trackman rows on
  (season, month, dow, inning, top_bottom, balls, strikes, outs,
   pitcher_trackman_id, batter_trackman_id)
Only rows where the key is unique on BOTH sides are accepted (no ambiguity).

This is a DIAGNOSTIC ONLY - the matched pitch's own trackman measurement can
never be used as a model feature (competition rule 6). The purpose is to size
the prize: if pitch type / velocity strongly moves control_success, then an
auxiliary "expected pitch type" model built from pre-pitch features is worth
building; if not, the whole trackman-arsenal direction is capped.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config
from agent2_tkm_profile import load_pitcher_map
from agent2_link_teams import TB_MAP

KEYS = ['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'balls_before', 'strikes_before', 'outs_before', 'ptid', 'btid']


def load_batter_map(min_margin=0.15, min_n=20):
    m = pd.read_csv('~/LG_data/scratch/agent2_map_batter.csv')
    ok = (m.margin >= min_margin) & (m.n_a >= min_n) & m.mutual
    hand_ok = ((m.hand_a == 1) & (m.hand_b == 'Left')) | ((m.hand_a == 2) & (m.hand_b == 'Right'))
    m = m[ok & hand_ok]
    return dict(zip(m.a_id.astype(int), m.b_id.astype(int)))


if __name__ == '__main__':
    pmap = load_pitcher_map(); bmap = load_batter_map()
    df = pd.read_csv(config.TRAIN_PATH)
    tm = pd.read_csv(config.TRACKMAN_PATH)
    tm['top_bottom'] = tm['top_bottom'].map(TB_MAP)
    df['ptid'] = df['pitcher_id'].map(pmap)
    df['btid'] = df['batter_id'].map(bmap)
    tm = tm.rename(columns={'pitcher_trackman_id': 'ptid', 'batter_trackman_id': 'btid'})
    d = df.dropna(subset=['ptid', 'btid']).copy()
    d['ptid'] = d['ptid'].astype(np.int64); d['btid'] = d['btid'].astype(np.int64)
    print(f"train rows with both ids resolved: {len(d):,} / {len(df):,}")

    # keep only keys unique on both sides
    a = d.groupby(KEYS).size().rename('na')
    b = tm.groupby(KEYS).size().rename('nb')
    j = pd.concat([a, b], axis=1).dropna()
    uniq = j[(j.na == 1) & (j.nb == 1)].index
    print(f"unique-on-both keys: {len(uniq):,}")

    dm = d.set_index(KEYS).loc[uniq].reset_index()
    tmm = tm.set_index(KEYS).loc[uniq].reset_index()
    M = dm[KEYS + [config.TARGET_COL]].join(
        tmm[['pitch_type_group', 'tagged_pitch_type', 'rel_speed', 'spin_rate',
             'induced_vert_break', 'horz_break', 'pitch_of_pa']].reset_index(drop=True))
    print(f"matched rows: {len(M):,} ({len(M)/len(df)*100:.1f}% of train)")

    y = M[config.TARGET_COL]
    print("\n=== control_success by pitch_type_group (matched sample) ===")
    print(M.groupby('pitch_type_group')[config.TARGET_COL].agg(['size', 'mean']).to_string())
    print(f"overall = {y.mean():.4f}")
    print("\n=== by tagged_pitch_type (n>=3000) ===")
    g = M.groupby('tagged_pitch_type')[config.TARGET_COL].agg(['size', 'mean'])
    print(g[g['size'] >= 3000].sort_values('mean').to_string())
    print("\n=== by pitch_type_group x count ===")
    print(M.pivot_table(index=['balls_before', 'strikes_before'],
                        columns='pitch_type_group', values=config.TARGET_COL,
                        aggfunc='mean').round(4).to_string())
    print("\n=== by rel_speed decile (fastballs only) ===")
    fb = M[M.pitch_type_group == 'fastball'].copy()
    fb['q'] = pd.qcut(fb.rel_speed, 10, labels=False, duplicates='drop')
    print(fb.groupby('q')[config.TARGET_COL].agg(['size', 'mean']).to_string())
    print("\n=== by pitch_of_pa ===")
    print(M.groupby(M.pitch_of_pa.clip(upper=8))[config.TARGET_COL].agg(['size', 'mean']).to_string())

    # variance explained by pitch type alone
    grp = M.groupby('tagged_pitch_type')[config.TARGET_COL].transform('mean')
    var_expl = grp.var()
    print(f"\nVar explained by tagged_pitch_type alone = {var_expl:.6f} "
          f"(=> skill points if perfectly known: {1e5*var_expl/0.25:.0f})")
    M.to_parquet('~/LG_data/scratch/agent2_matched_pitches.parquet')
