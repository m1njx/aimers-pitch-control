"""
agent2_recover_labels.py — Recover the HIDDEN per-pitch outcome labels that the
organiser never released, by differencing their cumulative `asof_*` counters.

Verified facts:
  * `asof_pitcher_n` increases by EXACTLY 1 between consecutive rows of the same
    pitcher (1,474,300 / 1,474,300 non-first rows). So the asof counters run over
    exactly the rows present in train.csv, with no gaps.
  * Therefore  d( asof_n * asof_rate )  between consecutive rows of a pitcher is
    the 0/1 indicator of that outcome for the PREVIOUS row.

This recovers, for essentially every train row, five labels the competition
never gave us:
    is_reverse, is_middle, is_ball, is_strike           (pitcher counters)
    fastball / breaking / offspeed                      (pitchmix counters)
plus a consistency check against the released `control_success`.

These are POST-pitch outcomes so they can never be model inputs. Their value is
(a) revealing how control_success is actually defined, and
(b) providing auxiliary targets / a full-coverage pitch-type label for building
    a legal "expected pitch type" pre-pitch feature.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config

PITCHER_SPECS = [('success', 'asof_pitcher_success_rate', 'asof_pitcher_n'),
                 ('reverse', 'asof_pitcher_reverse_rate', 'asof_pitcher_n'),
                 ('middle', 'asof_pitcher_middle_rate', 'asof_pitcher_n'),
                 ('ball', 'asof_pitcher_ball_rate', 'asof_pitcher_n'),
                 ('strike', 'asof_pitcher_strike_rate', 'asof_pitcher_n'),
                 ('fastball', 'asof_pitcher_fastball_rate', 'asof_pitcher_pitchmix_n'),
                 ('breaking', 'asof_pitcher_breaking_rate', 'asof_pitcher_pitchmix_n'),
                 ('offspeed', 'asof_pitcher_offspeed_rate', 'asof_pitcher_pitchmix_n')]
BATTER_SPECS = [('bat_success', 'asof_batter_success_rate', 'asof_batter_n'),
                ('bat_middle', 'asof_batter_middle_rate', 'asof_batter_n')]


def recover(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of recovered 0/1 labels aligned to df's rows."""
    out = pd.DataFrame(index=df.index)
    for ent, specs in [('pitcher_id', PITCHER_SPECS), ('batter_id', BATTER_SPECS)]:
        gid = df[ent]
        for name, rate_col, den_col in specs:
            n = df[den_col].astype(np.float64).values
            cnt = n * df[rate_col].fillna(0).astype(np.float64).values
            dn = pd.Series(n).groupby(gid.values).diff().values
            dc = pd.Series(cnt).groupby(gid.values).diff().values
            # increment at row i describes row i-1 -> shift back by one within group
            lab = np.where(np.abs(dn - 1) < 1e-6, np.round(dc), np.nan)
            s = pd.Series(lab, index=df.index)
            out[f'lab_{name}'] = s.groupby(gid.values).shift(-1)
    return out


if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_PATH)
    L = recover(df)
    y = df[config.TARGET_COL].values
    print("coverage (non-null) of recovered labels:")
    print(L.notna().mean().round(4).to_string())
    m = L['lab_success'].notna().values
    agree = (L['lab_success'].values[m] == y[m]).mean()
    print(f"\nRECOVERY CHECK: lab_success == control_success on {m.sum():,} rows -> "
          f"agreement = {agree:.6f}")
    vals = pd.unique(L['lab_success'].dropna())
    print(f"lab_success unique values: {np.sort(vals)[:6]} ...")

    print("\n=== joint structure of the recovered outcome flags ===")
    sub = L.dropna()
    print(f"rows with all labels: {len(sub):,}")
    for c in sub.columns:
        print(f"  {c:<16} mean={sub[c].mean():.4f}")
    print("\nsum checks:")
    print("  reverse+middle+success mean:",
          (sub.lab_reverse + sub.lab_middle + sub.lab_success).mean(),
          " value counts:", (sub.lab_reverse + sub.lab_middle + sub.lab_success).value_counts().head(4).to_dict())
    print("  ball+strike mean:", (sub.lab_ball + sub.lab_strike).mean(),
          (sub.lab_ball + sub.lab_strike).value_counts().head(4).to_dict())
    print("  fastball+breaking+offspeed:",
          (sub.lab_fastball + sub.lab_breaking + sub.lab_offspeed).value_counts().head(4).to_dict())

    print("\n=== control_success crosstabs ===")
    for c in ['lab_reverse', 'lab_middle', 'lab_ball', 'lab_strike',
              'lab_fastball', 'lab_breaking', 'lab_offspeed']:
        ct = pd.crosstab(sub[c], sub['lab_success'], normalize='index')
        print(f"\n P(success | {c}):")
        print(ct.round(4).to_string())

    L.to_csv('~/LG_data/scratch/agent2_recovered_labels.csv.gz',
             index=False, compression='gzip')
    print("\nsaved recovered labels")
