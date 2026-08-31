"""
agent5_recent_season.py — season-level recency decomposition ("recent_season").

Distinct from the REJECTED form_ladder (game-level recency, outer -24.99,
outputs/200): this operates at SEASON granularity, where sample sizes per
cell are orders of magnitude larger than per-game cells, so it should be far
less noisy while still capturing "was this entity hot/cold last season"
signal that asof_dec's cur/hist split does NOT capture (hist = full career
average through end of last season, blending many years together; this adds
the MOST RECENT season in isolation as a separate signal).

Mechanism (pure feature engineering on top of AsofDecomposer2's already-fit
A_tr/A_val, no leakage):
  1. From df_tr_f (fold's own training slice) + A_tr (its cur_rate/cur_n
     computed by AsofDecomposer2, i.e. "in-season progressive value AS OF
     that pitch"), take the LAST row of each (pitcher_id, season) group.
     Since cur_rate/cur_n are cumulative WITHIN that season, the season's
     final row's cur_rate/cur_n literally equals "that season's full-season
     rate/n in isolation" (no admixture of prior-season data, since cur_*
     already subtracted the hist boundary).
  2. Build a lookup table keyed (pitcher_id, season) -> that season's final
     in-season rate/n.
  3. For a row in season s (train OR val, doesn't matter), look up key
     (pitcher_id, s-1) = "last calendar season's isolated rate". This uses
     ONLY df_tr_f (fold's own train slice) -- for df_val_f rows (season =
     val_season), s-1 = val_season-1 which IS inside df_tr_f by
     construction, so this works uniformly with zero special-casing and
     zero leakage (no val-season or future information touched).
  4. EB-shrink the raw last-season rate toward the row's own hist_rate
     (career-to-date-before-this-season average, already in A) with m=60,
     and also emit the raw rate/n and a "slope" (last-season EB minus career
     hist) as an explicit trend signal.
  5. If the entity has no row in season s-1 (debut, gap, or s=2019 with no
     2018 data), the lookup is NaN -> NaN raw features (GBDTs handle
     natively, consistent with how AsofDecomposer2 already emits NaN for
     cur_rate when cur_n < MIN_CUR).
"""
import numpy as np
import pandas as pd

EB_M = 60.0
MIN_N = 5

# (out_tag, cur_rate_col, hist_rate_col, cur_n_col)
SPECS = [
    ('succ', 'cs_p_succ_rate', 'cs_p_succ_hist', 'cs_pit_cur_n'),
    ('mid',  'cs_p_mid_rate',  'cs_p_mid_hist',  'cs_pit_cur_n'),
]


def _build_table(df_tr_f, A_tr, specs):
    tmp = pd.DataFrame({'pitcher_id': df_tr_f['pitcher_id'].values,
                         'season': df_tr_f['season'].values})
    for i, (tag, rc, hc, nc) in enumerate(specs):
        tmp[f'r{i}'] = A_tr[rc].values
        tmp[f'n{i}'] = A_tr[nc].values
    last = tmp.groupby(['pitcher_id', 'season']).tail(1).set_index(['pitcher_id', 'season'])
    return last


def _lookup_prev_season(df, table):
    key = pd.MultiIndex.from_arrays([df['pitcher_id'].values, df['season'].values - 1])
    r = table.reindex(key)
    r.index = df.index
    return r


def add_recent_season(df_tr_f, df_val_f, val_season, A_tr, A_val, X_tr_f, X_val_f, use_mid=False):
    specs = SPECS if use_mid else SPECS[:1]
    table = _build_table(df_tr_f, A_tr, specs)
    R_tr = _lookup_prev_season(df_tr_f, table)
    R_val = _lookup_prev_season(df_val_f, table)

    out_tr = pd.DataFrame(index=X_tr_f.index)
    out_val = pd.DataFrame(index=X_val_f.index)

    for out_df, R, A in [(out_tr, R_tr, A_tr), (out_val, R_val, A_val)]:
        for i, (tag, rc, hc, nc) in enumerate(specs):
            last_rate = R[f'r{i}'].values.astype(np.float64)
            last_n = np.nan_to_num(R[f'n{i}'].values.astype(np.float64))
            hist_rate = A[hc].values.astype(np.float64)
            fb = np.where(np.isfinite(hist_rate), hist_rate, 0.5)
            last_rate_valid = np.where(last_n >= MIN_N, last_rate, np.nan)
            eb = (np.nan_to_num(last_n * np.nan_to_num(last_rate_valid, nan=fb)) + EB_M * fb) / (last_n + EB_M)
            out_df[f'rs_{tag}_last_rate'] = last_rate_valid.astype(np.float32)
            out_df[f'rs_{tag}_last_n'] = last_n.astype(np.float32)
            out_df[f'rs_{tag}_last_eb'] = eb.astype(np.float32)
            out_df[f'rs_{tag}_slope'] = (eb - fb).astype(np.float32)

    return pd.concat([X_tr_f, out_tr], axis=1), pd.concat([X_val_f, out_val], axis=1)
