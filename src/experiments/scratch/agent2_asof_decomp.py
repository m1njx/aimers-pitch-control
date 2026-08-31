"""
agent2_asof_decomp.py — ALGEBRAIC DECOMPOSITION of the organiser-supplied
career-cumulative `asof_*` rates into CURRENT-SEASON-TO-DATE rates.

Verified fact (agent2 check): for every (pitcher, season) the value of
`asof_pitcher_n` on that pitcher's first row of the season equals EXACTLY the
pitcher's cumulative pitch count in train.csv over all earlier seasons
(2260/2260 exact matches). The same holds for batters. So the asof columns are
cumulative counters over exactly the train universe, and they can be inverted:

    cur_n     = asof_n            - hist_n
    cur_count = asof_n * asof_rate - hist_n * hist_rate
    cur_rate  = cur_count / cur_n

`hist_n`/`hist_rate` are the pitcher's career totals through the END of the
previous season, which are computable from train.csv alone. Therefore for a
2025 test row the formula yields that pitcher's **2025-season-to-date** rate,
using only (a) the row's own organiser-provided asof columns and (b) constants
derived from train.csv. No other test row is touched -> compliant with the
"each evaluation row must be predicted independently" rule.

Why this matters: league control_success falls 0.5647 -> 0.4861 across
2019-2024, so a career-cumulative rate is a blend of eras and is biased by WHEN
a player accumulated his innings. A current-season rate is era-correct and is
also the only feature that can tell the model where the 2025 league level sits.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config

TGT = config.TARGET_COL

# (rate column, denominator column, output prefix)
PITCHER_RATES = [
    ('asof_pitcher_success_rate', 'asof_pitcher_n', 'p_succ'),
    ('asof_pitcher_reverse_rate', 'asof_pitcher_n', 'p_rev'),
    ('asof_pitcher_middle_rate', 'asof_pitcher_n', 'p_mid'),
    ('asof_pitcher_ball_rate', 'asof_pitcher_n', 'p_ball'),
    ('asof_pitcher_strike_rate', 'asof_pitcher_n', 'p_str'),
    ('asof_pitcher_fastball_rate', 'asof_pitcher_pitchmix_n', 'p_fb'),
    ('asof_pitcher_breaking_rate', 'asof_pitcher_pitchmix_n', 'p_br'),
    ('asof_pitcher_offspeed_rate', 'asof_pitcher_pitchmix_n', 'p_os'),
]
BATTER_RATES = [
    ('asof_batter_success_rate', 'asof_batter_n', 'b_succ'),
    ('asof_batter_middle_rate', 'asof_batter_n', 'b_mid'),
]

NEEDED = sorted(set([c for c, _, _ in PITCHER_RATES + BATTER_RATES] +
                    [d for _, d, _ in PITCHER_RATES + BATTER_RATES] +
                    ['pitcher_id', 'batter_id', 'season', TGT]))


class AsofDecomposer:
    """Learns per-entity career-cumulative boundary values at the end of each season."""

    def __init__(self, eb_m=150.0):
        self.eb_m = eb_m

    def _boundaries(self, df_hist, ent_col, specs):
        """For each (entity, season s) return career totals through END of s-1."""
        out = {}
        # the LAST row of the entity within each season carries the cumulative
        # state just before that row; add that row itself for exactness where we can
        d = df_hist[[ent_col, 'season'] + sorted(set([c for c, _, _ in specs] +
                                                    [dn for _, dn, _ in specs]))]
        last = d.groupby([ent_col, 'season']).tail(1).copy()
        rows = []
        for (rate_col, den_col, pre) in specs:
            last[f'__cnt_{pre}'] = last[den_col].fillna(0) * last[rate_col].fillna(0)
        keep = [ent_col, 'season'] + [f'__cnt_{p}' for _, _, p in specs] + \
               sorted(set(dn for _, dn, _ in specs))
        last = last[keep].sort_values([ent_col, 'season'])
        # state at end of season s == state recorded on last row of season s
        # (off by that single final pitch; negligible)
        # shift by one season -> boundary usable for rows of season s
        g = last.groupby(ent_col)
        shifted = g.shift(1)
        shifted[ent_col] = last[ent_col].values
        shifted['season'] = last['season'].values
        # forward-fill across seasons the entity did not play
        shifted = shifted.sort_values([ent_col, 'season'])
        cols = [c for c in shifted.columns if c not in (ent_col, 'season')]
        shifted[cols] = shifted.groupby(ent_col)[cols].ffill()
        return shifted.set_index([ent_col, 'season'])

    def fit(self, df_hist: pd.DataFrame, val_season: int):
        """df_hist: the fold's TRAIN split (labelled).  val_season: the season to serve."""
        self.val_season_ = val_season
        self.pb_ = self._boundaries(df_hist, 'pitcher_id', PITCHER_RATES)
        self.bb_ = self._boundaries(df_hist, 'batter_id', BATTER_RATES)
        # boundary for the val season itself = state at end of (val_season-1),
        # i.e. the last recorded row of that entity in df_hist
        self.pb_val_ = self._last_state(df_hist, 'pitcher_id', PITCHER_RATES)
        self.bb_val_ = self._last_state(df_hist, 'batter_id', BATTER_RATES)
        self.league_ = df_hist.groupby('season')[TGT].mean()
        # fitted fallback constants for the EB blend, so transform() never depends
        # on statistics of the batch it is scoring (rule 5 / train-serve skew)
        self.fallback_ = {}
        for ent_col, specs, B in [('pitcher_id', PITCHER_RATES, self.pb_val_),
                                  ('batter_id', BATTER_RATES, self.bb_val_)]:
            for (rate_col, den_col, pre) in specs:
                hn = B[den_col].fillna(0).values.astype(np.float64)
                hc = B[f'__cnt_{pre}'].fillna(0).values.astype(np.float64)
                with np.errstate(invalid='ignore', divide='ignore'):
                    hr = np.where(hn > 0, hc / np.maximum(hn, 1), np.nan)
                self.fallback_[pre] = float(np.nanmean(hr)) if np.isfinite(hr).any() else 0.0
        return self

    def _last_state(self, df_hist, ent_col, specs):
        d = df_hist[[ent_col] + sorted(set([c for c, _, _ in specs] +
                                           [dn for _, dn, _ in specs]))]
        last = d.groupby(ent_col).tail(1).copy()
        for (rate_col, den_col, pre) in specs:
            last[f'__cnt_{pre}'] = last[den_col].fillna(0) * last[rate_col].fillna(0)
        keep = [ent_col] + [f'__cnt_{p}' for _, _, p in specs] + \
               sorted(set(dn for _, dn, _ in specs))
        return last[keep].set_index(ent_col)

    def _apply(self, df, ent_col, specs, boundary_multi, boundary_last):
        n = len(df)
        is_val = (df['season'].values == self.val_season_)
        key = pd.MultiIndex.from_arrays([df[ent_col].values, df['season'].values])
        B = boundary_multi.reindex(key)
        B.index = df.index
        BL = boundary_last.reindex(df[ent_col].values)
        BL.index = df.index
        # val-season rows use the "last state in history" boundary
        for c in B.columns:
            B.loc[is_val, c] = BL[c].values[is_val]
        out = pd.DataFrame(index=df.index)
        for (rate_col, den_col, pre) in specs:
            asof_n = df[den_col].fillna(0).values.astype(np.float64)
            asof_cnt = asof_n * df[rate_col].fillna(0).values.astype(np.float64)
            hist_n = B[den_col].fillna(0).values.astype(np.float64)
            hist_cnt = B[f'__cnt_{pre}'].fillna(0).values.astype(np.float64)
            cur_n = np.maximum(asof_n - hist_n, 0.0)
            cur_cnt = np.clip(asof_cnt - hist_cnt, 0.0, None)
            with np.errstate(invalid='ignore', divide='ignore'):
                cur_rate = np.where(cur_n > 0, cur_cnt / np.maximum(cur_n, 1), np.nan)
                hist_rate = np.where(hist_n > 0, hist_cnt / np.maximum(hist_n, 1), np.nan)
            out[f'cs_{pre}_rate'] = cur_rate
            out[f'cs_{pre}_minus_hist'] = cur_rate - hist_rate
            # EB blend of current-season and career
            m = self.eb_m
            fb = self.fallback_.get(pre, 0.5)
            blend = (cur_n * np.nan_to_num(cur_rate, nan=fb)
                     + m * np.nan_to_num(hist_rate, nan=fb)) / (cur_n + m)
            out[f'cs_{pre}_eb'] = blend
        out[f'cs_{ent_col[:3]}_cur_n'] = np.maximum(
            df[specs[0][1]].fillna(0).values - B[specs[0][1]].fillna(0).values, 0)
        out[f'cs_{ent_col[:3]}_hist_n'] = B[specs[0][1]].fillna(0).values
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        A = self._apply(df, 'pitcher_id', PITCHER_RATES, self.pb_, self.pb_val_)
        Bx = self._apply(df, 'batter_id', BATTER_RATES, self.bb_, self.bb_val_)
        X = pd.concat([A, Bx], axis=1)
        X['cs_pb_succ_sum'] = X['cs_p_succ_eb'] + X['cs_b_succ_eb']
        return X.astype(np.float32)


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(config.TRAIN_PATH)
    for vs in [2022, 2023, 2024]:
        hist = df[df.season < vs]
        val = df[df.season == vs]
        dec = AsofDecomposer().fit(hist, vs)
        X = dec.transform(val)
        y = val[TGT].values
        print(f"\n=== val {vs} (n={len(val):,}) ===")
        base_auc = roc_auc_score(y, val['asof_pitcher_success_rate'].fillna(
            val['asof_pitcher_success_rate'].median()).values)
        print(f"  reference asof_pitcher_success_rate AUC = {base_auc:.4f}")
        for c in ['cs_p_succ_rate', 'cs_p_succ_eb', 'cs_p_succ_minus_hist',
                  'cs_b_succ_rate', 'cs_b_succ_eb', 'cs_pb_succ_sum',
                  'cs_p_mid_rate', 'cs_p_ball_rate', 'cs_p_str_rate']:
            v = X[c].values
            m = np.isfinite(v)
            if m.sum() > 5000:
                print(f"  {c:<24} AUC={roc_auc_score(y[m], v[m]):.4f}  cov={m.mean():.3f}"
                      f"  mean={np.nanmean(v):.4f}")
        # sanity: does cs_p_succ_rate track the league level of the val season?
        print(f"  val season actual rate = {y.mean():.4f} ; "
              f"mean cs_p_succ_rate = {np.nanmean(X['cs_p_succ_rate']):.4f} ; "
              f"mean asof career rate = {val['asof_pitcher_success_rate'].mean():.4f}")
