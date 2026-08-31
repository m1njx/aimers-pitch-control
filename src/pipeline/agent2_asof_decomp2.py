"""
agent2_asof_decomp2.py — v2 of the asof algebraic decomposition.

Fixes found while validating the deployment path on the real test.csv:
  1. The v1 boundary came from the LAST training row's asof value, which
     excludes that row itself -> hist_n was short by exactly 1. For an entity
     whose first evaluation row is also their first pitch of the new season this
     produced cur_n = 1 and a nonsense cur_rate (observed 0.0016 on a real
     test.csv row). v2 takes the pitcher/batter success boundary EXACTLY from
     the train labels (groupby size / sum), and adds the +1 correction to the
     other rate counters.
  2. cur_n < MIN_CUR (default 3) now yields NaN instead of a 1-sample rate.
  3. All fallback constants are fitted, never computed from the scoring batch.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config

TGT = config.TARGET_COL
MIN_CUR = 3

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
EXACT = {'p_succ': ('pitcher_id', 'asof_pitcher_n'),
         'b_succ': ('batter_id', 'asof_batter_n')}


class AsofDecomposer2:
    def __init__(self, eb_m=150.0, min_cur=MIN_CUR):
        self.eb_m = eb_m
        self.min_cur = min_cur

    # ---------- boundary tables ----------
    def _season_boundary(self, df_hist, ent_col, specs):
        """state at END of season s-1, keyed (entity, s)"""
        cols = sorted(set([c for c, _, _ in specs] + [d for _, d, _ in specs]))
        last = df_hist.groupby([ent_col, 'season'])[cols].tail(1).copy()
        last[ent_col] = df_hist.groupby([ent_col, 'season'])[ent_col].tail(1).values
        last['season'] = df_hist.groupby([ent_col, 'season'])['season'].tail(1).values
        for (rc, dc, pre) in specs:
            last[f'__cnt_{pre}'] = (last[dc].fillna(0) * last[rc].fillna(0)) + last[rc].fillna(0)
            last[f'__den_{pre}'] = last[dc].fillna(0) + 1.0
        keep = [ent_col, 'season'] + [f'__cnt_{p}' for _, _, p in specs] + \
               [f'__den_{p}' for _, _, p in specs]
        last = last[keep].sort_values([ent_col, 'season'])
        g = last.groupby(ent_col)
        sh = g.shift(1)
        sh[ent_col] = last[ent_col].values
        sh['season'] = last['season'].values
        sh = sh.sort_values([ent_col, 'season'])
        vc = [c for c in sh.columns if c not in (ent_col, 'season')]
        sh[vc] = sh.groupby(ent_col)[vc].ffill()
        return sh.set_index([ent_col, 'season'])

    def _end_boundary(self, df_hist, ent_col, specs):
        cols = sorted(set([c for c, _, _ in specs] + [d for _, d, _ in specs]))
        last = df_hist.groupby(ent_col)[cols].tail(1).copy()
        last[ent_col] = df_hist.groupby(ent_col)[ent_col].tail(1).values
        for (rc, dc, pre) in specs:
            last[f'__cnt_{pre}'] = (last[dc].fillna(0) * last[rc].fillna(0)) + last[rc].fillna(0)
            last[f'__den_{pre}'] = last[dc].fillna(0) + 1.0
        keep = [ent_col] + [f'__cnt_{p}' for _, _, p in specs] + [f'__den_{p}' for _, _, p in specs]
        return last[keep].set_index(ent_col)

    def fit(self, df_hist: pd.DataFrame, val_season: int = 2025):
        self.val_season_ = val_season
        self.pb_ = self._season_boundary(df_hist, 'pitcher_id', PITCHER_RATES)
        self.bb_ = self._season_boundary(df_hist, 'batter_id', BATTER_RATES)
        self.pb_val_ = self._end_boundary(df_hist, 'pitcher_id', PITCHER_RATES)
        self.bb_val_ = self._end_boundary(df_hist, 'batter_id', BATTER_RATES)
        # EXACT success counters from labels (no off-by-one, no rate rounding)
        self.exact_ = {}
        for pre, (ent_col, _) in EXACT.items():
            g = df_hist.groupby([ent_col, 'season'])[TGT].agg(['sum', 'size']).reset_index()
            g = g.sort_values([ent_col, 'season'])
            grp = g.groupby(ent_col)
            g['cum_s'] = grp['sum'].cumsum() - g['sum']
            g['cum_n'] = grp['size'].cumsum() - g['size']
            season_tab = g.set_index([ent_col, 'season'])[['cum_s', 'cum_n']]
            tot = df_hist.groupby(ent_col)[TGT].agg(['sum', 'size'])
            tot.columns = ['cum_s', 'cum_n']
            self.exact_[pre] = (season_tab, tot, ent_col)
        self.league_ = df_hist.groupby('season')[TGT].mean()
        self.fallback_ = {}
        for specs, B in [(PITCHER_RATES, self.pb_val_), (BATTER_RATES, self.bb_val_)]:
            for (rc, dc, pre) in specs:
                den = B[f'__den_{pre}'].values
                cnt = B[f'__cnt_{pre}'].values
                with np.errstate(invalid='ignore', divide='ignore'):
                    hr = np.where(den > 0, cnt / np.maximum(den, 1), np.nan)
                self.fallback_[pre] = float(np.nanmean(hr)) if np.isfinite(hr).any() else 0.5
        return self

    def _apply(self, df, ent_col, specs, Bm, Bl):
        is_val = (df['season'].values == self.val_season_)
        key = pd.MultiIndex.from_arrays([df[ent_col].values, df['season'].values])
        B = Bm.reindex(key); B.index = df.index
        BL = Bl.reindex(df[ent_col].values); BL.index = df.index
        for c in B.columns:
            B.loc[is_val, c] = BL[c].values[is_val]
        out = pd.DataFrame(index=df.index)
        for (rc, dc, pre) in specs:
            asof_n = df[dc].fillna(0).values.astype(np.float64)
            asof_cnt = asof_n * df[rc].fillna(0).values.astype(np.float64)
            if pre in self.exact_:
                st, tot, ec = self.exact_[pre]
                E = st.reindex(key); E.index = df.index
                EL = tot.reindex(df[ent_col].values); EL.index = df.index
                hn = E['cum_n'].values.copy(); hs = E['cum_s'].values.copy()
                hn[is_val] = EL['cum_n'].values[is_val]
                hs[is_val] = EL['cum_s'].values[is_val]
                hist_n = np.nan_to_num(hn); hist_cnt = np.nan_to_num(hs)
            else:
                hist_n = np.nan_to_num(B[f'__den_{pre}'].values.astype(np.float64))
                hist_cnt = np.nan_to_num(B[f'__cnt_{pre}'].values.astype(np.float64))
                hist_n = np.where(hist_n > 0, hist_n, 0.0)
            cur_n = np.maximum(asof_n - hist_n, 0.0)
            cur_cnt = np.clip(asof_cnt - hist_cnt, 0.0, None)
            with np.errstate(invalid='ignore', divide='ignore'):
                cur_rate = np.where(cur_n >= self.min_cur, cur_cnt / np.maximum(cur_n, 1), np.nan)
                hist_rate = np.where(hist_n > 0, hist_cnt / np.maximum(hist_n, 1), np.nan)
            fb = self.fallback_.get(pre, 0.5)
            out[f'cs_{pre}_rate'] = cur_rate
            out[f'cs_{pre}_hist'] = hist_rate
            out[f'cs_{pre}_minus_hist'] = cur_rate - hist_rate
            m = self.eb_m
            out[f'cs_{pre}_eb'] = ((np.nan_to_num(cur_n * np.nan_to_num(cur_rate, nan=fb))
                                    + m * np.nan_to_num(hist_rate, nan=fb)) / (cur_n + m))
        out[f'cs_{ent_col[:3]}_cur_n'] = np.maximum(
            df[specs[0][1]].fillna(0).values - hist_n, 0)
        out[f'cs_{ent_col[:3]}_hist_n'] = hist_n
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        A = self._apply(df, 'pitcher_id', PITCHER_RATES, self.pb_, self.pb_val_)
        Bx = self._apply(df, 'batter_id', BATTER_RATES, self.bb_, self.bb_val_)
        X = pd.concat([A, Bx], axis=1)
        X['cs_pb_succ_sum'] = X['cs_p_succ_eb'] + X['cs_b_succ_eb']
        X['cs_pb_succ_diff'] = X['cs_p_succ_eb'] - X['cs_b_succ_eb']
        return X.astype(np.float32)


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(config.TRAIN_PATH)
    for vs in [2022, 2023, 2024]:
        hist = df[df.season < vs]; val = df[df.season == vs]
        X = AsofDecomposer2().fit(hist, vs).transform(val)
        y = val[TGT].values
        print(f"\n=== val {vs} ===")
        for c in ['cs_p_succ_rate', 'cs_p_succ_eb', 'cs_b_succ_rate', 'cs_pb_succ_sum']:
            v = X[c].values; m = np.isfinite(v)
            print(f"  {c:<20} AUC={roc_auc_score(y[m], v[m]):.4f} cov={m.mean():.4f} "
                  f"mean={np.nanmean(v):.4f}  (val actual {y.mean():.4f})")
    te = pd.read_csv(config.TEST_PATH)
    Xt = AsofDecomposer2().fit(df, 2025).transform(te)
    print("\n=== test.csv deployment check ===")
    print(Xt[['cs_p_succ_rate', 'cs_p_succ_eb', 'cs_pit_cur_n', 'cs_b_succ_rate',
              'cs_bat_cur_n']].to_string())
