"""
agent2_entity_rel.py — LEAGUE-DRIFT-ADJUSTED entity ability features.

Motivation (fresh-eyes finding): the league-wide control_success rate falls
monotonically 0.5647 -> 0.4861 over 2019..2024. The organiser-supplied
`asof_pitcher_success_rate` is a CAREER-CUMULATIVE raw rate, so it silently
mixes eras: a veteran whose innings are mostly from 2019 looks "better" than an
identical pitcher whose innings are mostly from 2024, purely because of league
drift. Measured single-feature AUC of asof_batter_success_rate collapses from
0.536 (2022) to 0.499 (2023) - consistent with era contamination.

Fix: compute per-(entity, season) rates, subtract THAT season's league rate, then
aggregate. Features are strictly built from seasons < the row's own season, so
train and val/test rows have identical semantics and there is no leakage.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import config

TGT = config.TARGET_COL


class SeasonRelativeAbility:
    def __init__(self, m_prev=200.0, m_career=400.0):
        self.m_prev = m_prev
        self.m_career = m_career

    def fit(self, df_hist: pd.DataFrame, max_season_needed: int):
        """df_hist: labelled rows (the fold's TRAIN split only)."""
        league = df_hist.groupby('season')[TGT].mean()
        self.league_ = league
        self.tables_ = {}
        for col, pre in [('pitcher_id', 'pit'), ('batter_id', 'bat')]:
            g = df_hist.groupby([col, 'season'])[TGT].agg(['sum', 'size']).reset_index()
            g['rel'] = g['sum'] / g['size'] - g['season'].map(league)
            g['w_rel'] = g['rel'] * g['size']
            g = g.sort_values([col, 'season'])
            # cumulative over strictly-earlier seasons
            grp = g.groupby(col)
            g['cum_n'] = grp['size'].cumsum() - g['size']
            g['cum_wrel'] = grp['w_rel'].cumsum() - g['w_rel']
            g['prev_n'] = grp['size'].shift(1)
            g['prev_rel'] = grp['rel'].shift(1)
            g['prev_season'] = grp['season'].shift(1)
            # only count prev season if it is literally season-1
            ok = (g['prev_season'] == g['season'] - 1)
            g.loc[~ok, ['prev_n', 'prev_rel']] = np.nan
            rows = g[[col, 'season', 'cum_n', 'cum_wrel', 'prev_n', 'prev_rel']].copy()
            # rows only exist for seasons the entity actually played; we need a
            # lookup for ANY (entity, season) up to max_season_needed -> forward fill
            ents = rows[col].unique()
            full = pd.MultiIndex.from_product(
                [ents, range(int(df_hist.season.min()), max_season_needed + 1)],
                names=[col, 'season'])
            rows = rows.set_index([col, 'season']).reindex(full)
            # cum_* : carry forward the running total; prev_* only valid for exact season
            rows[['cum_n', 'cum_wrel']] = rows.groupby(level=0)[['cum_n', 'cum_wrel']].ffill()
            # for a season the entity did not play we still want the running total
            # ffill above handles it, but the first appearance season's cum is the
            # value stored there (already excludes own season) - correct.
            rows[f'{pre}_career_rel'] = rows['cum_wrel'] / (rows['cum_n'] + self.m_career)
            rows[f'{pre}_career_n'] = rows['cum_n']
            rows[f'{pre}_prev_rel'] = (rows['prev_rel'] * rows['prev_n']
                                       / (rows['prev_n'] + self.m_prev))
            rows[f'{pre}_prev_n'] = rows['prev_n']
            self.tables_[col] = rows[[f'{pre}_career_rel', f'{pre}_career_n',
                                      f'{pre}_prev_rel', f'{pre}_prev_n']].astype(np.float32)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for col in ['pitcher_id', 'batter_id']:
            key = pd.MultiIndex.from_arrays([df[col].values, df['season'].values])
            T = self.tables_[col].reindex(key)
            T.index = df.index
            out.append(T)
        X = pd.concat(out, axis=1)
        X['pb_rel_sum'] = X['pit_career_rel'] + X['bat_career_rel']
        X['pb_rel_diff'] = X['pit_career_rel'] - X['bat_career_rel']
        return X.astype(np.float32)


if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_PATH, usecols=['season', 'pitcher_id', 'batter_id', TGT])
    from sklearn.metrics import roc_auc_score
    for vs in [2022, 2023, 2024]:
        hist = df[df.season < vs]
        val = df[df.season == vs]
        sr = SeasonRelativeAbility().fit(hist, vs)
        X = sr.transform(val)
        print(f"\n=== val {vs} ===  null frac:\n{X.isna().mean().round(3).to_dict()}")
        y = val[TGT].values
        for c in X.columns:
            m = X[c].notna().values
            if m.sum() > 5000:
                print(f"  {c:<18} AUC={roc_auc_score(y[m], X[c].values[m]):.4f}  cov={m.mean():.3f}")
        # compare with raw asof
        raw = pd.read_csv(config.TRAIN_PATH, usecols=['season', 'asof_pitcher_success_rate',
                                                      'asof_batter_success_rate', TGT])
        r = raw[raw.season == vs]
        for c in ['asof_pitcher_success_rate', 'asof_batter_success_rate']:
            m = r[c].notna().values
            print(f"  [asof] {c:<32} AUC={roc_auc_score(r[TGT].values[m], r[c].values[m]):.4f}")
