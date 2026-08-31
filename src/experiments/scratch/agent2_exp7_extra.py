"""
agent2_exp7_extra.py — LightGBM-only INNER-FOLD screening of further feature
ideas stacked on top of the winning `asof_dec` set.

Selection rule: candidates are judged on inner folds (2022, 2023) ONLY.
Nothing here is allowed to look at 2024.

v0 asof_dec           control
v1 +form_ladder       multi-scale recency ladder: career / season-to-date /
                      last5 / last3 / last1 game rates and their differences,
                      incl. the algebraically implied "games 2-3" and "4-5" rates
v2 +park              park_id (= home team id) and the ordered team-pair, which
                      the model can only reach today via a 3-way interaction of
                      top_bottom x pitcher_team x batter_team
v3 +hist_cond         pitcher's PRIOR-SEASON success/reverse/middle rate
                      conditioned on the count, EB-smoothed. Uses the recovered
                      hidden labels (reverse/middle are not otherwise available)
v4 +pxb               explicit pitcher x batter ability interactions
"""
import sys, os, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_asof_decomp import AsofDecomposer
from agent2_recover_labels import recover
from core.eval_utils import calc_brier_skill_score

SEEDS = [7, 123]
TGT = config.TARGET_COL


def form_ladder(df, A):
    X = pd.DataFrame(index=df.index)
    p1 = df['asof_pitcher_prev1_game_success_rate']
    p3 = df['asof_pitcher_prev3_game_success_rate']
    p5 = df['asof_pitcher_prev5_game_success_rate']
    m1 = df['asof_pitcher_prev1_game_middle_rate']
    m3 = df['asof_pitcher_prev3_game_middle_rate']
    m5 = df['asof_pitcher_prev5_game_middle_rate']
    cs = A['cs_p_succ_rate']
    X['fl_g23'] = (3 * p3 - p1) / 2.0          # games 2-3 (equal-weight approx)
    X['fl_g45'] = (5 * p5 - 3 * p3) / 2.0      # games 4-5
    X['fl_slope'] = p1 - p5
    X['fl_p1_minus_cs'] = p1 - cs
    X['fl_p5_minus_cs'] = p5 - cs
    X['fl_p5_minus_career'] = p5 - df['asof_pitcher_success_rate']
    X['fl_mid_slope'] = m1 - m5
    X['fl_mid_g23'] = (3 * m3 - m1) / 2.0
    return X.astype(np.float32)


def park_feats(df):
    X = pd.DataFrame(index=df.index)
    is_top = (df['top_bottom'] == 'T').values
    home = np.where(is_top, df['pitcher_team_id'].values, df['batter_team_id'].values)
    away = np.where(is_top, df['batter_team_id'].values, df['pitcher_team_id'].values)
    X['park_id'] = home.astype(np.int32)
    X['pk_is_pitcher_home'] = is_top.astype(np.int8)
    X['pk_team_pair'] = (home.astype(np.int32) * 100 + away.astype(np.int32))
    return X


class HistCondRates:
    """Pitcher x count historical rates for success / reverse / middle,
    computed from PRIOR SEASONS ONLY (relative to each row's own season)."""

    def __init__(self, m=60.0):
        self.m = m

    def fit(self, df_hist, L_hist, val_season):
        d = pd.DataFrame({'pid': df_hist['pitcher_id'].values,
                          'season': df_hist['season'].values,
                          'b': df_hist['balls_before'].fillna(0).astype(int).values,
                          's': df_hist['strikes_before'].fillna(0).astype(int).values,
                          'succ': df_hist[TGT].values.astype(float),
                          'rev': L_hist['lab_reverse'].values,
                          'mid': L_hist['lab_middle'].values})
        d = d.dropna()
        self.val_season = val_season
        # cumulative-over-earlier-seasons per (pid, b, s)
        g = d.groupby(['pid', 'b', 's', 'season'])[['succ', 'rev', 'mid']].agg(['sum', 'size'])
        g.columns = ['succ_s', 'n', 'rev_s', 'n2', 'mid_s', 'n3']
        g = g[['succ_s', 'rev_s', 'mid_s', 'n']].reset_index().sort_values(
            ['pid', 'b', 's', 'season'])
        grp = g.groupby(['pid', 'b', 's'])
        for c in ['succ_s', 'rev_s', 'mid_s', 'n']:
            g['cum_' + c] = grp[c].cumsum() - g[c]
        self.tab_ = g.set_index(['pid', 'b', 's', 'season'])[
            ['cum_succ_s', 'cum_rev_s', 'cum_mid_s', 'cum_n']]
        # end-of-history state for val-season rows
        lastg = g.groupby(['pid', 'b', 's']).tail(1)
        end = lastg.copy()
        for c in ['succ_s', 'rev_s', 'mid_s', 'n']:
            end['cum_' + c] = end['cum_' + c] + end[c]
        self.end_ = end.set_index(['pid', 'b', 's'])[
            ['cum_succ_s', 'cum_rev_s', 'cum_mid_s', 'cum_n']]
        self.prior_ = dict(succ=float(d.succ.mean()), rev=float(d.rev.mean()),
                           mid=float(d.mid.mean()))
        return self

    def transform(self, df):
        b = df['balls_before'].fillna(0).astype(int).values
        s = df['strikes_before'].fillna(0).astype(int).values
        pid = df['pitcher_id'].values
        season = df['season'].values
        k1 = pd.MultiIndex.from_arrays([pid, b, s, season])
        T = self.tab_.reindex(k1)
        k2 = pd.MultiIndex.from_arrays([pid, b, s])
        E = self.end_.reindex(k2)
        isval = season == self.val_season
        Tv = T.values.copy()
        Tv[isval] = E.values[isval]
        n = np.nan_to_num(Tv[:, 3])
        X = pd.DataFrame(index=df.index)
        for i, (nm, key) in enumerate([('succ', 'succ'), ('rev', 'rev'), ('mid', 'mid')]):
            cnt = np.nan_to_num(Tv[:, i])
            X[f'hc_{nm}'] = (cnt + self.m * self.prior_[key]) / (n + self.m)
        X['hc_n'] = n
        return X.astype(np.float32)


def pxb(A):
    X = pd.DataFrame(index=A.index)
    X['pxb_prod'] = A['cs_p_succ_eb'] * A['cs_b_succ_eb']
    X['pxb_logit'] = (np.log(np.clip(A['cs_p_succ_eb'], 1e-3, 1 - 1e-3) /
                             (1 - np.clip(A['cs_p_succ_eb'], 1e-3, 1 - 1e-3))) +
                      np.log(np.clip(A['cs_b_succ_eb'], 1e-3, 1 - 1e-3) /
                             (1 - np.clip(A['cs_b_succ_eb'], 1e-3, 1 - 1e-3))))
    X['pxb_mid'] = A['cs_p_mid_rate'] + A['cs_b_mid_rate']
    return X.astype(np.float32)


def run(val_seasons=(2022, 2023), seeds=SEEDS):
    df = pd.read_csv(config.TRAIN_PATH)
    L = recover(df)
    res = {}
    for vs in val_seasons:
        tr = ((df.season >= 2019) & (df.season < vs)).values
        va = (df.season == vs).values
        df_tr = df[tr].copy(); df_val = df[va].copy()
        X_tr_b, X_val_b = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(X_tr_b)
        dec = AsofDecomposer().fit(df_tr, vs)
        A_tr = dec.transform(df_tr); A_val = dec.transform(df_val)
        X_tr_b = pd.concat([X_tr_b, A_tr], axis=1)
        X_val_b = pd.concat([X_val_b, A_val], axis=1)
        hc = HistCondRates().fit(df_tr, L[tr], vs)
        y_tr = df[TGT].values[tr]; y_val = df[TGT].values[va]
        log(f"val={vs}: control X={X_tr_b.shape}")

        blocks = {
            'v0_control': (None, None, []),
            'v1_form': (form_ladder(df_tr, A_tr), form_ladder(df_val, A_val), []),
            'v2_park': (park_feats(df_tr), park_feats(df_val), ['park_id', 'pk_team_pair']),
            'v3_histcond': (hc.transform(df_tr), hc.transform(df_val), []),
            'v4_pxb': (pxb(A_tr), pxb(A_val), []),
        }
        for name, (At, Av, newcat) in blocks.items():
            X_tr = X_tr_b if At is None else pd.concat([X_tr_b, At], axis=1)
            X_val = X_val_b if Av is None else pd.concat([X_val_b, Av], axis=1)
            cats = cc + newcat
            cat_idx = [X_tr.columns.get_loc(c) for c in cats]
            acc = np.zeros(len(X_val))
            for seed in seeds:
                m = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                       min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                       random_state=seed, verbosity=-1, n_jobs=-1)
                m.fit(X_tr, y_tr, categorical_feature=cat_idx)
                acc += m.predict_proba(X_val)[:, 1]
            p = acc / len(seeds)
            sk = calc_brier_skill_score(y_val, np.clip(p, 1e-6, 1 - 1e-6))
            best = max(((calc_brier_skill_score(y_val, np.clip(p + s, 1e-6, 1-1e-6))[0], round(s, 3))
                        for s in np.arange(-0.03, 0.021, 0.001)))
            res.setdefault(name, {})[vs] = dict(skill0=sk[0], brier=sk[1],
                                                best_skill=best[0], best_shift=best[1],
                                                nfeat=X_tr.shape[1])
            log(f"  [{name}] val={vs} skill={sk[0]:.2f} best={best[0]:.2f}@{best[1]:+.3f} "
                f"({X_tr.shape[1]} feats)")
        with open('~/LG_data/scratch/agent2_exp7.json', 'w') as f:
            json.dump(res, f, indent=2)
    print("\n=== INNER-ONLY screening summary (mean over inner folds) ===")
    for name, d in res.items():
        m0 = np.mean([v['skill0'] for v in d.values()])
        mb = np.mean([v['best_skill'] for v in d.values()])
        print(f"  {name:<14} skill(no shift)={m0:8.2f}   best-shift={mb:8.2f}")
    return res


if __name__ == '__main__':
    run()
