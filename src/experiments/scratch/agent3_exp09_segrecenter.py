"""EXP09: segment-wise recentering.

Global recentering forces mean(p) = r_hat. But the era drift is not perfectly uniform
across situations (e.g. 3-2 counts dropped more than 0-2). Recenter *within* segments
using each segment's own extrapolated base rate.
"""
import sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE, calc_skill
from agent3_run import LGB_L2, SEASON_R
import lightgbm as lgb

SEEDS = (7, 123)
AS_OF = {0: 2021, 1: 2022, 2: 2023}
VAL = {0: 2022, 1: 2023, 2: 2024}
FTS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022, 2023]}


def seg_keys(sd, kind):
    b = sd['balls_before'].values; s = sd['strikes_before'].values
    if kind == 'global':
        return np.zeros(len(sd), dtype=np.int64)
    if kind == 'count':
        return b * 3 + s
    if kind == 'strikes':
        return s
    if kind == 'balls':
        return b
    if kind == 'platoon':
        return sd['pitcher_hand'].values * 2 + sd['batter_hand'].values
    if kind == 'inning':
        return np.clip(sd['inning'].values, 1, 9)
    if kind == 'count_platoon':
        return (b * 3 + s) * 4 + sd['pitcher_hand'].values * 2 + sd['batter_hand'].values
    if kind == 'outs':
        return sd['outs_before'].values
    raise ValueError(kind)


def main():
    t0 = time.time()
    RAW, Y, SD, STR, YTR = {}, {}, {}, {}, {}
    for k in range(3):
        X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
        rs = pd.Series(y_tr).groupby(s_tr).mean()
        yt = y_tr - pd.Series(s_tr).map(rs).values
        cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
        p = np.zeros(len(X_va))
        for sd in SEEDS:
            pr = dict(LGB_L2); pr['random_state'] = sd
            m = lgb.LGBMRegressor(**pr)
            m.fit(X_tr, yt, categorical_feature=cidx)
            p += m.predict(X_va)
        RAW[k] = p / len(SEEDS)      # deviation-from-era predictions, mean ~0
        Y[k] = y_va; SD[k] = sd_va
        STR[k] = sd_tr; YTR[k] = y_tr
        del X_tr, X_va
        print(f'fold{k} fit {time.time()-t0:.0f}s', flush=True)

    def rhat_seg(k, kind, mode, min_n=300):
        """segment base-rate estimate for the val season from train seasons only."""
        sd_tr, y_tr = STR[k], YTR[k]
        g = seg_keys(sd_tr, kind)
        seas = sd_tr['season'].values
        d = pd.DataFrame({'g': g, 's': seas, 'y': y_tr})
        piv = d.pivot_table(index='g', columns='s', values='y', aggfunc='mean')
        cnt = d.pivot_table(index='g', columns='s', values='y', aggfunc='size')
        last = FTS[k][-1]
        if mode == 'last':
            r = piv[last]
        elif mode == 'last_plus_meandelta':
            cols = FTS[k]
            r = piv[last] + piv[cols].diff(axis=1).mean(axis=1)
        elif mode == 'last2avg':
            r = piv[FTS[k][-2:]].mean(axis=1)
        # global fallback / shrinkage for small cells
        gl = float(np.average(piv[last].values, weights=cnt[last].values))
        n = cnt[last]
        lam = n / (n + min_n)
        r = lam * r + (1 - lam) * gl
        return r.to_dict(), gl

    print()
    results = []
    for kind in ['global', 'count', 'strikes', 'balls', 'platoon', 'inning', 'outs', 'count_platoon']:
        for mode in ['last', 'last_plus_meandelta']:
            for lam in ([1.0] if kind == 'global' else [1.0, 0.5]):
                pl = []
                for k in range(3):
                    rmap, gl = rhat_seg(k, kind, mode)
                    gmap, _ = rhat_seg(k, 'global', mode)
                    gv = list(gmap.values())[0]
                    g = seg_keys(SD[k], kind)
                    rt = pd.Series(g).map(rmap).fillna(gv).values
                    rt = lam * rt + (1 - lam) * gv
                    p = RAW[k].copy()
                    # per-segment recentering of the deviation predictions
                    dfp = pd.DataFrame({'g': g, 'p': p})
                    mu = dfp.groupby('g')['p'].transform('mean').values
                    pl.append(np.clip(p - mu + rt, 1e-6, 1 - 1e-6))
                r = report(f'seg={kind:<14} mode={mode:<20} lam={lam}', pl, [Y[k] for k in range(3)])
                results.append(r)
    print(f'\ntotal {time.time()-t0:.0f}s')
    np.save(CACHE / 'exp09_raw.npy', np.concatenate([RAW[k] for k in range(3)]))


if __name__ == '__main__':
    main()
