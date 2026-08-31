"""EXP07: era-neutralized target.

`season` is EXCLUDED from the feature set, so the GBDT literally cannot adjust for the
league base-rate era drift (0.5647 -> 0.4861): every leaf averages 2019..2023 together,
which injects pure era noise into every split AND mis-centers the output.

Fix: train on y_adj = y - r_season (deviation from that season's league rate), then
predict p = r_hat(val_season) + model_output. r_hat is extrapolated from train seasons only.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2, SEASON_R, r_hat_for
import lightgbm as lgb

SEEDS = (7, 123)
FTS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022, 2023]}
AS_OF = {0: 2021, 1: 2022, 2: 2023}


def fit_pred(X_tr, y_tr, X_va, w=None, seeds=SEEDS):
    cc = cat_cols_of(X_tr)
    cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, y_tr, categorical_feature=cidx, sample_weight=w)
        p += m.predict(X_va)
    return p / len(seeds)


def main():
    t0 = time.time()
    out = {}
    for tag in ['raw', 'era', 'era_dec07']:
        P, Y = [], []
        for k in range(3):
            X_tr, X_va, y_tr, y_va, s_tr = get_fold(k)
            rs = pd.Series(y_tr).groupby(s_tr).mean()
            w = None
            if tag == 'raw':
                yt = y_tr
            else:
                yt = y_tr - pd.Series(s_tr).map(rs).values
            if tag == 'era_dec07':
                w = 0.7 ** (AS_OF[k] - s_tr)
            p = fit_pred(X_tr, yt, X_va, w=w)
            if tag != 'raw':
                p = p + r_hat_for(k, 'last')
            P.append(np.clip(p, 1e-6, 1 - 1e-6)); Y.append(y_va)
            del X_tr, X_va
            print(f'  {tag} fold{k} {time.time()-t0:.0f}s', flush=True)
        out[tag] = (P, Y)
        report(f'{tag}', P, Y)
        # also with post-hoc recenter to r_hat(last) for the raw variant
        if tag == 'raw':
            pl = [np.clip(P[k] + (r_hat_for(k, 'last') - P[k].mean()), 1e-6, 1 - 1e-6) for k in range(3)]
            report('raw + rc=last', pl, Y)
        np.save(CACHE / f'exp07_{tag}.npy', np.concatenate(P))
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
