"""EXP08: regime-restricted training window.

2023 is a structural break: asof_pitcher_ball_rate corr flips +0.014 -> -0.047,
asof_pitcher_strike_rate flips -0.024 -> +0.035, asof_batter_success_rate dies
(0.089 -> 0.000). Training on 2019-2022 with equal weight therefore teaches the model
sign-flipped relationships for the 2023+ regime.
Test: hard-restrict the training window to the last N seasons.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2, SEASON_R, r_hat_for
import lightgbm as lgb

SEEDS = (7, 123)
AS_OF = {0: 2021, 1: 2022, 2: 2023}


def fit_pred(X_tr, y_tr, X_va, w=None, seeds=SEEDS, params=None):
    cc = cat_cols_of(X_tr)
    cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr.update(params or {}); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, y_tr, categorical_feature=cidx, sample_weight=w)
        p += m.predict(X_va)
    return p / len(seeds)


def main():
    t0 = time.time()
    configs = [('all', None, None), ('last3', 3, None), ('last2', 2, None), ('last1', 1, None),
               ('all_d0.7', None, 0.7), ('all_d0.5', None, 0.5), ('all_d0.85', None, 0.85)]
    for tag, nlast, decay in configs:
        P, Y = [], []
        for k in range(3):
            X_tr, X_va, y_tr, y_va, s_tr = get_fold(k)
            if nlast is not None:
                msk = s_tr > AS_OF[k] - nlast
                X_tr = X_tr[msk]; yy = y_tr[msk]; ss = s_tr[msk]
            else:
                yy = y_tr; ss = s_tr
            rs = pd.Series(yy).groupby(ss).mean()
            yt = yy - pd.Series(ss).map(rs).values          # era-neutralized target
            w = None if decay is None else decay ** (AS_OF[k] - ss)
            p = fit_pred(X_tr, yt, X_va, w=w) + r_hat_for(k, 'last')
            P.append(np.clip(p, 1e-6, 1 - 1e-6)); Y.append(y_va)
            del X_tr, X_va
        report(f'era+rc | train={tag}', P, Y, extra=f'{time.time()-t0:.0f}s')
        np.save(CACHE / f'exp08_{tag}.npy', np.concatenate(P))


if __name__ == '__main__':
    main()
