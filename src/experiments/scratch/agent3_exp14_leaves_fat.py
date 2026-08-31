"""EXP14: (a) smaller-tree sweep (leaves31 beat leaves45 on outer), (b) fatigue features."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
from agent3_tkm_fatigue import attach_fat
import lightgbm as lgb

AS_OF = {0: 2021, 1: 2022, 2: 2023}
SCHEMES = [('seg_relative', 'last', 'count', 1.0),
           ('seg_relative', 'last_plus_halfdelta', 'count', 1.0)]


def fit_fold(k, parts, seeds, params):
    X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
    if 'sit' in parts:
        F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
        X_tr = attach(F, sd_tr, X_tr, list(F.columns)); X_va = attach(F, sd_va, X_va, list(F.columns))
    if 'fat' in parts:
        F = pd.read_parquet(CACHE / f'tkm_fat_{AS_OF[k]}.parquet')
        X_tr = attach_fat(F, sd_tr, X_tr); X_va = attach_fat(F, sd_va, X_va)
    rs = pd.Series(y_tr).groupby(s_tr).mean()
    yt = y_tr - pd.Series(s_tr).map(rs).values
    cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr.update(params); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, yt, categorical_feature=cidx)
        p += m.predict(X_va)
    nf = X_tr.shape[1]; del X_tr, X_va
    return p / len(seeds), y_va, sd_va, sd_tr, y_tr, nf


def go(tag, parts, seeds, params, save=None):
    t0 = time.time()
    R, Y, SV, ST, YT = {}, {}, {}, {}, {}
    for k in range(3):
        R[k], Y[k], SV[k], ST[k], YT[k], nf = fit_fold(k, parts, seeds, params)
    for sch, mode, kind, al in SCHEMES:
        pl = [calibrate(R[k], SV[k], ST[k], YT[k], k, sch, mode, kind, al) for k in range(3)]
        report(f'{tag}({nf}f) | {mode}', pl, [Y[k] for k in range(3)])
    if save:
        np.save(CACHE / f'{save}.npy', np.concatenate([R[k] for k in range(3)]))
    print(f'   {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    for lv in [10, 15, 20, 25, 31]:
        go(f'lv{lv}', ['sit'], (7,), dict(num_leaves=lv))
    print()
    for lv in [15, 20, 31]:
        go(f'lv{lv} n800lr.02', ['sit'], (7,), dict(num_leaves=lv, n_estimators=800, learning_rate=0.02))
    print()
    go('lv31 +fat', ['sit', 'fat'], (7,), dict(num_leaves=31))
    go('lv31 fat-only', ['fat'], (7,), dict(num_leaves=31))
