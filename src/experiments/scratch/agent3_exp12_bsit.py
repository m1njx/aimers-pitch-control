"""EXP12: batter-side + matchup trackman features on top of the EXP10 best recipe."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
from agent3_tkm_bat import attach_bat, attach_matchup
import lightgbm as lgb

SEEDS = (7, 123)
AS_OF = {0: 2021, 1: 2022, 2: 2023}
SCHEMES = [('add', 'last', None, 1.0),
           ('seg_relative', 'last', 'count', 1.0),
           ('seg_relative', 'last_plus_halfdelta', 'count', 1.0),
           ('seg_relative', 'last_plus_meandelta', 'count', 1.0)]


def fit_fold(k, parts, seeds=SEEDS, params=None):
    X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
    if 'sit' in parts:
        F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
        X_tr = attach(F, sd_tr, X_tr, list(F.columns)); X_va = attach(F, sd_va, X_va, list(F.columns))
    if 'bsit' in parts:
        F = pd.read_parquet(CACHE / f'tkm_bsit_{AS_OF[k]}.parquet')
        X_tr = attach_bat(F, sd_tr, X_tr); X_va = attach_bat(F, sd_va, X_va)
    if 'mu' in parts:
        F = pd.read_parquet(CACHE / f'tkm_mu_{AS_OF[k]}.parquet')
        X_tr = attach_matchup(F, sd_tr, X_tr); X_va = attach_matchup(F, sd_va, X_va)
    rs = pd.Series(y_tr).groupby(s_tr).mean()
    yt = y_tr - pd.Series(s_tr).map(rs).values
    cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr.update(params or {}); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, yt, categorical_feature=cidx)
        p += m.predict(X_va)
    nf = X_tr.shape[1]
    del X_tr, X_va
    return p / len(seeds), y_va, sd_va, sd_tr, y_tr, nf


def run_parts(tag, parts, seeds=SEEDS, params=None, save=None):
    t0 = time.time()
    RAW, Y, SDV, SDT, YTR = {}, {}, {}, {}, {}
    nf = 0
    for k in range(3):
        RAW[k], Y[k], SDV[k], SDT[k], YTR[k], nf = fit_fold(k, parts, seeds, params)
    for sch, mode, kind, al in SCHEMES:
        pl = [calibrate(RAW[k], SDV[k], SDT[k], YTR[k], k, sch, mode, kind or 'count', al)
              for k in range(3)]
        report(f'{tag}({nf}f) | {sch}/{mode}', pl, [Y[k] for k in range(3)])
    if save:
        np.save(CACHE / f'{save}.npy', np.concatenate([RAW[k] for k in range(3)]))
    print(f'   {time.time()-t0:.0f}s\n', flush=True)
    return RAW, Y, SDV, SDT, YTR


if __name__ == '__main__':
    run_parts('sit', ['sit'], save='exp12_sit')
    run_parts('sit+bsit', ['sit', 'bsit'], save='exp12_sit_bsit')
    run_parts('sit+bsit+mu', ['sit', 'bsit', 'mu'], save='exp12_all')
    run_parts('bsit only', ['bsit'])
