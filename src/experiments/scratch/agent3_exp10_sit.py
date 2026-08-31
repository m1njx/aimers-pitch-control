"""EXP10: situational trackman pitch-mix features + unified calibration grid."""
import sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
import lightgbm as lgb

SEEDS = (7, 123)
AS_OF = {0: 2021, 1: 2022, 2: 2023}
SITCOLS = None


def fit_fold(k, use_sit, seeds=SEEDS, params=None, extra=None):
    X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
    if use_sit:
        F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
        cols = SITCOLS or list(F.columns)
        X_tr = attach(F, sd_tr, X_tr, cols)
        X_va = attach(F, sd_va, X_va, cols)
    if extra is not None:
        X_tr, X_va = extra(k, X_tr, X_va, sd_tr, sd_va)
    rs = pd.Series(y_tr).groupby(s_tr).mean()
    yt = y_tr - pd.Series(s_tr).map(rs).values
    cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr.update(params or {}); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, yt, categorical_feature=cidx)
        p += m.predict(X_va)
    del X_tr, X_va
    return p / len(seeds), y_va, sd_va, sd_tr, y_tr


def eval_grid(tag, RAW, Y, SDV, SDT, YTR, schemes=None):
    schemes = schemes or [
        ('add', 'last', None, 1.0), ('add', 'last_plus_halfdelta', None, 1.0),
        ('add', 'last_plus_meandelta', None, 1.0),
        ('force_global', 'last', None, 1.0),
        ('force_seg', 'last', 'count', 1.0),
        ('force_seg', 'last_plus_meandelta', 'count', 1.0),
        ('seg_relative', 'last', 'count', 1.0),
        ('seg_relative', 'last', 'count', 0.5),
        ('seg_relative', 'last_plus_halfdelta', 'count', 1.0),
        ('seg_relative', 'last_plus_meandelta', 'count', 1.0),
        ('seg_relative', 'last', 'count_platoon', 1.0),
        ('seg_relative', 'last', 'balls', 1.0),
    ]
    out = []
    for sch, mode, kind, al in schemes:
        pl = [calibrate(RAW[k], SDV[k], SDT[k], YTR[k], k, sch, mode, kind or 'count', al)
              for k in range(3)]
        out.append(report(f'{tag} | {sch}/{mode}/{kind}/a={al}', pl, [Y[k] for k in range(3)]))
    return out


def main():
    t0 = time.time()
    for tag, use_sit in [('base', False), ('base+sit', True)]:
        RAW, Y, SDV, SDT, YTR = {}, {}, {}, {}, {}
        for k in range(3):
            RAW[k], Y[k], SDV[k], SDT[k], YTR[k] = fit_fold(k, use_sit)
            print(f'  {tag} fold{k} {time.time()-t0:.0f}s', flush=True)
        np.save(CACHE / f'exp10_raw_{tag}.npy', np.concatenate([RAW[k] for k in range(3)]))
        eval_grid(tag, RAW, Y, SDV, SDT, YTR)
        print()
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
