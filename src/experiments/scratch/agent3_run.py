"""agent3_run.py — flexible fast runner for agent3 experiments."""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from agent3_lib import get_fold, cat_cols_of, calc_skill, calc_raw_brier, report, CACHE
import lightgbm as lgb

SEASON_R = {2019: 0.564670, 2020: 0.532712, 2021: 0.532762, 2022: 0.528920,
            2023: 0.499957, 2024: 0.486105}
FOLD_TRAIN_SEASONS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022],
                      2: [2019, 2020, 2021, 2022, 2023]}
VAL_SEASON = {0: 2022, 1: 2023, 2: 2024}
AS_OF = {0: 2021, 1: 2022, 2: 2023}

LGB_L2 = dict(objective='regression', n_estimators=250, num_leaves=45, learning_rate=0.05,
              min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
              verbosity=-1, n_jobs=9)


def recenter(p, r_hat):
    return np.clip(p + (r_hat - p.mean()), 1e-6, 1 - 1e-6)


def r_hat_for(k, mode='last'):
    seasons = FOLD_TRAIN_SEASONS[k]
    ys = np.array([SEASON_R[s] for s in seasons])
    if mode == 'last':
        return ys[-1]
    if mode == 'last_plus_meandelta':
        return ys[-1] + np.diff(ys).mean()
    if mode == 'lin_last3':
        b, a = np.polyfit(np.array(seasons[-3:], float), ys[-3:], 1)
        return a + b * VAL_SEASON[k]
    if mode == 'none':
        return None
    raise ValueError(mode)


def recency_w(s_tr, decay, as_of):
    return decay ** (as_of - s_tr)


def run(name, extra_fn=None, seeds=(7, 123), params=None, decay=None,
        recenter_modes=('none', 'last', 'last_plus_meandelta'), drop=None, folds=(0, 1, 2),
        model='lgb_l2', return_preds=False):
    t0 = time.time()
    P, Y = {}, {}
    for k in folds:
        X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
        if extra_fn is not None:
            X_tr, X_va = extra_fn(k, X_tr, X_va, sd_tr, sd_va)
        if drop:
            X_tr = X_tr.drop(columns=[c for c in drop if c in X_tr.columns])
            X_va = X_va.drop(columns=[c for c in drop if c in X_va.columns])
        cc = cat_cols_of(X_tr)
        cidx = [X_tr.columns.get_loc(c) for c in cc]
        w = None if decay is None else recency_w(s_tr, decay, AS_OF[k])
        pp = np.zeros(len(y_va))
        for sd in seeds:
            pr = dict(LGB_L2); pr.update(params or {}); pr['random_state'] = sd
            m = lgb.LGBMRegressor(**pr)
            m.fit(X_tr, y_tr, categorical_feature=cidx, sample_weight=w)
            pp += m.predict(X_va)
        P[k] = pp / len(seeds); Y[k] = y_va
        del X_tr, X_va
    res = {}
    for mode in recenter_modes:
        pl = []
        for k in folds:
            rh = r_hat_for(k, mode)
            pl.append(P[k] if rh is None else recenter(P[k], rh))
        res[mode] = report(f'{name} | rc={mode}', pl, [Y[k] for k in folds])
    print(f'   ({time.time()-t0:.0f}s, n_feat varies)')
    if return_preds:
        return res, P, Y
    return res
