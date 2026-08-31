"""EXP16: linear / additive models.

Small trees (leaves=4..10) beat big trees by a huge margin, which says the signal is
nearly additive and low-order. So a *properly regularised linear model* may match or beat
GBDT here, and should blend well (different inductive bias).
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

AS_OF = {0: 2021, 1: 2022, 2: 2023}
SCHEMES = [('seg_relative', 'last', 'count', 1.0),
           ('seg_relative', 'last_plus_halfdelta', 'count', 1.0)]


def prep_fold(k):
    X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
    F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
    X_tr = attach(F, sd_tr, X_tr, list(F.columns)); X_va = attach(F, sd_va, X_va, list(F.columns))
    rs = pd.Series(y_tr).groupby(s_tr).mean()
    yt = y_tr - pd.Series(s_tr).map(rs).values
    return X_tr, X_va, yt, y_va, sd_tr, sd_va, y_tr


def ridge_pred(X_tr, yt, X_va, alpha, n_bins=0):
    cc = set(cat_cols_of(X_tr))
    num = [c for c in X_tr.columns if c not in cc]
    A = X_tr[num].to_numpy(np.float32); B = X_va[num].to_numpy(np.float32)
    med = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), med, A); B = np.where(np.isnan(B), med, B)
    sc = StandardScaler().fit(A)
    A = sc.transform(A); B = sc.transform(B)
    if n_bins:
        kb = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='quantile',
                              subsample=200000, random_state=0).fit(A)
        A = np.hstack([A, kb.transform(A)]); B = np.hstack([B, kb.transform(B)])
    # one-hot the categoricals
    oh_tr, oh_va = [], []
    for c in cc:
        vals = pd.Index(sorted(pd.unique(X_tr[c])))
        if len(vals) > 60:
            continue
        oh_tr.append(pd.get_dummies(pd.Categorical(X_tr[c], categories=vals)).to_numpy(np.float32))
        oh_va.append(pd.get_dummies(pd.Categorical(X_va[c], categories=vals)).to_numpy(np.float32))
    if oh_tr:
        A = np.hstack([A] + oh_tr); B = np.hstack([B] + oh_va)
    m = Ridge(alpha=alpha, solver='lsqr')
    m.fit(A, yt)
    return m.predict(B)


def run(tag, fn, save=None):
    t0 = time.time()
    R, Y, SV, ST, YT = {}, {}, {}, {}, {}
    for k in range(3):
        X_tr, X_va, yt, y_va, sd_tr, sd_va, y_tr = prep_fold(k)
        R[k] = fn(X_tr, yt, X_va)
        Y[k], SV[k], ST[k], YT[k] = y_va, sd_va, sd_tr, y_tr
        del X_tr, X_va
    for sch, mode, kind, al in SCHEMES:
        pl = [calibrate(R[k], SV[k], ST[k], YT[k], k, sch, mode, kind, al) for k in range(3)]
        report(f'{tag} | {mode}', pl, [Y[k] for k in range(3)])
    if save:
        np.save(CACHE / f'{save}.npy', np.concatenate([R[k] for k in range(3)]))
    print(f'   {time.time()-t0:.0f}s', flush=True)
    return R, Y, SV, ST, YT


def lgb_fn(params, seeds=(7, 123)):
    def f(X_tr, yt, X_va):
        cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
        p = np.zeros(len(X_va))
        for sd in seeds:
            pr = dict(LGB_L2); pr.update(params); pr['random_state'] = sd
            m = lgb.LGBMRegressor(**pr)
            m.fit(X_tr, yt, categorical_feature=cidx)
            p += m.predict(X_va)
        return p / len(seeds)
    return f


if __name__ == '__main__':
    for a in [100.0, 1000.0, 10000.0]:
        run(f'Ridge a={a}', lambda A, y, B, a=a: ridge_pred(A, y, B, a), save=f'exp16_ridge{int(a)}')
    for a in [1000.0, 10000.0]:
        run(f'Ridge-bin8 a={a}', lambda A, y, B, a=a: ridge_pred(A, y, B, a, n_bins=8),
            save=f'exp16_ridgebin{int(a)}')
    run('LGB linear_tree lv10', lgb_fn(dict(num_leaves=10, linear_tree=True, reg_lambda=10.0)))
    run('LGB extra_trees lv10', lgb_fn(dict(num_leaves=10, extra_trees=True)))
    run('LGB lv10 ref', lgb_fn(dict(num_leaves=10)), save='exp16_lgb10')
