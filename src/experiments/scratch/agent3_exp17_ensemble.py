"""EXP17: multi-family ensemble on the era-target + small-tree + sit-feature recipe.

Members: LightGBM(small), LightGBM linear_tree, CatBoost(shallow), XGBoost(shallow), Ridge.
Blend weights are chosen on INNER folds (2022,2023) ONLY; outer(2024) is reported honestly.
"""
import sys, time, itertools, json
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE, calc_skill
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
from agent3_exp16_linear import ridge_pred
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

AS_OF = {0: 2021, 1: 2022, 2: 2023}
SEEDS = (7, 123)


def prep(k):
    X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
    F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
    X_tr = attach(F, sd_tr, X_tr, list(F.columns))
    X_va = attach(F, sd_va, X_va, list(F.columns))
    rs = pd.Series(y_tr).groupby(s_tr).mean()
    yt = (y_tr - pd.Series(s_tr).map(rs).values).astype(np.float64)
    return X_tr, X_va, yt, y_va, sd_tr, sd_va, y_tr


def m_lgb(X_tr, yt, X_va, params):
    cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in SEEDS:
        pr = dict(LGB_L2); pr.update(params); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr); m.fit(X_tr, yt, categorical_feature=cidx)
        p += m.predict(X_va)
    return p / len(SEEDS)


def m_cb(X_tr, yt, X_va, depth=4, iters=800, lr=0.05):
    cc = cat_cols_of(X_tr)
    A = X_tr.copy(); B = X_va.copy()
    for c in cc:
        A[c] = A[c].astype(int).astype(str); B[c] = B[c].astype(int).astype(str)
    for c in [x for x in A.columns if x not in cc]:
        A[c] = A[c].astype(np.float32); B[c] = B[c].astype(np.float32)
    p = np.zeros(len(X_va))
    for sd in SEEDS:
        m = CatBoostRegressor(loss_function='RMSE', depth=depth, iterations=iters,
                              learning_rate=lr, l2_leaf_reg=10.0, random_seed=sd,
                              verbose=0, cat_features=cc, thread_count=-1)
        m.fit(A, yt); p += m.predict(B)
    return p / len(SEEDS)


def m_xgb(X_tr, yt, X_va, depth=3, n=800, lr=0.03):
    cc = cat_cols_of(X_tr)
    A = X_tr.copy(); B = X_va.copy()
    for c in cc:
        A[c] = A[c].astype('category').cat.codes.astype(np.float32)
        B[c] = B[c].astype('category').cat.codes.astype(np.float32)
    A = A.astype(np.float32); B = B.astype(np.float32)
    p = np.zeros(len(X_va))
    for sd in SEEDS:
        m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=depth, n_estimators=n,
                             learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=10.0, random_state=sd, n_jobs=-1)
        m.fit(A, yt); p += m.predict(B)
    return p / len(SEEDS)


MEMBERS = {
    'lgb10': lambda A, y, B: m_lgb(A, y, B, dict(num_leaves=10, n_estimators=1500, learning_rate=0.01)),
    'lgblin': lambda A, y, B: m_lgb(A, y, B, dict(num_leaves=10, linear_tree=True, reg_lambda=10.0)),
    'cb4': lambda A, y, B: m_cb(A, y, B),
    'xgb3': lambda A, y, B: m_xgb(A, y, B),
    'ridge': lambda A, y, B: ridge_pred(A, y, B, 10000.0),
}


def main():
    t0 = time.time()
    P = {n: {} for n in MEMBERS}
    Y, SV, ST, YT = {}, {}, {}, {}
    for k in range(3):
        X_tr, X_va, yt, y_va, sd_tr, sd_va, y_tr = prep(k)
        Y[k], SV[k], ST[k], YT[k] = y_va, sd_va, sd_tr, y_tr
        for name, fn in MEMBERS.items():
            P[name][k] = fn(X_tr, yt, X_va)
            print(f'  fold{k} {name} {time.time()-t0:.0f}s', flush=True)
        del X_tr, X_va
    for name in MEMBERS:
        np.save(CACHE / f'exp17_{name}.npy', np.concatenate([P[name][k] for k in range(3)]))
    np.save(CACHE / 'exp17_y.npy', np.concatenate([Y[k] for k in range(3)]))

    def ev(raw, mode='last_plus_halfdelta'):
        pl = [calibrate(raw[k], SV[k], ST[k], YT[k], k, 'seg_relative', mode, 'count', 1.0)
              for k in range(3)]
        sk = [calc_skill(Y[k], pl[k]) for k in range(3)]
        return sk

    print('\n--- individual members ---')
    for name in MEMBERS:
        for mode in ['last', 'last_plus_halfdelta']:
            sk = ev(P[name], mode)
            print(f'[{name:<8} {mode:<20}] full={np.mean(sk):8.2f} inner={np.mean(sk[:2]):8.2f} '
                  f'OUTER={sk[2]:8.2f} folds={[round(x,1) for x in sk]}')

    print('\n--- correlation of member predictions (fold2/2024) ---')
    M = pd.DataFrame({n: P[n][2] for n in MEMBERS})
    print(M.corr().round(4).to_string())

    # weight search on INNER folds only
    names = list(MEMBERS)
    grid = np.arange(0, 1.01, 0.1)
    best = None
    print('\n--- inner-selected blend weights ---')
    for mode in ['last', 'last_plus_halfdelta']:
        bi = None
        for w in itertools.product(grid, repeat=len(names)):
            s = sum(w)
            if s <= 0 or abs(s - 1.0) > 1e-9:
                continue
            raw = {k: sum(wi * P[n][k] for wi, n in zip(w, names)) for k in range(3)}
            sk = ev(raw, mode)
            inner = np.mean(sk[:2])
            if bi is None or inner > bi[0]:
                bi = (inner, w, sk)
        inner, w, sk = bi
        print(f'[{mode}] w={dict(zip(names, [round(x,2) for x in w]))} '
              f'full={np.mean(sk):8.2f} inner={inner:8.2f} OUTER={sk[2]:8.2f} '
              f'folds={[round(x,1) for x in sk]}')
    print(f'\ntotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
