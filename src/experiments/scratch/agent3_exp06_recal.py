"""EXP06: temporal linear recalibration p' = a + b*(p - mean(p)).

Hypothesis: a model trained on past seasons is BOTH mis-centered (intercept) and
over-dispersed (slope < 1) when applied to a future season. Both cost Brier.
The correction (a,b) can be estimated honestly from an internal backtest one season
earlier (train<=S-2 -> val S-1), then applied to the real val season S.
"""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from agent3_lib import get_fold, cat_cols_of, calc_skill, report, CACHE
from agent3_run import LGB_L2, SEASON_R
import lightgbm as lgb

SEEDS = (7, 123)


def fit_pred(X_tr, y_tr, X_va, seeds=SEEDS):
    cc = cat_cols_of(X_tr)
    cidx = [X_tr.columns.get_loc(c) for c in cc]
    p = np.zeros(len(X_va))
    for sd in seeds:
        pr = dict(LGB_L2); pr['random_state'] = sd
        m = lgb.LGBMRegressor(**pr)
        m.fit(X_tr, y_tr, categorical_feature=cidx)
        p += m.predict(X_va)
    return p / len(seeds)


def opt_ab(p, y):
    """least squares y ~ a + b*(p - p.mean())"""
    pc = p - p.mean()
    b = float(np.dot(pc, y - y.mean()) / np.dot(pc, pc))
    a = float(y.mean())
    return a, b


def main():
    t0 = time.time()
    P, Y = {}, {}
    for k in range(3):
        X_tr, X_va, y_tr, y_va, s_tr = get_fold(k)
        P[k] = fit_pred(X_tr, y_tr, X_va)
        Y[k] = y_va
        print(f'fold{k} done {time.time()-t0:.0f}s')
        del X_tr, X_va

    # extra backtest fold: train 2019-2020 -> val 2021 (needed to recalibrate fold0)
    X_tr, X_va, y_tr, y_va, s_tr = get_fold(0)  # tr=2019-2021 val=2022
    msk = s_tr < 2021
    Xb_tr = X_tr[msk]; yb_tr = y_tr[msk]
    Xb_va = X_tr[~msk]; yb_va = y_tr[~msk]
    Pm1 = {0: (fit_pred(Xb_tr, yb_tr, Xb_va), yb_va)}
    print(f'backtest 2021 done {time.time()-t0:.0f}s')
    del X_tr, X_va, Xb_tr, Xb_va
    Pm1[1] = (P[0], Y[0])   # for fold1 (val 2023) the prior backtest is fold0 (val 2022)
    Pm1[2] = (P[1], Y[1])   # for fold2 (val 2024) the prior backtest is fold1 (val 2023)

    print()
    report('raw', [P[k] for k in range(3)], [Y[k] for k in range(3)])

    ab_oracle = {k: opt_ab(P[k], Y[k]) for k in range(3)}
    ab_bt = {k: opt_ab(Pm1[k][0], Pm1[k][1]) for k in range(3)}
    print('oracle  (a,b) per fold:', {k: (round(v[0], 4), round(v[1], 3)) for k, v in ab_oracle.items()})
    print('backtest(a,b) per fold:', {k: (round(v[0], 4), round(v[1], 3)) for k, v in ab_bt.items()})

    def apply(ab, kk):
        a, b = ab
        return np.clip(a + b * (P[kk] - P[kk].mean()), 1e-6, 1 - 1e-6)

    report('ORACLE a+b', [apply(ab_oracle[k], k) for k in range(3)], [Y[k] for k in range(3)])
    # oracle intercept only (= recenter), oracle slope only
    report('ORACLE a only', [np.clip(Y[k].mean() + (P[k] - P[k].mean()), 1e-6, 1 - 1e-6) for k in range(3)],
           [Y[k] for k in range(3)])
    report('ORACLE b only', [np.clip(P[k].mean() + ab_oracle[k][1] * (P[k] - P[k].mean()), 1e-6, 1 - 1e-6) for k in range(3)],
           [Y[k] for k in range(3)])

    # honest: slope from backtest, intercept from season-r extrapolation ('last')
    FTS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022, 2023]}
    for a_mode in ['last', 'last_plus_meandelta', 'backtest_a']:
        for b_src in ['backtest_b', 'fixed0.8', 'fixed0.6', 'fixed0.5', 'fixed1.0']:
            pl = []
            for k in range(3):
                if a_mode == 'last':
                    a = SEASON_R[FTS[k][-1]]
                elif a_mode == 'last_plus_meandelta':
                    ys = np.array([SEASON_R[s] for s in FTS[k]]); a = ys[-1] + np.diff(ys).mean()
                else:
                    a = ab_bt[k][0]
                b = ab_bt[k][1] if b_src == 'backtest_b' else float(b_src.replace('fixed', ''))
                pl.append(np.clip(a + b * (P[k] - P[k].mean()), 1e-6, 1 - 1e-6))
            report(f'a={a_mode} b={b_src}', pl, [Y[k] for k in range(3)])

    np.save(CACHE / 'exp06_P.npy', np.concatenate([P[k] for k in range(3)]))
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
