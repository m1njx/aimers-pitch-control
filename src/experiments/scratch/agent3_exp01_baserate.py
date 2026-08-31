"""EXP01: Base-rate drift diagnostic.

Hypothesis: the metric's baseline is r*(1-r) with r = VAL fold's own mean.
Season base rate drifts hard (0.5647 -> 0.4861). A model trained on past seasons
predicts the past mean -> large reliability penalty -> hundreds of skill points lost.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, calc_skill, calc_raw_brier, report
import lightgbm as lgb

SEASON_R = {2019: 0.564670, 2020: 0.532712, 2021: 0.532762, 2022: 0.528920,
            2023: 0.499957, 2024: 0.486105}
FOLD_TRAIN_SEASONS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022],
                      2: [2019, 2020, 2021, 2022, 2023]}
VAL_SEASON = {0: 2022, 1: 2023, 2: 2024}


def extrapolate(seasons, target_season, mode):
    ys = np.array([SEASON_R[s] for s in seasons])
    xs = np.array(seasons, dtype=float)
    if mode == 'trainmean':
        return None  # handled separately (row-weighted)
    if mode == 'last':
        return ys[-1]
    if mode == 'lin_all':
        b, a = np.polyfit(xs, ys, 1)
        return a + b * target_season
    if mode == 'lin_last3':
        b, a = np.polyfit(xs[-3:], ys[-3:], 1)
        return a + b * target_season
    if mode == 'last_plus_meandelta':
        d = np.diff(ys).mean()
        return ys[-1] + d * (target_season - seasons[-1])
    if mode == 'last_plus_lastdelta':
        d = ys[-1] - ys[-2]
        return ys[-1] + d * (target_season - seasons[-1])
    raise ValueError(mode)


def main():
    t0 = time.time()
    preds, ys, info = {}, {}, {}
    for k in range(3):
        X_tr, X_va, y_tr, y_va, s_tr = get_fold(k)
        cc = cat_cols_of(X_tr)
        cidx = [X_tr.columns.get_loc(c) for c in cc]
        m = lgb.LGBMRegressor(objective='regression', n_estimators=250, num_leaves=45,
                             learning_rate=0.05, min_child_samples=20, colsample_bytree=0.8,
                             subsample=0.8, random_state=7, verbosity=-1, n_jobs=-1)
        m.fit(X_tr, y_tr, categorical_feature=cidx)
        p = m.predict(X_va)
        preds[k] = p
        ys[k] = y_va
        info[k] = dict(train_row_mean=float(y_tr.mean()), pred_mean=float(p.mean()),
                       true_r=float(y_va.mean()))
        print(f"fold{k} val={VAL_SEASON[k]}: train_row_mean={y_tr.mean():.4f} "
              f"pred_mean={p.mean():.4f} TRUE_r={y_va.mean():.4f} "
              f"gap={p.mean()-y_va.mean():+.4f}  ({time.time()-t0:.0f}s)")

    print()
    base = report('no-shift (raw)', [preds[k] for k in range(3)], [ys[k] for k in range(3)])
    # oracle shift
    orc = [preds[k] + (ys[k].mean() - preds[k].mean()) for k in range(3)]
    report('ORACLE recenter', orc, [ys[k] for k in range(3)])
    # extrapolation rules (legit, train-only)
    for mode in ['last', 'lin_all', 'lin_last3', 'last_plus_meandelta', 'last_plus_lastdelta']:
        sh = []
        rh_list = []
        for k in range(3):
            rh = extrapolate(FOLD_TRAIN_SEASONS[k], VAL_SEASON[k], mode)
            rh_list.append(round(float(rh), 4))
            sh.append(np.clip(preds[k] + (rh - preds[k].mean()), 1e-6, 1 - 1e-6))
        report(f'recenter[{mode}]', sh, [ys[k] for k in range(3)], extra=f'r_hat={rh_list}')

    np.save('~/LG_data/scratch/agent3_cache/exp01_preds.npy',
            np.concatenate([preds[k] for k in range(3)]))
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
