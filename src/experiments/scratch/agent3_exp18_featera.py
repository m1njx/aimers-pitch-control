"""EXP18: within-season standardisation of the asof_* FEATURES (feature-side era neutralisation).

EXP07 neutralised the era drift on the TARGET side. But the inputs drift too: a pitcher with
asof_pitcher_success_rate = 0.53 is league-average in 2021 and clearly above-average in 2024.
The model cannot know which era a row belongs to (`season` is excluded), so the same feature
value means different things in different rows.

Fix: express each asof_* feature relative to its own season (x - mu_s)/sd_s. For the
validation/test season we must NOT look at its own rows (competition rule bans test-internal
distribution statistics), so mu/sd are EXTRAPOLATED from the train seasons.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_lib import get_fold, cat_cols_of, report, CACHE
from agent3_run import LGB_L2
from agent3_calib import calibrate
from agent3_tkm_sit import attach
import lightgbm as lgb

AS_OF = {0: 2021, 1: 2022, 2: 2023}
VAL = {0: 2022, 1: 2023, 2: 2024}
SEEDS = (7, 123)
SCHEMES = [('seg_relative', 'last', 'count', 1.0),
           ('seg_relative', 'last_plus_halfdelta', 'count', 1.0)]

ASOF = None  # resolved at runtime


def extrap(vals, seasons, target, mode):
    v = np.asarray(vals, float)
    if mode == 'last':
        return v[-1]
    if mode == 'lin':
        b, a = np.polyfit(np.asarray(seasons, float), v, 1)
        return a + b * target
    if mode == 'half':
        return v[-1] + 0.5 * np.diff(v).mean() * (target - seasons[-1])
    raise ValueError(mode)


def run(tag, cols_mode, ex_mode, params, use_sit=True):
    t0 = time.time()
    R, Y, SV, ST, YT = {}, {}, {}, {}, {}
    for k in range(3):
        X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va = get_fold(k, side=True)
        if use_sit:
            F = pd.read_parquet(CACHE / f'tkm_sit_{AS_OF[k]}.parquet')
            X_tr = attach(F, sd_tr, X_tr, list(F.columns))
            X_va = attach(F, sd_va, X_va, list(F.columns))
        if cols_mode == 'asof':
            cols = [c for c in X_tr.columns if c.startswith('asof_') or c.startswith('pitcher_success_trend')]
        elif cols_mode == 'rates':
            cols = [c for c in X_tr.columns
                    if ('rate' in c and c.startswith('asof_')) or c.startswith('pitcher_success_trend')]
        else:
            cols = []
        seasons = sorted(np.unique(s_tr))
        for c in cols:
            v = X_tr[c].values.astype(np.float64)
            mus, sds = [], []
            for s in seasons:
                m = s_tr == s
                mus.append(np.nanmean(v[m])); sds.append(np.nanstd(v[m]) + 1e-9)
            mu_map = dict(zip(seasons, mus)); sd_map = dict(zip(seasons, sds))
            X_tr[c] = ((v - pd.Series(s_tr).map(mu_map).values) /
                       pd.Series(s_tr).map(sd_map).values).astype(np.float32)
            mv = extrap(mus, seasons, VAL[k], ex_mode)
            sv = extrap(sds, seasons, VAL[k], ex_mode)
            X_va[c] = ((X_va[c].values.astype(np.float64) - mv) / sv).astype(np.float32)
        rs = pd.Series(y_tr).groupby(s_tr).mean()
        yt = y_tr - pd.Series(s_tr).map(rs).values
        cc = cat_cols_of(X_tr); cidx = [X_tr.columns.get_loc(c) for c in cc]
        p = np.zeros(len(X_va))
        for sd in SEEDS:
            pr = dict(LGB_L2); pr.update(params); pr['random_state'] = sd
            m = lgb.LGBMRegressor(**pr); m.fit(X_tr, yt, categorical_feature=cidx)
            p += m.predict(X_va)
        R[k] = p / len(SEEDS); Y[k], SV[k], ST[k], YT[k] = y_va, sd_va, sd_tr, y_tr
        del X_tr, X_va
    for sch, mode, kind, al in SCHEMES:
        pl = [calibrate(R[k], SV[k], ST[k], YT[k], k, sch, mode, kind, al) for k in range(3)]
        report(f'{tag} | {mode}', pl, [Y[k] for k in range(3)])
    print(f'   {time.time()-t0:.0f}s', flush=True)
    return R, Y, SV, ST, YT


if __name__ == '__main__':
    P = dict(num_leaves=10, n_estimators=1500, learning_rate=0.01)
    run('ref (no feat-era)', 'none', 'last', P)
    run('featera[asof] ex=last', 'asof', 'last', P)
    run('featera[asof] ex=lin', 'asof', 'lin', P)
    run('featera[asof] ex=half', 'asof', 'half', P)
    run('featera[rates] ex=last', 'rates', 'last', P)
