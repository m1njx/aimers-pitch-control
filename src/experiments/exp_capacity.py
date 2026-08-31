"""
exp_capacity.py — experiment: GBDT model capacity.

WHY THIS IS THE STRONGEST REMAINING LEAD
-----------------------------------------
The production recipe trains, on 1,475,092 rows x 119-133 features:

    LightGBM   300 trees,  31 leaves, lr 0.05
    CatBoost   300 iters,  depth 6,   lr 0.06
    XGBoost    250 rounds, depth 5,   lr 0.05

Those are small for a dataset of this size. 300 trees at 31 leaves is roughly
9k leaf nodes total against 1.5M rows -- on the order of 160 rows per leaf if
they were evenly filled, and boosting concentrates them far more unevenly than
that. Nothing in the surviving reports shows this was ever swept; the numbers look
like an early default that got frozen when the pipeline started working.

Unlike era-normalisation and recency-reweighting (both closed by outputs/504),
this axis has a clear mechanism to be wrong in a recoverable direction: if the
models are underfit, added capacity is a monotone gain until it isn't, and the
inner folds will show it.

The counter-hypothesis is real and must be taken seriously: the target is
extremely low-signal (the whole model beats the base rate by ~1% of Brier), so
small capacity may be correct regularisation rather than an oversight. A monotone
DECLINE with capacity would say exactly that, and would be a useful result too.

PROTOCOL
--------
Inner years only (2022, 2023) for selection, per the nested-validation rule.
2 seeds per level to keep runtime sane -- this is a screen, not a final answer;
anything that clears noise gets re-run at full seed count before being trusted.
MLP copied verbatim from the baseline cache: capacity is the only manipulated
variable, and the baseline level is already cached by build_cache.py.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_capacity.py --levels L2 L3
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
sys.path.insert(0, os.path.join(LG, 'work/submit_v42'))
sys.path.insert(0, LG)

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb

from build_cache import build_features, cast_cb, cast_xgb, CAT_COLS, CACHE
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

# L1 is the production recipe and is already cached by build_cache.py.
LEVELS = {
    # L0/L05 added 2026-08-24 AFTER the first sweep came back monotone-declining
    # (L1 1407.1 > L2 1335.0 > L3 1002.8, with seed_sd exploding 5.5 -> 20.5 -> 93.1).
    # That gradient points DOWNWARD in capacity, so the honest follow-up is to ask
    # whether L1 is itself already past the optimum rather than at it.
    # Caveat: submission #2 (leaves=15, min_child=500, lr=0.02) scored 684.98 on the
    # real leaderboard, -138.97 -- over-regularisation has failed before. But that was
    # the 69-feature pre-asof_dec pipeline; the feature set has changed completely since.
    'L0':  dict(lgb_n=150,  lgb_leaves=15,  lgb_mcs=100, cb_it=150,  cb_depth=4, xgb_n=125,  xgb_depth=4),
    'L05': dict(lgb_n=200,  lgb_leaves=23,  lgb_mcs=75,  cb_it=200,  cb_depth=5, xgb_n=175,  xgb_depth=4),
    'L1': dict(lgb_n=300,  lgb_leaves=31,  lgb_mcs=50,  cb_it=300,  cb_depth=6, xgb_n=250,  xgb_depth=5),
    'L2': dict(lgb_n=900,  lgb_leaves=63,  lgb_mcs=100, cb_it=900,  cb_depth=6, xgb_n=750,  xgb_depth=6),
    'L3': dict(lgb_n=2000, lgb_leaves=127, lgb_mcs=200, cb_it=2000, cb_depth=8, xgb_n=1500, xgb_depth=7),
}


def run(df, year, seeds, level, out_dir):
    cfg = LEVELS[level]
    os.makedirs(out_dir, exist_ok=True)
    need = [s for s in seeds if not os.path.exists(os.path.join(out_dir, f'pred_{year}_{s}.npz'))]
    if not need:
        print(f'  {level} year={year}: 전부 캐시됨, 생략', flush=True)
        return

    tr = df[df.season < year]
    va = df[df.season == year]
    print(f'\n=== capacity {level} eval={year}  train {len(tr):,}  cfg={cfg} ===', flush=True)
    t0 = time.time()

    prep = PitchPreprocessor()
    prep.fit(tr, as_of_season=year - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    bs = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cs = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          tr['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
    dec = AsofDecomposer2(); dec.fit(tr, val_season=year)

    Xtr, Xtr133 = build_features(tr, prep, dec, cat_map)
    Xva, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)
    print(f'  features built ({time.time()-t0:.0f}s)', flush=True)

    Xtr_cb, Xva_cb = cast_cb(Xtr), cast_cb(Xva)
    Xtr_xg, Xva_xg = cast_xgb(Xtr), cast_xgb(Xva)
    Xtr133m, Xva133m = Xtr133.values.astype(np.float32), Xva133.values.astype(np.float32)

    for seed in need:
        t1 = time.time(); out = {}
        p = dict(objective='regression', metric='rmse', learning_rate=0.05,
                 num_leaves=cfg['lgb_leaves'], seed=seed, verbose=-1,
                 n_estimators=cfg['lgb_n'], min_child_samples=cfg['lgb_mcs'],
                 subsample=0.8, colsample_bytree=0.8, num_threads=6)
        out['lgb_bin'] = lgb.train(p, lgb.Dataset(Xtr, label=ytr)).predict(Xva)
        print(f'    lgb_bin {time.time()-t1:.0f}s', flush=True)
        p2 = dict(p); p2['seed'] = seed + 1
        out['lgb_mse'] = lgb.train(p2, lgb.Dataset(Xtr133m, label=ytr)).predict(Xva133m)
        print(f'    lgb_mse {time.time()-t1:.0f}s', flush=True)

        m = CatBoostClassifier(iterations=cfg['cb_it'], learning_rate=0.06,
                               depth=cfg['cb_depth'], random_seed=seed, verbose=0,
                               thread_count=6)
        m.fit(Pool(Xtr_cb, ytr, cat_features=CAT_COLS))
        out['cb_bin'] = m.predict_proba(Xva_cb)[:, 1]
        print(f'    cb_bin  {time.time()-t1:.0f}s', flush=True)

        bst = xgb.train(dict(objective='binary:logistic', eta=0.05,
                             max_depth=cfg['xgb_depth'], subsample=0.8,
                             colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        xgb.DMatrix(Xtr_xg, label=ytr), num_boost_round=cfg['xgb_n'])
        out['xgb_bin'] = bst.predict(xgb.DMatrix(Xva_xg))
        print(f'    xgb_bin {time.time()-t1:.0f}s', flush=True)

        base = np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz'))
        out['mlp'] = base['mlp']
        np.savez_compressed(os.path.join(out_dir, f'pred_{year}_{seed}.npz'), **out)
        print(f'  seed {seed}: {level} 완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {level} {year} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


def score_dir(d, years, seeds):
    from evaluate import PROD, predict as bp, skill
    per = {}
    for y in years:
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        sc = []
        for s in seeds:
            f = os.path.join(d, f'pred_{y}_{s}.npz')
            if os.path.exists(f):
                sc.append(skill(bp(dict(PROD), dict(np.load(f))), yv))
        if sc:
            per[y] = sc
    if not per:
        return None
    means = {y: float(np.mean(v)) for y, v in per.items()}
    sd = [np.std(v, ddof=1) for v in per.values() if len(v) > 1]
    return dict(inner=float(np.mean(list(means.values()))), season_mean=means,
                seed_sd=float(np.mean(sd)) if sd else float('nan'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123])
    ap.add_argument('--levels', nargs='+', default=['L2', 'L3'])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    dirs = {'L1': CACHE}
    for lv in a.levels:
        d = os.path.join(LG, f'harness/cache_cap_{lv}')
        for y in a.years:
            run(df, y, a.seeds, lv, d)
        dirs[lv] = d

    print('\n=== capacity 스캔 결과 (inner 전용) ===', flush=True)
    res = {}
    for lv, d in dirs.items():
        r = score_dir(d, a.years, a.seeds)
        if r:
            res[lv] = r
            print(f'  {lv}: inner={r["inner"]:8.1f}  연도별={ {k: round(v,1) for k,v in r["season_mean"].items()} }  seed_sd={r["seed_sd"]:.1f}', flush=True)
    if 'L1' in res and len(res) > 1:
        b = res['L1']['inner']
        noise = float(np.mean([v['seed_sd'] for v in res.values() if not np.isnan(v['seed_sd'])]))
        best = max(res, key=lambda k: res[k]['inner'])
        print(f'\n  → 최고={best}  L1(현행) 대비 델타={res[best]["inner"]-b:+.1f}  노이즈={noise:.1f}')
        print(f'  → 신뢰가능={bool(res[best]["inner"]-b > noise and best != "L1")}')
