"""
exp_ebm.py — experiment: AsofDecomposer2 EB shrinkage constant (eb_m).

WHY
---
`cs_*_eb` is the single most load-bearing derived feature family in this pipeline:
outputs/501 measured the pitcher season-to-date channel at ~604 points on 2024,
far above every other signal source (batter ~47, count ~49, pitcher x count = 0).
That feature is produced by shrinking the season-to-date rate toward the prior with
a fixed constant, `AsofDecomposer2(eb_m=150.0)`.

150 appears to have been chosen once and never revisited. An independent probe in
outputs/501 put the optimum for a pitcher-only EB predictor near k2 ~= 100, which is
close but not identical -- close enough that it is worth one clean test, and far
enough that a real difference is plausible.

CAVEAT (state up front, do not oversell)
----------------------------------------
The GBDTs receive the raw `cs_*_rate`, the shrunk `cs_*_eb`, AND the underlying
counts `cs_*_cur_n` / `cs_*_hist_n` simultaneously. A tree ensemble can therefore
approximate its own shrinkage from those inputs, which makes it quite likely that
eb_m is already close to irrelevant to the fitted model. Expected effect is small.
This is being run because it is cheap and unambiguous, not because it is promising.

Same protocol as exp_era / exp_recency: inner years only for selection, GBDT
components retrained, MLP copied verbatim from the baseline cache so eb_m is the
single manipulated variable.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_ebm.py --years 2022 2023 --seeds 7 123 2025 --ebm 50 150 400 1000
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


def run(df, year, seeds, eb_m, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    need = [s for s in seeds if not os.path.exists(os.path.join(out_dir, f'pred_{year}_{s}.npz'))]
    if not need:
        print(f'  eb_m={eb_m} year={year}: 전부 캐시됨, 생략', flush=True)
        return

    tr = df[df.season < year]
    va = df[df.season == year]
    print(f'\n=== ebm 실험 eval={year} eb_m={eb_m}  train {len(tr):,} ===', flush=True)
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

    dec = AsofDecomposer2(eb_m=float(eb_m))      # <-- the single manipulated variable
    dec.fit(tr, val_season=year)

    Xtr, Xtr133 = build_features(tr, prep, dec, cat_map)
    Xva, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)
    print(f'  features built ({time.time()-t0:.0f}s)', flush=True)

    Xtr_cb, Xva_cb = cast_cb(Xtr), cast_cb(Xva)
    Xtr_xg, Xva_xg = cast_xgb(Xtr), cast_xgb(Xva)
    Xtr133m, Xva133m = Xtr133.values.astype(np.float32), Xva133.values.astype(np.float32)

    for seed in need:
        t1 = time.time(); out = {}
        p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
                 seed=seed, verbose=-1, n_estimators=300, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.8)
        out['lgb_bin'] = lgb.train(p, lgb.Dataset(Xtr, label=ytr)).predict(Xva)
        p2 = dict(p); p2['seed'] = seed + 1
        out['lgb_mse'] = lgb.train(p2, lgb.Dataset(Xtr133m, label=ytr)).predict(Xva133m)

        m = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                               random_seed=seed, verbose=0, thread_count=6)
        m.fit(Pool(Xtr_cb, ytr, cat_features=CAT_COLS))
        out['cb_bin'] = m.predict_proba(Xva_cb)[:, 1]

        bst = xgb.train(dict(objective='binary:logistic', eta=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        xgb.DMatrix(Xtr_xg, label=ytr), num_boost_round=250)
        out['xgb_bin'] = bst.predict(xgb.DMatrix(Xva_xg))

        base = np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz'))
        out['mlp'] = base['mlp']

        np.savez_compressed(os.path.join(out_dir, f'pred_{year}_{seed}.npz'), **out)
        print(f'  seed {seed}: 완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {year} eb_m={eb_m} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025])
    ap.add_argument('--ebm', type=float, nargs='+', default=[50, 150, 400, 1000])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    sys.path.insert(0, os.path.join(LG, 'harness'))
    from evaluate import PROD, predict as blend_predict, skill

    dirs = {}
    for m in a.ebm:
        d = os.path.join(LG, f'harness/cache_ebm_{int(m):04d}')
        for y in a.years:
            run(df, y, a.seeds, m, d)
        dirs[m] = d

    print('\n=== eb_m 스캔 결과 (inner 전용) ===', flush=True)
    rows = []
    for m, d in dirs.items():
        per = {}
        for y in a.years:
            yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
            sc = []
            for s in a.seeds:
                f = os.path.join(d, f'pred_{y}_{s}.npz')
                if os.path.exists(f):
                    sc.append(skill(blend_predict(dict(PROD), dict(np.load(f))), yv))
            if sc:
                per[y] = sc
        if not per:
            continue
        means = {y: float(np.mean(v)) for y, v in per.items()}
        inner = float(np.mean(list(means.values())))
        sd = float(np.mean([np.std(v, ddof=1) for v in per.values() if len(v) > 1]))
        rows.append((m, inner, means, sd))
        print(f'  eb_m={m:<7} inner={inner:8.1f}  연도별={ {k: round(v,1) for k,v in means.items()} }  seed_sd={sd:.1f}', flush=True)

    if rows:
        base = [r for r in rows if abs(r[0] - 150.0) < 1e-9]
        b = base[0][1] if base else float('nan')
        best = max(rows, key=lambda r: r[1])
        noise = float(np.mean([r[3] for r in rows]))
        print(f'\n  → 최고 eb_m={best[0]}  현행(150) 대비 델타={best[1]-b:+.1f}  노이즈={noise:.1f}')
        print(f'  → 신뢰가능={bool(best[1]-b > noise)}')
