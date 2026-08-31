"""
exp_recency.py — experiment: season-recency training-row reweighting.

WHY REVISIT THIS
-----------------
outputs/163-169 tried exactly this (decay-weighted training rows, most recent
season weighted highest) and could not reach a conclusion: report 168 found the
original decay=0.7 selection was chosen using a fold average that INCLUDED the
outer (2024) fold, i.e. circular validation, and a clean inner-only reselection
(decay=0.95) actually lost -23.5 on outer. The project's verdict (169) was to
abandon the idea, not because it was disproven, but because the harness at the
time couldn't reliably tell signal from noise (noise floor probe showed a single
outer fold swinging +-31.75 points on identical config).

This is being retested now because the tool that failed then -- an honest,
multi-seed, multi-year harness -- now exists (outputs/503) and specifically fixes
the failure mode that killed the earlier attempt.

DESIGN
------
weight(row) = decay ** (fold_end_season - row_season), decay=1.0 reproduces
uniform weighting (~= baseline, modulo training noise). Applied as native sample
weight to all four GBDT-family components; MLP is left untouched, copied from the
baseline cache, exactly as in exp_era.py, to keep a single manipulated variable.

This is orthogonal to exp_era.py's offset mechanism -- it changes which rows the
loss weights harder, not what level each row is centered on -- so the two can
later be composed (recency weight + era offset together) if both clear noise
independently.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_recency.py --years 2024 2023 2022 --seeds 7 123 2025 --decay 0.85
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

REC_CACHE = os.path.join(LG, 'harness/cache_recency')


def run(df, year, seeds, decay=1.0, out_dir=None, era_offset=None):
    """era_offset: optional (scale, offset_fn) pair to compose with exp_era's
    per-row offset. Left None to test recency in isolation (era_scale=0 equivalent)."""
    out_dir = out_dir or REC_CACHE
    os.makedirs(out_dir, exist_ok=True)
    tr = df[df.season < year]
    va = df[df.season == year]
    print(f'\n=== recency 실험 eval={year} decay={decay}  train {len(tr):,} ===', flush=True)

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

    w = decay ** (year - 1 - tr['season'].values.astype(np.float64))
    w = w * (len(w) / w.sum())   # renormalise so total weight mass is unchanged

    Xtr_cb, Xva_cb = cast_cb(Xtr), cast_cb(Xva)
    Xtr_xg, Xva_xg = cast_xgb(Xtr), cast_xgb(Xva)
    Xtr133m, Xva133m = Xtr133.values.astype(np.float32), Xva133.values.astype(np.float32)

    for seed in seeds:
        f = os.path.join(out_dir, f'pred_{year}_{seed}.npz')
        if os.path.exists(f):
            print(f'  seed {seed}: cached, skip', flush=True); continue
        t1 = time.time(); out = {}

        p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
                 seed=seed, verbose=-1, n_estimators=300, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.8)
        out['lgb_bin'] = lgb.train(p, lgb.Dataset(Xtr, label=ytr, weight=w)).predict(Xva)
        p2 = dict(p); p2['seed'] = seed + 1
        out['lgb_mse'] = lgb.train(p2, lgb.Dataset(Xtr133m, label=ytr, weight=w)).predict(Xva133m)

        m = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                               random_seed=seed, verbose=0, thread_count=6)
        m.fit(Pool(Xtr_cb, ytr, cat_features=CAT_COLS, weight=w))
        out['cb_bin'] = m.predict_proba(Xva_cb)[:, 1]

        dtr = xgb.DMatrix(Xtr_xg, label=ytr, weight=w)
        dva = xgb.DMatrix(Xva_xg)
        bst = xgb.train(dict(objective='binary:logistic', eta=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        dtr, num_boost_round=250)
        out['xgb_bin'] = bst.predict(dva)

        base = np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz'))
        out['mlp'] = base['mlp']

        np.savez_compressed(f, **out)
        print(f'  seed {seed}: recency GBDT 4종 학습완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {year} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2024, 2023, 2022])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025])
    ap.add_argument('--decay', type=float, default=0.85)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    out_dir = os.path.join(LG, f'harness/cache_recency_d{int(round(a.decay*100)):03d}')
    for y in a.years:
        run(df, y, a.seeds, decay=a.decay, out_dir=out_dir)
