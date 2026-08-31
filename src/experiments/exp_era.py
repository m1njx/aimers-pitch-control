"""
exp_era.py — experiment: era-normalised GBDT training.

HYPOTHESIS
----------
The league control-success rate drifts hard: .5647 (2019) -> .4861 (2024).
`season` is deliberately excluded from the feature set (config.py: trees cannot
extrapolate to 2025), so the drift reaches the model only through the asof_*
career aggregates, which mix eras and are therefore biased. outputs/501 measured
this directly: a raw career-rate predictor scores -225 on 2024, while the same
predictor with each season's league mean removed scores +151.

So: give every training row an offset equal to its own season's league level, and
let the model fit only the deviation from that level. At inference, supply the
level projected from the training seasons alone. The model then learns era-invariant
structure and the level is handled explicitly instead of being absorbed into a
post-hoc constant.

SCOPE
-----
GBDT components only (lgb_bin / cb_bin / xgb_bin / lgb_mse) via each library's
native offset hook -- init_score, baseline, base_margin -- so the objective is
unchanged and era-normalisation is the single manipulated variable. The MLP
predictions are copied verbatim from the baseline cache, so a blend built from
this cache differs from baseline in exactly one respect.

This doubles as the missing check from outputs/503 s.4: the harness has only ever
been validated on the blend/calibration axis. This is a TRAINING-side change, and
whether the harness ranks it credibly is itself part of what is being measured.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_era.py --years 2024 2023 2022 --seeds 7 123 2025
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

from build_cache import (build_features, cast_cb, cast_xgb, CAT_COLS, CACHE)
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

ERA_CACHE = os.path.join(LG, 'harness/cache_era')
os.makedirs(ERA_CACHE, exist_ok=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def project_level(rates):
    """Linear extrapolation of the league rate from TRAINING seasons only."""
    s = np.array(sorted(rates)); r = np.array([rates[k] for k in s])
    if len(s) < 2:
        return float(r[-1]), None
    b, a = np.polyfit(s, r, 1)
    return float(a + b * (s[-1] + 1)), (b, a)


def run(df, year, seeds, scale=1.0, out_dir=None):
    """scale=1.0 is full era-normalisation (original experiment). scale=0.0 collapses
    the per-row offset to a single training-wide constant, which GBDT boosting absorbs
    into its own leaf bias -- so scale=0.0 is expected to behave like the unmodified
    baseline up to training noise. That makes scale itself a meaningful, cheap-to-grid
    knob for how aggressively to apply the era correction, without changing any other
    hyperparameter (single manipulated variable preserved)."""
    out_dir = out_dir or ERA_CACHE
    os.makedirs(out_dir, exist_ok=True)
    tr = df[df.season < year]
    va = df[df.season == year]
    rates = tr.groupby('season')['control_success'].mean().to_dict()
    r_bar = tr['control_success'].mean()
    r_hat_full, fit = project_level(rates)
    r_hat = r_bar + scale * (r_hat_full - r_bar)
    r_actual = va['control_success'].mean()
    print(f'\n=== era 실험 eval={year} scale={scale}  train {len(tr):,} ===', flush=True)
    print('  학습시즌 리그율: ' + ' '.join(f'{k}:{v:.4f}' for k, v in sorted(rates.items())), flush=True)
    print(f'  → {year} 투영값 r_hat={r_hat:.4f}  (실제 {r_actual:.4f}, 오차 {r_hat-r_actual:+.4f})', flush=True)

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

    # per-row offsets: the row's own season level for training, projected for eval,
    # each interpolated toward the training-wide mean by `scale`
    season_lvl_tr = tr['season'].map(rates).values.astype(np.float64)
    lvl_tr = r_bar + scale * (season_lvl_tr - r_bar)
    lvl_va = np.full(len(va), r_hat)
    off_tr_id, off_va_id = lvl_tr, lvl_va                 # identity link (regression)
    off_tr_lg, off_va_lg = logit(lvl_tr), logit(lvl_va)   # logit link (classifiers)

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
        d = lgb.Dataset(Xtr, label=ytr, init_score=off_tr_id)
        out['lgb_bin'] = off_va_id + lgb.train(p, d).predict(Xva)   # init_score not in predict()

        d = lgb.Dataset(Xtr133m, label=ytr, init_score=off_tr_id)
        p2 = dict(p); p2['seed'] = seed + 1
        out['lgb_mse'] = off_va_id + lgb.train(p2, d).predict(Xva133m)

        m = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                               random_seed=seed, verbose=0, thread_count=6)
        m.fit(Pool(Xtr_cb, ytr, cat_features=CAT_COLS, baseline=off_tr_lg))
        out['cb_bin'] = m.predict_proba(Pool(Xva_cb, cat_features=CAT_COLS, baseline=off_va_lg))[:, 1]

        dtr = xgb.DMatrix(Xtr_xg, label=ytr, base_margin=off_tr_lg)
        dva = xgb.DMatrix(Xva_xg, base_margin=off_va_lg)
        bst = xgb.train(dict(objective='binary:logistic', eta=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        dtr, num_boost_round=250)
        out['xgb_bin'] = bst.predict(dva)

        # MLP copied unchanged from the baseline cache -> single manipulated variable
        base = np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz'))
        out['mlp'] = base['mlp']

        np.savez_compressed(f, **out)
        print(f'  seed {seed}: era GBDT 4종 학습완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {year} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2024, 2023, 2022])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    for y in a.years:
        run(df, y, a.seeds)
