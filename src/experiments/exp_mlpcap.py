"""
exp_mlpcap.py — experiment: SimpleMLP capacity / training length.

WHY THIS MATTERS AS MUCH AS THE GBDT CAPACITY SWEEP
---------------------------------------------------
The MLP carries 40-50% of the final blend weight (v42: W_MLP_MSE 0.40, v50: 0.50 --
and v50 is the current leaderboard best). Yet it is trained as:

    hidden = (128, 64), dropout 0.12, 5 epochs, Adam lr 1e-3, batch 2048

on 1,475,092 rows x 133 features. Five epochs of a two-layer 128/64 MLP is a very
small amount of fitting for that much data, and the width is narrow relative to the
input dimension. If this component is undertrained, half the blend is being carried
by an underpowered model -- and unlike the GBDT side, nothing here was ever swept
in any surviving report.

A second reason to look here specifically: outputs/500 showed v48 (which changed
ONLY the MLP -- 5-seed plain -> 15-seed SWA) moved the real leaderboard by 12 points.
The MLP is demonstrably a live lever on the actual score, not a rounding detail.

LEVELS
------
    M1  (128, 64)       5 epochs   <- production baseline (already in harness/cache)
    M2  (256, 128)      8 epochs
    M3  (512, 256, 128) 12 epochs

GBDT predictions are copied verbatim from the baseline cache, so the MLP is the
single manipulated variable -- the mirror image of exp_capacity.py, which holds the
MLP fixed and varies the GBDTs.

Inner years only for selection (nested-validation rule). This is a screen: 2 seeds.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_mlpcap.py --levels M2 M3
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
sys.path.insert(0, os.path.join(LG, 'work/submit_v42'))
sys.path.insert(0, LG)

import torch
import torch.nn as nn

from build_cache import (build_features, mlp_arrays, SimpleMLP_MSE, CACHE, CAT_COLS)
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

LEVELS = {
    'M1': dict(hidden=(128, 64),       epochs=5,  dropout=0.12, lr=1e-3),
    'M2': dict(hidden=(256, 128),      epochs=8,  dropout=0.15, lr=1e-3),
    'M3': dict(hidden=(512, 256, 128), epochs=12, dropout=0.20, lr=1e-3),
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
    print(f'\n=== mlpcap {level} eval={year}  train {len(tr):,}  cfg={cfg} ===', flush=True)
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

    _, Xtr133 = build_features(tr, prep, dec, cat_map)
    _, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)
    nz_tr, ca_tr, art = mlp_arrays(Xtr133)
    nz_va, ca_va, _ = mlp_arrays(Xva133, art)
    print(f'  features built ({time.time()-t0:.0f}s)', flush=True)

    for seed in need:
        t1 = time.time()
        torch.manual_seed(seed)
        net = SimpleMLP_MSE(len(art['num_cols']), art['cards'],
                            hidden=cfg['hidden'], dropout=cfg['dropout'])
        opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        crit = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(torch.tensor(nz_tr), torch.tensor(ca_tr),
                                            torch.tensor(ytr, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
        net.train()
        for ep in range(cfg['epochs']):
            tot = 0.0
            for bn, bc, by in dl:
                opt.zero_grad()
                loss = crit(net(bn, bc), by)
                loss.backward(); opt.step()
                tot += float(loss) * len(by)
            print(f'    epoch {ep+1}/{cfg["epochs"]} mse={tot/len(ytr):.6f} ({time.time()-t1:.0f}s)', flush=True)
        net.eval()
        with torch.no_grad():
            mlp_pred = net(torch.tensor(nz_va), torch.tensor(ca_va)).numpy().astype(np.float64)

        # GBDT components copied unchanged -> MLP is the only manipulated variable
        base = dict(np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz')))
        base['mlp'] = mlp_pred
        np.savez_compressed(os.path.join(out_dir, f'pred_{year}_{seed}.npz'), **base)
        print(f'  seed {seed}: {level} 완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {level} {year} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123])
    ap.add_argument('--levels', nargs='+', default=['M2', 'M3'])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    from exp_capacity import score_dir

    dirs = {'M1': CACHE}
    for lv in a.levels:
        d = os.path.join(LG, f'harness/cache_mlp_{lv}')
        for y in a.years:
            run(df, y, a.seeds, lv, d)
        dirs[lv] = d

    print('\n=== MLP capacity 스캔 결과 (inner 전용) ===', flush=True)
    res = {}
    for lv, d in dirs.items():
        r = score_dir(d, a.years, a.seeds)
        if r:
            res[lv] = r
            print(f'  {lv}: inner={r["inner"]:8.1f}  연도별={ {k: round(v,1) for k,v in r["season_mean"].items()} }  seed_sd={r["seed_sd"]:.1f}', flush=True)
    if 'M1' in res and len(res) > 1:
        b = res['M1']['inner']
        noise = float(np.mean([v['seed_sd'] for v in res.values() if not np.isnan(v['seed_sd'])]))
        best = max(res, key=lambda k: res[k]['inner'])
        print(f'\n  → 최고={best}  M1(현행) 대비 델타={res[best]["inner"]-b:+.1f}  노이즈={noise:.1f}')
        print(f'  → 신뢰가능={bool(res[best]["inner"]-b > noise and best != "M1")}')
