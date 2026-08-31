"""
agent2_exp6_confirm5.py — 5-SEED CONFIRMATION of the headline result.

Standard 5 seeds [7, 123, 2025, 31415, 8675309], full 3-GBDT ensemble,
folds val = 2021/2022/2023/2024 (2021 is extra, used only for the shift trend).

Variants:
  base       SSOT 69 features, index bug FIXED
  asof_dec   base + current-season-to-date decomposition of the asof counters
"""
import sys, os, time
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_exp3_runner import fit_predict_raw
from agent2_asof_decomp import AsofDecomposer

OUT = '~/LG_data/scratch/cache_5seed'
os.makedirs(OUT, exist_ok=True)
SEEDS = [7, 123, 2025, 31415, 8675309]


def main(variants, val_seasons, seeds=SEEDS):
    df = pd.read_csv(config.TRAIN_PATH)
    for vs in val_seasons:
        df_tr = df[(df.season >= 2019) & (df.season < vs)].copy()
        df_val = df[df.season == vs].copy()
        t0 = time.time()
        X_tr_b, X_val_b = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(X_tr_b)
        y_tr = df_tr[config.TARGET_COL].values
        y_val = df_val[config.TARGET_COL].values
        dec = AsofDecomposer().fit(df_tr, vs)
        A_tr = dec.transform(df_tr); A_val = dec.transform(df_val)
        log(f"val={vs}: ready in {time.time()-t0:.0f}s")
        for vname in variants:
            f = f'{OUT}/{vname}_val{vs}.npz'
            if os.path.exists(f):
                log(f"  skip {f}"); continue
            if vname == 'base':
                X_tr, X_val = X_tr_b, X_val_b
            elif vname == 'asof_dec':
                X_tr = pd.concat([X_tr_b, A_tr], axis=1); X_val = pd.concat([X_val_b, A_val], axis=1)
            t1 = time.time()
            P = np.zeros((3, len(X_val)))
            for seed in seeds:
                pl, pc, px = fit_predict_raw(X_tr, y_tr, X_val, cc, seed)
                P[0] += pl; P[1] += pc; P[2] += px
            P /= len(seeds)
            np.savez_compressed(f, y=y_val.astype(np.int8), p_lgb=P[0].astype(np.float32),
                                p_cb=P[1].astype(np.float32), p_xgb=P[2].astype(np.float32),
                                val_season=vs, n_features=X_tr.shape[1])
            log(f"  [{vname}] val={vs} saved ({X_tr.shape[1]} feats) in {time.time()-t1:.0f}s")


if __name__ == '__main__':
    variants = sys.argv[1].split(',') if len(sys.argv) > 1 else ['asof_dec', 'base']
    vs = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2022, 2023, 2024, 2021]
    main(variants, vs)
