"""
agent2_exp9_final.py — FINAL configuration, 5 seeds, full 3-GBDT ensemble.

Config chosen on INNER folds only (round-2 screening, agent2_exp8_round2.py):
    asof decomposition v2  +  form ladder  +  historical count-conditional rates
2024 is scored exactly once with this frozen configuration.
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
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from agent2_exp7_extra import form_ladder, HistCondRates

OUT = '~/LG_data/scratch/cache_final'
os.makedirs(OUT, exist_ok=True)
SEEDS = [7, 123, 2025, 31415, 8675309]

if __name__ == '__main__':
    vs_list = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2022, 2023, 2024, 2021]
    df = pd.read_csv(config.TRAIN_PATH)
    L = recover(df)
    for vs in vs_list:
        f = f'{OUT}/final_val{vs}.npz'
        if os.path.exists(f):
            log(f"skip {f}"); continue
        tr = ((df.season >= 2019) & (df.season < vs)).values
        va = (df.season == vs).values
        df_tr = df[tr].copy(); df_val = df[va].copy()
        t0 = time.time()
        Xb_tr, Xb_val = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(Xb_tr)
        dec = AsofDecomposer2().fit(df_tr, vs)
        A_tr = dec.transform(df_tr); A_val = dec.transform(df_val)
        hc = HistCondRates().fit(df_tr, L[tr], vs)
        X_tr = pd.concat([Xb_tr, A_tr, form_ladder(df_tr, A_tr), hc.transform(df_tr)], axis=1)
        X_val = pd.concat([Xb_val, A_val, form_ladder(df_val, A_val), hc.transform(df_val)], axis=1)
        y_tr = df[config.TARGET_COL].values[tr]; y_val = df[config.TARGET_COL].values[va]
        log(f"val={vs}: X={X_tr.shape} ready in {time.time()-t0:.0f}s")
        P = np.zeros((3, len(X_val)))
        for seed in SEEDS:
            pl, pc, px = fit_predict_raw(X_tr, y_tr, X_val, cc, seed)
            P[0] += pl; P[1] += pc; P[2] += px
            log(f"   seed {seed} done")
        P /= len(SEEDS)
        np.savez_compressed(f, y=y_val.astype(np.int8), p_lgb=P[0].astype(np.float32),
                            p_cb=P[1].astype(np.float32), p_xgb=P[2].astype(np.float32),
                            val_season=vs, n_features=X_tr.shape[1])
        log(f"[final] val={vs} saved ({X_tr.shape[1]} feats)")
