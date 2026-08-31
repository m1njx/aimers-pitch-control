"""
agent2_exp4_runner.py — evaluates the asof algebraic-decomposition features
(agent2_asof_decomp.py) with the standard 3-GBDT ensemble, caching raw
per-model predictions like agent2_exp3_runner.py.

variants:
  asof_dec      base 69 features + current-season-to-date decomposition
  asof_dec_min  base + only the 4 strongest decomposition columns
  asof_dec_prof base + decomposition + trackman per-pitcher profile
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
from agent2_exp3_runner import fit_predict_raw, OUT, FIX_INDEX
from agent2_asof_decomp import AsofDecomposer
from agent2_tkm_profile import PitcherTrackmanProfile, load_pitcher_map

SEEDS = [7, 123]
MIN_COLS = ['cs_p_succ_rate', 'cs_p_succ_eb', 'cs_b_succ_rate', 'cs_pb_succ_sum']


def main(variants, val_seasons, seeds=SEEDS):
    df = pd.read_csv(config.TRAIN_PATH)
    need_tkm = any('prof' in v for v in variants)
    pmap = load_pitcher_map() if need_tkm else None
    tm_full = pd.read_csv(config.TRACKMAN_PATH) if need_tkm else None

    for vs in val_seasons:
        df_tr = df[(df.season >= 2019) & (df.season < vs)].copy()
        df_val = df[df.season == vs].copy()
        as_of = vs - 1
        t0 = time.time()
        X_tr_b, X_val_b = build_base_features(df_tr, df_val, as_of, fix_index=FIX_INDEX)
        cc = base_cat_cols(X_tr_b)
        y_tr = df_tr[config.TARGET_COL].values
        y_val = df_val[config.TARGET_COL].values
        dec = AsofDecomposer().fit(df_tr, vs)
        A_tr = dec.transform(df_tr); A_val = dec.transform(df_val)
        log(f"val={vs}: base+decomp ready ({A_tr.shape[1]} new cols) in {time.time()-t0:.0f}s")
        prof = None
        if need_tkm:
            prof = PitcherTrackmanProfile(pmap).fit(tm_full[tm_full.season <= as_of])

        for vname in variants:
            f = f'{OUT}/{vname}_val{vs}.npz'
            if os.path.exists(f):
                log(f"  skip {f}"); continue
            if vname == 'asof_dec':
                X_tr = pd.concat([X_tr_b, A_tr], axis=1); X_val = pd.concat([X_val_b, A_val], axis=1)
            elif vname == 'asof_dec_min':
                X_tr = pd.concat([X_tr_b, A_tr[MIN_COLS]], axis=1)
                X_val = pd.concat([X_val_b, A_val[MIN_COLS]], axis=1)
            elif vname == 'asof_dec_prof':
                X_tr = pd.concat([X_tr_b, A_tr, prof.transform(df_tr)], axis=1)
                X_val = pd.concat([X_val_b, A_val, prof.transform(df_val)], axis=1)
            else:
                raise ValueError(vname)
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
    variants = sys.argv[1].split(',')
    vs = [int(x) for x in sys.argv[2].split(',')]
    main(variants, vs)
