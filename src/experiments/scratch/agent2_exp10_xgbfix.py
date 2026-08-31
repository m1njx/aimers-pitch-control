"""
agent2_exp10_xgbfix.py — recompute ONLY the XGBoost member with the category
re-encoding bug fixed, for both the base and asof_dec feature sets, 5 seeds.
LightGBM / CatBoost predictions are reused from cache_5seed, so this costs a
third of a full rerun.
"""
import sys, os, time
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import xgboost as xgb
import config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_asof_decomp import AsofDecomposer

OUT = '~/LG_data/scratch/cache_xgbfix'
os.makedirs(OUT, exist_ok=True)
SEEDS = [7, 123, 2025, 31415, 8675309]

def xgb_pred(X_tr, y_tr, X_val, cat_cols, seed, fix):
    A = X_tr.copy(); B = X_val.copy()
    for c in cat_cols:
        if fix:
            A[c] = A[c].astype(np.float32); B[c] = B[c].astype(np.float32)
        else:
            A[c] = A[c].astype('category').cat.codes.astype(np.float32)
            B[c] = B[c].astype('category').cat.codes.astype(np.float32)
    m = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                          colsample_bytree=0.8, subsample=0.8, random_state=seed,
                          n_jobs=-1, eval_metric='logloss')
    m.fit(A.astype(np.float32), y_tr)
    return m.predict_proba(B.astype(np.float32))[:, 1]

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_PATH)
    for vs in [2022, 2023, 2024, 2021]:
        tr = ((df.season >= 2019) & (df.season < vs)).values
        va = (df.season == vs).values
        df_tr = df[tr].copy(); df_val = df[va].copy()
        Xb_tr, Xb_val = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(Xb_tr)
        dec = AsofDecomposer().fit(df_tr, vs)
        sets = {'base': (Xb_tr, Xb_val),
                'asof_dec': (pd.concat([Xb_tr, dec.transform(df_tr)], axis=1),
                             pd.concat([Xb_val, dec.transform(df_val)], axis=1))}
        y_tr = df[config.TARGET_COL].values[tr]; y_val = df[config.TARGET_COL].values[va]
        for name, (X_tr, X_val) in sets.items():
            f = f'{OUT}/{name}_val{vs}.npz'
            if os.path.exists(f): continue
            t0 = time.time(); acc = np.zeros(len(X_val))
            for seed in SEEDS:
                acc += xgb_pred(X_tr, y_tr, X_val, cc, seed, fix=True)
            np.savez_compressed(f, y=y_val.astype(np.int8),
                                p_xgb_fixed=(acc/len(SEEDS)).astype(np.float32))
            log(f"[xgbfix {name}] val={vs} done in {time.time()-t0:.0f}s")
