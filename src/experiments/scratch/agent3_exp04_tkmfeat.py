"""EXP04: add recovered-ID trackman pitcher features."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_run import run, AS_OF
from agent3_lib import CACHE

_cacheF = {}


def tkm_extra(k, X_tr, X_va, sd_tr, sd_va):
    aof = AS_OF[k]
    if aof not in _cacheF:
        _cacheF[aof] = pd.read_parquet(CACHE / f'tkm_pfeat_{aof}.parquet')
    F = _cacheF[aof]
    for X, sd in [(X_tr, sd_tr), (X_va, sd_va)]:
        add = F.reindex(sd['pitcher_id'].values)
        add.index = X.index
        for c in add.columns:
            X[c] = add[c].values
    return X_tr, X_va


if __name__ == '__main__':
    print('--- baseline (no tkm-id feats) ---')
    run('base', seeds=(7, 123))
    print('--- + trackman pitcher features (93) ---')
    run('base+tkmP', extra_fn=tkm_extra, seeds=(7, 123))
