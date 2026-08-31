"""EXP05: era-adjusted recency-weighted shrunk group encodings."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_run import run, AS_OF
from agent3_lib import CACHE, get_fold
from agent3_enc import build_encodings, apply_encodings


def make_enc_fn(cfg=None):
    def fn(k, X_tr, X_va, sd_tr, sd_va):
        _, _, y_tr, _, s_tr = get_fold(k)
        enc = build_encodings(sd_tr, y_tr, s_tr, AS_OF[k], cfg)
        X_tr = apply_encodings(enc, sd_tr, X_tr)
        X_va = apply_encodings(enc, sd_va, X_va)
        return X_tr, X_va
    return fn


if __name__ == '__main__':
    print('--- era-adjusted group encodings (decay=0.8) ---')
    run('era_enc d0.8', extra_fn=make_enc_fn({'decay': 0.8}), seeds=(7, 123))
    print('--- decay=1.0 (no recency in encoding) ---')
    run('era_enc d1.0', extra_fn=make_enc_fn({'decay': 1.0}), seeds=(7, 123))
    print('--- decay=0.6 ---')
    run('era_enc d0.6', extra_fn=make_enc_fn({'decay': 0.6}), seeds=(7, 123))
