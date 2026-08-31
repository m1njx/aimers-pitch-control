"""EXP13: hyperparameter search for the era-target LGBM on the EXP10 winning feature set."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, '~/LG_data/scratch')
from agent3_exp12_bsit import run_parts

GRID = [
    ('base250/45', {}),
    ('n500 lr.03', dict(n_estimators=500, learning_rate=0.03)),
    ('n800 lr.02', dict(n_estimators=800, learning_rate=0.02)),
    ('leaves31', dict(num_leaves=31)),
    ('leaves63', dict(num_leaves=63)),
    ('leaves127 n500 lr.03', dict(num_leaves=127, n_estimators=500, learning_rate=0.03)),
    ('mcs200', dict(min_child_samples=200)),
    ('mcs1000', dict(min_child_samples=1000)),
    ('mcs1000 n800 lr.02', dict(min_child_samples=1000, n_estimators=800, learning_rate=0.02)),
    ('l2reg10', dict(reg_lambda=10.0)),
    ('colsample.5', dict(colsample_bytree=0.5)),
    ('deep: n800 lr.02 leaves127 mcs500',
     dict(n_estimators=800, learning_rate=0.02, num_leaves=127, min_child_samples=500)),
]

if __name__ == '__main__':
    for tag, p in GRID:
        run_parts(f'HP[{tag}]', ['sit'], seeds=(7,), params=p)
