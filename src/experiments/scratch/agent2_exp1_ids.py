"""
agent2_exp1_ids.py — Experiment 1: raw pitcher_id / batter_id as categorical features.

MOTIVATION (fresh-eyes finding): config.ID_ONLY_COLS excludes pitcher_id and
batter_id from MODEL_FEATURE_COLS entirely. The model has NEVER seen the raw
entity ids. asof_pitcher_* rates are career-cumulative summaries, but the tree
can never build pitcher x situation interactions without the id itself.
CatBoost handles high-cardinality categoricals with ordered target statistics,
which is exactly the right tool here.

2-seed screening first.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import pandas as pd
import numpy as np
from agent2_common import run_variants, print_summary, log, PROBE_SEEDS

SEEDS = [7, 123]


def v_base(df_tr, df_val, as_of, X_tr, X_val, cc):
    return X_tr, X_val, cc


def v_pid(df_tr, df_val, as_of, X_tr, X_val, cc):
    X_tr['pitcher_id'] = df_tr['pitcher_id'].values.astype(int)
    X_val['pitcher_id'] = df_val['pitcher_id'].values.astype(int)
    return X_tr, X_val, cc + ['pitcher_id']


def v_pid_bid(df_tr, df_val, as_of, X_tr, X_val, cc):
    X_tr['pitcher_id'] = df_tr['pitcher_id'].values.astype(int)
    X_val['pitcher_id'] = df_val['pitcher_id'].values.astype(int)
    X_tr['batter_id'] = df_tr['batter_id'].values.astype(int)
    X_val['batter_id'] = df_val['batter_id'].values.astype(int)
    return X_tr, X_val, cc + ['pitcher_id', 'batter_id']


def v_pid_x_count(df_tr, df_val, as_of, X_tr, X_val, cc):
    """pitcher_id and an explicit pitcher x count cross categorical."""
    for df_src, X_dst in [(df_tr, X_tr), (df_val, X_val)]:
        X_dst['pitcher_id'] = df_src['pitcher_id'].values.astype(int)
        X_dst['pitcher_x_count'] = (df_src['pitcher_id'].astype(int) * 100
                                    + df_src['balls_before'].fillna(0).astype(int) * 10
                                    + df_src['strikes_before'].fillna(0).astype(int)).values
    return X_tr, X_val, cc + ['pitcher_id', 'pitcher_x_count']


if __name__ == '__main__':
    import config
    log("loading train...")
    df = pd.read_csv(config.TRAIN_PATH)
    variants = {
        'base': v_base,
        'pid': v_pid,
        'pid+bid': v_pid_bid,
        'pid+pid_x_count': v_pid_x_count,
    }
    s = run_variants(df, variants, seeds=SEEDS,
                     out_json='~/LG_data/scratch/agent2_exp1_ids.json')
    print_summary(s)
