"""
agent4_exp1_leaves_histcond.py

INNER-ONLY (val=2022, 2023) screening on top of asof_dec (already in every
variant's base features via agent4_lib.run_variants):

  A) LightGBM num_leaves sweep (agent3's EXP14/15 finding, re-tested here on
     top of asof_dec -- agent3 never combined the two).
  B) HistCondRates (recover_labels-based "hist_cond"), FIXED (no fold-boundary
     lookahead) vs ORIG_BUGGY (agent2's original, for comparison only) vs
     absent.
  C) form_ladder alone (already compliance-cleared in outputs/175).
  D) form_ladder + hist_cond_fixed combined.

2-seed screen first; only variants that clear the inner control by a
non-trivial margin get a 5-seed confirmation.
"""
import sys, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import config
import agent4_lib as L

df_train = pd.read_csv(config.TRAIN_PATH)


def v_control(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    return X_tr, X_val, cc, None


def make_leaves(n):
    def fn(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
        return X_tr, X_val, cc, {'num_leaves': n}
    return fn


def v_form(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = L.add_form_ladder(df_tr_f, df_val_f, A_tr, A_val, X_tr, X_val)
    return X_tr2, X_val2, cc, None


def v_histcond_fixed(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = L.add_hist_cond_FIXED(df_tr_f, df_val_f, vs, X_tr, X_val)
    return X_tr2, X_val2, cc, None


def v_histcond_buggy(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = L.add_hist_cond_ORIG_BUGGY(df_train, df_tr_f, df_val_f, vs, X_tr, X_val)
    return X_tr2, X_val2, cc, None


def v_form_hist(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = L.add_form_ladder(df_tr_f, df_val_f, A_tr, A_val, X_tr, X_val)
    X_tr3, X_val3 = L.add_hist_cond_FIXED(df_tr_f, df_val_f, vs, X_tr2, X_val2)
    return X_tr3, X_val3, cc, None


VARIANTS = {
    'control_asofdec':      v_control,
    'leaves10':              make_leaves(10),
    'leaves15':              make_leaves(15),
    'leaves20':              make_leaves(20),
    'leaves31':              make_leaves(31),
    'form_ladder':           v_form,
    'histcond_FIXED':        v_histcond_fixed,
    'histcond_ORIG_BUGGY':   v_histcond_buggy,
    'form_plus_histcond':    v_form_hist,
}

L.log("=== agent4 exp1: 2-seed INNER-ONLY screen (val=2022,2023) ===")
t0 = time.time()
summary = L.run_variants(df_train, VARIANTS, seeds=L.SCREEN_SEEDS,
                          out_json='~/LG_data/scratch/agent4_exp1_screen.json',
                          folds_subset=[0, 1])  # fold0=val2022, fold1=val2023
L.log(f"done in {(time.time()-t0)/60:.1f}min")
L.print_summary(summary, ref=summary['control_asofdec']['mean_skill'])

with open('~/LG_data/scratch/agent4_exp1_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
L.log("=== exp1 DONE ===")
