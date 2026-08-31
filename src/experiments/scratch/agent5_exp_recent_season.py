"""
agent5_exp_recent_season.py

INNER-ONLY (val=2022, 2023) screening of the season-level "recent_season"
decomposition (scratch/agent5_recent_season.py) on top of asof_dec (already
in every variant's base features via agent4_lib.run_variants).

Variants:
  control_asofdec       baseline (asof_dec only)
  recent_season_succ    + last-season-isolated success rate (raw/n/eb/slope)
  recent_season_full    + succ AND middle-rate last-season-isolated features
"""
import sys, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import pandas as pd
import config
import agent4_lib as L
from agent5_recent_season import add_recent_season

df_train = pd.read_csv(config.TRAIN_PATH)


def v_control(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    return X_tr, X_val, cc, None


def v_recent_succ(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = add_recent_season(df_tr_f, df_val_f, vs, A_tr, A_val, X_tr, X_val, use_mid=False)
    return X_tr2, X_val2, cc, None


def v_recent_full(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    X_tr2, X_val2 = add_recent_season(df_tr_f, df_val_f, vs, A_tr, A_val, X_tr, X_val, use_mid=True)
    return X_tr2, X_val2, cc, None


VARIANTS = {
    'control_asofdec':    v_control,
    'recent_season_succ': v_recent_succ,
    'recent_season_full': v_recent_full,
}

L.log("=== agent5 exp: 2-seed INNER-ONLY screen of recent_season (val=2022,2023) ===")
t0 = time.time()
summary = L.run_variants(df_train, VARIANTS, seeds=L.SCREEN_SEEDS,
                          out_json='~/LG_data/scratch/agent5_exp_recent_season_screen.json',
                          folds_subset=[0, 1])
L.log(f"done in {(time.time()-t0)/60:.1f}min")
L.print_summary(summary, ref=summary['control_asofdec']['mean_skill'])

with open('~/LG_data/scratch/agent5_exp_recent_season_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
L.log("=== recent_season screen DONE ===")
