"""
agent5_exp_outer_confirm_recent_season.py

Applies the SINGLE inner-selected candidate ("recent_season", chosen from
agent5_exp_recent_season.py's inner-only screen) to outer(2024) via the
OFFICIAL harness (core/eval_utils.py), exactly once. Mirrors
agent4_exp3_outer_confirm.py's pattern. DO NOT RUN until the inner screen
(agent5_exp_recent_season.py) has finished and a candidate variant
(recent_season_succ vs recent_season_full) has been locked in from
inner-only evidence.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation
from agent2_asof_decomp2 import AsofDecomposer2
from agent5_recent_season import add_recent_season


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- CANDIDATE DEFINITION (fill in from inner-only screen result) ----
CANDIDATE_NAME = "recent_season_succ"
USE_MID = False  # inner screen: succ inner_mean=1397.90(+9.70, both folds positive) vs full=1386.90(-1.31, mixed) -> succ selected


def add_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    A_tr = dec.transform(df_tr_f); A_tr.index = X_tr_f.index
    A_val = dec.transform(df_val_f); A_val.index = X_val_f.index
    X_tr_f = pd.concat([X_tr_f, A_tr], axis=1)
    X_val_f = pd.concat([X_val_f, A_val], axis=1)
    X_tr_f, X_val_f = add_recent_season(df_tr_f, df_val_f, val_season, A_tr, A_val, X_tr_f, X_val_f, use_mid=USE_MID)
    return X_tr_f, X_val_f


df_train = pd.read_csv(config.TRAIN_PATH)
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
ASOF_DEC_REF_OUTER = 805.74  # CORRECT label per 172 audit (pure asof_dec, not w2)
ASOF_DEC_REF_3FOLD = None  # fill in if available from a pure-asof_dec 5-seed 3-fold run

if __name__ == '__main__':
    if USE_MID is None:
        raise SystemExit("Fill in CANDIDATE_NAME / USE_MID from the inner-only screen before running.")
    log(f"=== agent5 outer confirm: candidate '{CANDIDATE_NAME}' (USE_MID={USE_MID}) ===")
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                      extra_feature_fn=add_features)
    log(f"[{CANDIDATE_NAME}] 5-seed 3fold={r['mean_fold_skill']:.2f} "
        f"outer(2024)={r['fold_details'][2]['skill_k']:.2f} "
        f"(vs asof_dec-only ref {ASOF_DEC_REF_OUTER} = {r['fold_details'][2]['skill_k']-ASOF_DEC_REF_OUTER:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    with open(f'/tmp/agent5_outer_confirm_{CANDIDATE_NAME}.json', 'w') as f:
        json.dump({'candidate': CANDIDATE_NAME, 'mean_fold_skill': r['mean_fold_skill'],
                   'fold_details': r['fold_details']}, f, indent=2)
    log("=== DONE ===")
