"""
agent4_exp3_outer_confirm.py

Applies the SINGLE inner-selected best candidate (chosen from exp1/exp2 screens,
inner-only) to outer(2024) via the OFFICIAL harness (core/eval_utils.py), exactly
once, mirroring report 191's independent-reproduction pattern. This script is a
TEMPLATE -- the actual model_params / extra_feature_fn are filled in after the
inner screens (exp1, exp2) finish and a candidate is chosen. Do not run until
the candidate is locked in from inner-only evidence.
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
from agent2_exp7_extra import form_ladder, HistCondRates
from agent2_recover_labels import recover


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- CANDIDATE DEFINITION (filled in from exp1 inner-only screen, 2026-08-13) ----
# exp1 result: form_ladder inner(22/23)=+13.00 (fold2022 +18.96, fold2023 +7.03, both positive).
# num_leaves sweep: all noise-level. histcond_FIXED: -1.15 (REJECT, ORIG_BUGGY's +11.43 was a
# leakage artifact). form_plus_histcond ~= form_ladder alone, so histcond adds nothing.
CANDIDATE_NAME = "form_ladder"
LGB_OVERRIDE = {}
USE_FORM = True
USE_HISTCOND = False


def add_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    A_tr = dec.transform(df_tr_f); A_tr.index = X_tr_f.index
    A_val = dec.transform(df_val_f); A_val.index = X_val_f.index
    X_tr_f = pd.concat([X_tr_f, A_tr], axis=1)
    X_val_f = pd.concat([X_val_f, A_val], axis=1)
    if USE_FORM:
        F_tr = form_ladder(df_tr_f, A_tr); F_tr.index = X_tr_f.index
        F_val = form_ladder(df_val_f, A_val); F_val.index = X_val_f.index
        X_tr_f = pd.concat([X_tr_f, F_tr], axis=1)
        X_val_f = pd.concat([X_val_f, F_val], axis=1)
    if USE_HISTCOND:
        L_tr = recover(df_tr_f)  # scoped to this fold's train slice only (fixed)
        hc = HistCondRates().fit(df_tr_f, L_tr, val_season)
        H_tr = hc.transform(df_tr_f); H_tr.index = X_tr_f.index
        H_val = hc.transform(df_val_f); H_val.index = X_val_f.index
        X_tr_f = pd.concat([X_tr_f, H_tr], axis=1)
        X_val_f = pd.concat([X_val_f, H_val], axis=1)
    return X_tr_f, X_val_f


df_train = pd.read_csv(config.TRAIN_PATH)
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7, **LGB_OVERRIDE},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
ASOF_DEC_REF_OUTER = 812.21
ASOF_DEC_REF_3FOLD = 1204.66

log(f"=== agent4 exp3: outer(2024) SINGLE confirmation for candidate '{CANDIDATE_NAME}' ===")
t0 = time.time()
r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                  weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                  extra_feature_fn=add_features)
log(f"[{CANDIDATE_NAME}] 5-seed 3fold={r['mean_fold_skill']:.2f} "
    f"(vs asof_dec {ASOF_DEC_REF_3FOLD} = {r['mean_fold_skill']-ASOF_DEC_REF_3FOLD:+.2f}) "
    f"outer(2024)={r['fold_details'][2]['skill_k']:.2f} "
    f"(vs asof_dec {ASOF_DEC_REF_OUTER} = {r['fold_details'][2]['skill_k']-ASOF_DEC_REF_OUTER:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

with open(f'/tmp/agent4_exp3_{CANDIDATE_NAME}.json', 'w') as f:
    json.dump({'candidate': CANDIDATE_NAME, 'mean_fold_skill': r['mean_fold_skill'],
               'fold_details': r['fold_details']}, f, indent=2)
log("=== exp3 DONE ===")
