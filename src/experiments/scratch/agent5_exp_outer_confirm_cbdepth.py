"""
agent5_exp_outer_confirm_cbdepth.py

Applies the SINGLE inner-selected CatBoost depth/l2_leaf_reg override (chosen
from agent5_exp_cb_depth.py's inner-only screen) to outer(2024) via the
OFFICIAL harness (core/eval_utils.py), exactly once. DO NOT RUN until the
inner screen has finished and a config is locked in from inner-only evidence.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation
from agent2_asof_decomp2 import AsofDecomposer2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- CANDIDATE DEFINITION (fill in from inner-only screen result) ----
CANDIDATE_NAME = "cb_l2reg30"
CB_OVERRIDE = {'l2_leaf_reg': 30.0}  # inner screen: l2reg30 inner_mean=1395.11(+6.91, both folds positive: +5.81/+8.02) -- best of 7 configs


def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f); val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index; val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


df_train = pd.read_csv(config.TRAIN_PATH)
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
ASOF_DEC_REF_OUTER = 805.74  # CORRECT label per 172 audit (pure asof_dec, not w2)

if __name__ == '__main__':
    if CB_OVERRIDE is None:
        raise SystemExit("Fill in CANDIDATE_NAME / CB_OVERRIDE from the inner-only screen before running.")
    BASE_MP = {
        'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
        'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0, **CB_OVERRIDE},
        'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
    }
    log(f"=== agent5 outer confirm: candidate '{CANDIDATE_NAME}' (CB_OVERRIDE={CB_OVERRIDE}) ===")
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                      extra_feature_fn=add_asof_dec_features)
    log(f"[{CANDIDATE_NAME}] 5-seed 3fold={r['mean_fold_skill']:.2f} "
        f"outer(2024)={r['fold_details'][2]['skill_k']:.2f} "
        f"(vs asof_dec-only ref {ASOF_DEC_REF_OUTER} = {r['fold_details'][2]['skill_k']-ASOF_DEC_REF_OUTER:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    with open(f'/tmp/agent5_outer_confirm_{CANDIDATE_NAME}.json', 'w') as f:
        json.dump({'candidate': CANDIDATE_NAME, 'mean_fold_skill': r['mean_fold_skill'],
                   'fold_details': r['fold_details']}, f, indent=2)
    log("=== DONE ===")
