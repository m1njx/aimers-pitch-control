"""
167_catboost_bootstrap_growpolicy.py
아이디어 4: CatBoost 고급 옵션 — bootstrap_type(Bernoulli/MVS) x subsample,
grow_policy(Lossguide/Depthwise). CatBoost가 앙상블에서 75% 비중을 차지하므로
여기서의 개선이 가장 레버리지가 큼. 164번(boosting_type Plain/Ordered)과 별개 축.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

df_train = pd.read_csv(config.TRAIN_PATH)

BASE_CB = {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

variants = {
    'cb_bernoulli_sub0.8': {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
                             'cb': {**BASE_CB, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8},
                             'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}},
    'cb_mvs_sub0.8': {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
                       'cb': {**BASE_CB, 'bootstrap_type': 'MVS', 'subsample': 0.8},
                       'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}},
    'cb_lossguide_leaves31': {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
                               'cb': {**BASE_CB, 'grow_policy': 'Lossguide', 'max_leaves': 31},
                               'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}},
    'cb_depthwise': {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
                      'cb': {**BASE_CB, 'grow_policy': 'Depthwise'},
                      'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}},
}

results = {}
log(f"=== 167: CatBoost bootstrap_type/grow_policy screening (seeds={SCREEN_SEEDS}) ===")
for name, mp in variants.items():
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=mp,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[{name}] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    results[name] = {'screen_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/167_screen_result.json', 'w') as f:
    json.dump(results, f, indent=2)

promoted = [name for name, r in results.items() if r['screen_skill'] > BASELINE_REF - 2.0 and r['screen_skill'] > 841.0]
log(f"\nPromoted to full 5-seed confirm (screen roughly matches/beats baseline): {promoted if promoted else 'NONE'}")

full_results = {}
for name in promoted:
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=variants[name],
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[FULL 5-seed][{name}] skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    full_results[name] = {'full_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/167_full_result.json', 'w') as f:
    json.dump(full_results, f, indent=2)

log("\n=== 167 DONE ===")
