"""
164_catboost_ordered_xgb_dart.py
아이디어 6: CatBoost Plain vs Ordered boosting_type, XGBoost DART booster.
2-seed 스크리닝(DL_SEEDS=[7,123])으로 먼저 방향을 본 뒤, 노이즈 바닥(±15.10)을
넘는 후보만 5-seed 정식 확인.
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

BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

variants = {
    'baseline(Plain-default,GBDT)': dict(BASE_MP),
    'cb_ordered': {**BASE_MP, 'cb': {**BASE_MP['cb'], 'boosting_type': 'Ordered'}},
    'cb_plain_explicit': {**BASE_MP, 'cb': {**BASE_MP['cb'], 'boosting_type': 'Plain'}},
    'xgb_dart': {**BASE_MP, 'xgb': {**BASE_MP['xgb'], 'booster': 'dart', 'rate_drop': 0.1}},
    'cb_ordered+xgb_dart': {**BASE_MP,
                             'cb': {**BASE_MP['cb'], 'boosting_type': 'Ordered'},
                             'xgb': {**BASE_MP['xgb'], 'booster': 'dart', 'rate_drop': 0.1}},
}

results = {}
log(f"=== 164: CatBoost Plain/Ordered + XGBoost DART screening (seeds={SCREEN_SEEDS}) ===")
for name, mp in variants.items():
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=mp,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[{name}] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs baseline_ref={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    results[name] = {'screen_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/164_screen_result.json', 'w') as f:
    json.dump(results, f, indent=2)

# Promote any variant beating baseline screen by > noise floor (use baseline's own screen score as ref, not global 843.69,
# since 2-seed screen has its own noise level; use a conservative +10pt bar over the baseline(Plain-default) screen result)
baseline_screen = results['baseline(Plain-default,GBDT)']['screen_skill']
promoted = [name for name, r in results.items()
            if name != 'baseline(Plain-default,GBDT)' and r['screen_skill'] > baseline_screen + 10.0]

log(f"\nBaseline 2-seed screen: {baseline_screen:.2f}")
log(f"Promoted to full 5-seed confirm (screen > baseline+10): {promoted if promoted else 'NONE'}")

full_results = {}
for name in promoted:
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=variants[name],
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[FULL 5-seed][{name}] skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    full_results[name] = {'full_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/164_full_result.json', 'w') as f:
    json.dump(full_results, f, indent=2)

log("\n=== 164 DONE ===")
