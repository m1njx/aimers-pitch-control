"""
171_cb_ordered_retuned.py
164번에서 cb_ordered(CatBoost Ordered boosting)가 -155.38점으로 크게 REJECT됐음.
가설: Ordered 모드는 Plain과 같은 iterations=250/learning_rate=0.05로는 학습이
덜 수렴해서(Ordered는 각 트리마다 순열 기반 통계를 다시 계산하느라 정보 이용이
비효율적) 언더피팅됐을 가능성. iterations를 늘리고 learning_rate를 높여 재시도.
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
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

LGB_MP = {'colsample_bytree': 0.7, 'subsample': 0.7}
XGB_MP = {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}

variants = {
    'cb_ordered_iter500_lr0.05': {'lgb': LGB_MP, 'xgb': XGB_MP,
        'cb': {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0, 'boosting_type': 'Ordered'}},
    'cb_ordered_iter250_lr0.1': {'lgb': LGB_MP, 'xgb': XGB_MP,
        'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 10.0, 'boosting_type': 'Ordered'}},
    'cb_ordered_iter500_lr0.08_l2_5': {'lgb': LGB_MP, 'xgb': XGB_MP,
        'cb': {'iterations': 500, 'depth': 6, 'learning_rate': 0.08, 'l2_leaf_reg': 5.0, 'boosting_type': 'Ordered'}},
}

results = {}
log(f"=== 171: cb_ordered 재튜닝 (더 많은 iterations/더 높은 lr, 2-seed) ===")
for name, mp in variants.items():
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=mp,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[{name}] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    results[name] = {'screen_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/171_screen_result.json', 'w') as f:
    json.dump(results, f, indent=2)

promoted = [name for name, r in results.items() if r['screen_skill'] > 831.0]
log(f"\nPromoted to full 5-seed confirm: {promoted if promoted else 'NONE'}")

full_results = {}
for name in promoted:
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=variants[name],
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[FULL 5-seed][{name}] skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    full_results[name] = {'full_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/171_full_result.json', 'w') as f:
    json.dump(full_results, f, indent=2)

log("\n=== 171 DONE ===")
