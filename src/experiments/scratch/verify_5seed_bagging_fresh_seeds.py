"""
verify_5seed_bagging_fresh_seeds.py
137번의 5-seed 배깅(851.72점, seeds=42,100,2024,777,999)이 그 특정 5개 시드 조합의
우연이 아닌지 확인하기 위해, 겹치지 않는 새로운 5개 시드로 SSOT baseline(15/75/10)을
다시 배깅 검증한다. 3-seed(854.81)와 함께 비교해 어느 쪽이 더 안정적인 기댓값에
가까운지(실제 LB 839.60과의 거리로) 판단하는 근거로 쓴다.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation

df_train = pd.read_csv(config.TRAIN_PATH)

sota_mp = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
sota_weights = (0.15, 0.75, 0.10)
sota_shifts = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}

FRESH_SEEDS_5 = [7, 123, 2025, 31415, 8675309]
LB_ACTUAL = 839.6025545093
PREV_3SEED = 854.81
PREV_5SEED_137 = 851.72

t0 = time.time()
print(f"[Run] Fresh 5-seed bagged evaluation, seeds={FRESH_SEEDS_5} ...")
r5b = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                    weights=sota_weights, shifts=sota_shifts,
                                    random_seeds=FRESH_SEEDS_5)
print(f"  Fresh 5-seed: Skill={r5b['mean_fold_skill']:.2f}점 Brier={r5b['overall_raw_brier']:.6f} "
      f"elapsed={(time.time()-t0)/60:.1f}min")

# Also do a 7-seed bagged run combining both seed sets (42,100,2024,777,999,+2 more)
# to see if the estimate keeps stabilizing as seed count grows.
SEEDS_7 = [42, 100, 2024, 777, 999, 7, 123]
t1 = time.time()
print(f"\n[Run] 7-seed bagged evaluation, seeds={SEEDS_7} ...")
r7b = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                    weights=sota_weights, shifts=sota_shifts,
                                    random_seeds=SEEDS_7)
print(f"  7-seed: Skill={r7b['mean_fold_skill']:.2f}점 Brier={r7b['overall_raw_brier']:.6f} "
      f"elapsed={(time.time()-t1)/60:.1f}min")

results = {
    '3seed_854.81_(42,100,2024)': PREV_3SEED,
    '5seed_137_(42,100,2024,777,999)': PREV_5SEED_137,
    f'5seed_fresh_{FRESH_SEEDS_5}': r5b['mean_fold_skill'],
    f'7seed_{SEEDS_7}': r7b['mean_fold_skill'],
}

print("\n=== SUMMARY: distance from each local estimate to actual LB (839.6025545093) ===")
for label, skill in results.items():
    print(f"  {label}: local={skill:.2f}점, gap_to_actual_LB={LB_ACTUAL - skill:+.2f}점")

with open('/tmp/verify_5seed_fresh_result.txt', 'w') as f:
    f.write(f"Fresh 5-seed ({FRESH_SEEDS_5}): Skill={r5b['mean_fold_skill']:.2f} Brier={r5b['overall_raw_brier']:.6f}\n")
    for fd in r5b['fold_details']:
        f.write(f"  Fold {fd['fold']} ({fd['val_season']}): Brier={fd['raw_brier_k']:.6f} Skill={fd['skill_k']:.2f}\n")
    f.write(f"\n7-seed ({SEEDS_7}): Skill={r7b['mean_fold_skill']:.2f} Brier={r7b['overall_raw_brier']:.6f}\n")
    for fd in r7b['fold_details']:
        f.write(f"  Fold {fd['fold']} ({fd['val_season']}): Brier={fd['raw_brier_k']:.6f} Skill={fd['skill_k']:.2f}\n")
    f.write(f"\n--- Comparison table ---\n")
    for label, skill in results.items():
        f.write(f"{label}: {skill:.2f} (gap to actual LB {LB_ACTUAL:.2f} = {LB_ACTUAL - skill:+.2f})\n")

print("\nDone. Saved to /tmp/verify_5seed_fresh_result.txt")
