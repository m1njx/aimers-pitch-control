"""
verify_5seed_bagging.py
Noise floor 재측정 결과 현재 SSOT의 시드 변동폭이 ±15.10점(2-sigma)으로 매우 커서,
단일 시드(42) 853.62점은 다소 운 좋은 값일 가능성이 높다. run_standard_sota_evaluation은
random_seeds 리스트로 예측 자체를 평균내는(bagging) 기능을 이미 지원하므로, 5-seed 배깅이
분산을 실제로 줄이고 평균 성능도 유지/개선하는지 직접 검증한다.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
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
SEEDS = [42, 100, 2024, 777, 999]

t0 = time.time()
print(f"[Run] 5-seed BAGGED evaluation (predictions averaged across seeds {SEEDS})...")
r_bagged = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                         weights=sota_weights, shifts=sota_shifts,
                                         random_seeds=SEEDS)
print(f"  Bagged 5-seed: Skill={r_bagged['mean_fold_skill']:.2f}점, Brier={r_bagged['overall_raw_brier']:.6f}")
print(f"  Elapsed: {(time.time()-t0)/60:.1f}min")

single_seed_mean = (853.62 + 848.06 + 838.48 + 839.96 + 832.11) / 5
single_seed_std = 7.55

with open('/tmp/bagging_5seed_result.txt', 'w') as f:
    f.write(f"5-seed BAGGED (predictions averaged) skill: {r_bagged['mean_fold_skill']:.2f}\n")
    f.write(f"Raw Brier: {r_bagged['overall_raw_brier']:.6f}\n")
    f.write(f"vs single-seed-42 official SSOT: 853.62\n")
    f.write(f"vs single-seed mean (5 separate runs): {single_seed_mean:.2f} (std={single_seed_std:.2f})\n")
    for fd in r_bagged['fold_details']:
        f.write(f"  Fold {fd['fold']} ({fd['val_season']}): Brier={fd['raw_brier_k']:.6f} Skill={fd['skill_k']:.2f}\n")

print("\nDone. Saved to /tmp/bagging_5seed_result.txt")
