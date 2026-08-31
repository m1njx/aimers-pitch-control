"""
verify_3seed_bagging.py
5-seed 배깅(851.72점)은 서버 10분 제한 초과 위험이 있어, 시간 예산 안에 안전하게 들어오는
3-seed 배깅의 실제 CV 성능을 측정한다.
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

t0 = time.time()
r3 = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                   weights=sota_weights, shifts=sota_shifts,
                                   random_seeds=[42, 100, 2024])
print(f"3-seed bagged: Skill={r3['mean_fold_skill']:.2f}점 Brier={r3['overall_raw_brier']:.6f} elapsed={(time.time()-t0)/60:.1f}min")

with open('/tmp/bagging_3seed_result.txt', 'w') as f:
    f.write(f"3-seed bagged skill: {r3['mean_fold_skill']:.2f}\n")
    f.write(f"Raw Brier: {r3['overall_raw_brier']:.6f}\n")
    for fd in r3['fold_details']:
        f.write(f"  Fold {fd['fold']} ({fd['val_season']}): Brier={fd['raw_brier_k']:.6f} Skill={fd['skill_k']:.2f}\n")
print("Done.")
