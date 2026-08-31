"""
remeasure_noise_floor_current_ssot.py
90번 보고서의 Noise Floor(±1.70점)는 구 SSOT 구성(LGBM 20%+CatBoost70%+XGB10%)에서 측정된 것.
124번 보고서에서 가중치가 15/75/10으로 바뀐 이후 재측정된 적이 없음.
방금 explore_mono_full_ensemble.py에서 baseline seed 42/100/2024가 853.62/848.06/838.48로
15점 넘게 벌어지는 것을 발견 -> 현재 SSOT 기준으로 5-seed(42,100,2024,777,999) 노이즈 바닥을
report 90과 동일한 방법론으로 재측정한다.
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

# Already have 42=853.62, 100=848.06, 2024=838.48 from explore_mono_full_ensemble.py baseline.
# Complete the 5-seed set with 777, 999 (same seeds as Report 90) for direct comparability.
NEW_SEEDS = [777, 999]
KNOWN = {42: 853.62, 100: 848.06, 2024: 838.48}

t0 = time.time()
results = dict(KNOWN)
for seed in NEW_SEEDS:
    print(f"\n[Run] seed={seed} ...")
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                      weights=sota_weights, shifts=sota_shifts,
                                      random_seeds=[seed])
    results[seed] = r['mean_fold_skill']
    print(f"  seed={seed}: Skill={r['mean_fold_skill']:.2f}점, Brier={r['overall_raw_brier']:.6f}, elapsed={time.time()-t0:.1f}s")

all_seeds = [42, 100, 2024, 777, 999]
skills = [results[s] for s in all_seeds]
mean_skill = float(np.mean(skills))
std_skill = float(np.std(skills))
rng = max(skills) - min(skills)

print("\n=== CURRENT SSOT (15/75/10) 5-SEED NOISE FLOOR ===")
for s in all_seeds:
    print(f"  seed={s}: {results[s]:.2f}점")
print(f"Mean: {mean_skill:.2f}점, Std: {std_skill:.2f}점, 2-sigma: {2*std_skill:.2f}점, Range: {rng:.2f}점")

with open('/tmp/noise_floor_current_ssot_result.txt', 'w') as f:
    f.write("Current SSOT (15/75/10) 5-seed noise floor remeasurement\n")
    for s in all_seeds:
        f.write(f"seed={s}: {results[s]:.2f}\n")
    f.write(f"Mean={mean_skill:.2f} Std={std_skill:.2f} 2sigma={2*std_skill:.2f} Range={rng:.2f}\n")
    f.write(f"Old Report-90 noise floor (stale, 20/70/10 config): 2sigma=1.70 range=2.35\n")

print("\nDone. Saved to /tmp/noise_floor_current_ssot_result.txt")
