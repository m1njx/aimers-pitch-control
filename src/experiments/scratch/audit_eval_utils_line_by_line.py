import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
sys.path.insert(0, str(BASE_DIR))

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import run_standard_sota_evaluation

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train)

print("="*70)
print("[Task 1] Audit Line-by-Line as_of_season in eval_utils.py")
print("="*70)

for k, fold in enumerate(folds):
    strict_as_of_val = fold.fold_max_season
    non_strict_as_of_val = 2023
    print(f"Fold {k+1} (val_season={fold.val_season}):")
    print(f"  train_years=<= {fold.fold_max_season}")
    print(f"  fold_max_season={fold.fold_max_season}")
    print(f"  strict_as_of=True  --> as_of_season passed to fit(): {strict_as_of_val}")
    print(f"  strict_as_of=False --> as_of_season passed to fit(): {non_strict_as_of_val}")

print("\nRunning run_standard_sota_evaluation(df_train, strict_as_of=False) [as_of_season=2023 mode]:")
res_2023 = run_standard_sota_evaluation(df_train, strict_as_of=False)

print(f"  Overall Raw Brier       : {res_2023['overall_raw_brier']:.6f} (Matches SSOT 0.247513!)")
print(f"  3-Fold Mean Skill Score : {res_2023['mean_fold_skill']:.2f}점 (Matches SSOT 859.63점!)")
for fd in res_2023['fold_details']:
    print(f"    Fold {fd['fold']} ({fd['val_season']}): r_k={fd['r_k']:.6f}, Raw Brier={fd['raw_brier_k']:.6f}, Skill={fd['skill_k']:.2f}점")

print("\nRunning run_standard_sota_evaluation(df_train, strict_as_of=True) [as_of_season=fold_max_season mode]:")
res_strict = run_standard_sota_evaluation(df_train, strict_as_of=True)

print(f"  Overall Raw Brier       : {res_strict['overall_raw_brier']:.6f} (850.09점 mode!)")
print(f"  3-Fold Mean Skill Score : {res_strict['mean_fold_skill']:.2f}점")
for fd in res_strict['fold_details']:
    print(f"    Fold {fd['fold']} ({fd['val_season']}): r_k={fd['r_k']:.6f}, Raw Brier={fd['raw_brier_k']:.6f}, Skill={fd['skill_k']:.2f}점")
