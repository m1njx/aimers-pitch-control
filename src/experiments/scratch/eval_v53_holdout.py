import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
work_v53_dir = os.path.join(BASE_DIR, 'work', 'submit_v53')
model_dir = os.path.join(work_v53_dir, 'model')
data_dir = os.path.join(BASE_DIR, 'open', 'data')

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
is_val24 = (df['season'] == 2024)
is_val23 = (df['season'] == 2023)

y_24 = df.loc[is_val24, 'control_success'].values.astype(np.float32)
y_23 = df.loc[is_val23, 'control_success'].values.astype(np.float32)

print(f"Validation sets: 2024={len(y_24):,} rows, 2023={len(y_23):,} rows")
