import os
import joblib
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
model_dir_50 = os.path.join(BASE_DIR, 'work', 'submit_v50', 'model')
model_dir_51 = os.path.join(BASE_DIR, 'work', 'submit_v51', 'model')

# Let's inspect the models in v51
print("Inspecting v51 trained models...")
# Check LightGBM MSE models
for s in [7, 123, 2025]:
    m50 = lgb.Booster(model_file=os.path.join(model_dir_50, f'lgbm_mse_model_seed{s}.txt'))
    m51 = lgb.Booster(model_file=os.path.join(model_dir_51, f'lgbm_mse_model_seed{s}.txt'))
    print(f"Seed {s}: m50 num_trees={m50.num_trees()}, m51 num_trees={m51.num_trees()}")
