import os, sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_DIR = os.path.expanduser('~/LG_data')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v36', 'model')
m = lgb.Booster(model_file=os.path.join(model_dir, 'lgbm_mse_model_seed7.txt'))
print("Feature names in LGB MSE model:", m.feature_name()[:10])
print("Num features in LGB MSE model:", len(m.feature_name()))
