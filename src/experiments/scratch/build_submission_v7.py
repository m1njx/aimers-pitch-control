import sys
import os
import shutil
import zipfile
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/LG_data'))
import config
from preprocessing import PitchPreprocessor

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUBMIT_DIR = BASE_DIR / 'work/submit_v7'
MODEL_DIR = SUBMIT_DIR / 'model'
DUMMY_DIR = BASE_DIR / 'work/dummy_eval_v7'

# 145번/150번에서 확정한 42-제외 5-seed (시드42 행운 편향 제거)
SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

if SUBMIT_DIR.exists():
    shutil.rmtree(SUBMIT_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if DUMMY_DIR.exists():
    shutil.rmtree(DUMMY_DIR)
DUMMY_DIR.mkdir(parents=True, exist_ok=True)

NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("=" * 70)
print("[Task 1] Full Re-training on Entire Train Dataset (42-excluded 5-seed bagging)")
print("=" * 70)

t0_train = time.time()
df_train = pd.read_csv(config.TRAIN_PATH)
y_train = df_train[config.TARGET_COL].values
print(f"Loaded train dataset: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns")

t_prep0 = time.time()
prep = PitchPreprocessor()
prep.fit(df_train, as_of_season=None, is_final=True)
X_train = prep.transform(df_train)

base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
X_train['count_x_base'] = (cc_str + '_' + base_str)

cat_map = {v: i for i, v in enumerate(X_train['count_x_base'].unique())}
X_train['count_x_base'] = X_train['count_x_base'].map(cat_map).fillna(-1).astype(int)
prep.count_x_base_map = cat_map

cat_cols = [c for c in X_train.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
t_prep = time.time() - t_prep0
print(f"Preprocessor fit & transform complete in {t_prep:.1f}s. Feature matrix shape: {X_train.shape}")
print(f"Categorical features ({len(cat_cols)}개): {cat_cols}")

X_tr_cb = X_train.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
    X_tr_cb[c] = X_tr_cb[c].astype(np.float32)

X_tr_xgb = X_train.copy()
for c in cat_cols:
    X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
X_tr_xgb = X_tr_xgb.astype(np.float32)

for seed in SEEDS:
    print(f"\n--- [seed={seed}] Training LGBM / CatBoost / XGBoost ---")

    t1 = time.time()
    m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                random_state=seed, verbosity=-1, n_jobs=-1)
    m_lgb.fit(X_train, y_train, categorical_feature=cat_idx)
    m_lgb.booster_.save_model(str(MODEL_DIR / f'lgbm_model_seed{seed}.txt'))
    print(f"  LightGBM seed={seed} done in {time.time()-t1:.1f}s")

    t2 = time.time()
    m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                               random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
    m_cb.fit(X_tr_cb, y_train)
    m_cb.save_model(str(MODEL_DIR / f'catboost_model_seed{seed}.cbm'))
    print(f"  CatBoost seed={seed} done in {time.time()-t2:.1f}s")

    t3 = time.time()
    m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                               colsample_bytree=0.8, subsample=0.8, random_state=seed,
                               n_jobs=-1, eval_metric='logloss')
    m_xgb.fit(X_tr_xgb, y_train)
    m_xgb.save_model(str(MODEL_DIR / f'xgb_model_seed{seed}.json'))
    print(f"  XGBoost seed={seed} done in {time.time()-t3:.1f}s")

joblib.dump(prep, MODEL_DIR / 'preprocessor_artifacts.pkl')
joblib.dump(prep.trackman_builder, MODEL_DIR / 'trackman_artifacts.pkl')
print("\nPreprocessor & Trackman artifacts saved to model/ directory.")

t_train_duration = time.time() - t0_train
print(f"\nFull Re-training (42-excluded 5-seed bagging, 15 models) completed in {t_train_duration:.1f} seconds!")

# =========================================================================
# WORK 2: Write script.py for Inference (5-seed bagged, __file__-based paths)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 2] Writing inference script.py")
print("=" * 70)

script_content = r"""import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

t0 = time.time()
print("Starting DACON 7th Submission Inference Pipeline (seed42-excluded 5-seed bagged ensemble)...")

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

data_dir = os.path.join(SCRIPT_DIR, "data")
if not os.path.exists(data_dir):
    data_dir = "data"
output_dir = os.path.join(SCRIPT_DIR, "output")
if not os.path.exists(output_dir):
    output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(SCRIPT_DIR, "model")

test_path = os.path.join(data_dir, "test.csv")
if not os.path.exists(test_path):
    test_path = "data/test.csv"

print(f"Loading test data from: {test_path}")
df_test = pd.read_csv(test_path)
print(f"Test data shape: {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep.trackman_builder = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))

X_test = prep.transform(df_test)

base_str = ((df_test['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_test['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)

cat_map = getattr(prep, 'count_x_base_map', {})
X_test['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

cat_cols = [c for c in X_test.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]
cat_idx = [X_test.columns.get_loc(c) for c in cat_cols if c in X_test.columns]

X_test_cb = X_test.copy()
for c in cat_cols:
    X_test_cb[c] = X_test_cb[c].astype(int).astype(str)
for c in [col for col in X_test_cb.columns if col not in cat_cols]:
    X_test_cb[c] = X_test_cb[c].astype(np.float32)

X_test_xgb = X_test.copy()
for c in cat_cols:
    X_test_xgb[c] = X_test_xgb[c].astype('category').cat.codes.astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

print("Loading trained models and predicting (5-seed bagged per model type, seed42 excluded)...")

p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))

for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test)

    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]

    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]

n_seeds = len(SEEDS)
p_lgb = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)

# Weighted Ensemble (LGBM 15% + CatBoost 75% + XGBoost 10%), each model 5-seed prediction-bagged (seed42 excluded)
p_final = np.clip(W_LGB * p_lgb + W_CB * p_cb + W_XGB * p_xgb, 1e-6, 1 - 1e-6)

sub = pd.DataFrame({
    'row_id': df_test['row_id'],
    'control_success': p_final
})

sub_path = os.path.join(output_dir, 'submission.csv')
sub.to_csv(sub_path, index=False)

elapsed = time.time() - t0
print(f"Inference completed & submission saved to {sub_path} in {elapsed:.2f} seconds!")
"""

with open(SUBMIT_DIR / 'script.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

print("script.py written successfully.")

# =========================================================================
# WORK 3: Write requirements.txt (>= lower-bound style, matching successful v4 pattern)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 3] Writing requirements.txt")
print("=" * 70)

req_content = """lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=1.7.0
"""
with open(SUBMIT_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)
print("requirements.txt written:")
print(req_content)

# =========================================================================
# WORK 4: Copy required local pipeline modules (lesson from 5th submission failure)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 4] Copying local pipeline modules")
print("=" * 70)

modules_to_copy = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
for m_file in modules_to_copy:
    src_file = BASE_DIR / m_file
    shutil.copy(src_file, SUBMIT_DIR / m_file)
    print(f"  Copied: {m_file}")

# =========================================================================
# WORK 5: Package submit_v7.zip
# =========================================================================
print("\n" + "=" * 70)
print("[Task 5] Packaging submit_v7.zip")
print("=" * 70)

zip_path = BASE_DIR / 'work/submit_v7.zip'
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = sorted(zipf.namelist())
print(f"Created zip archive: {zip_path} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
print(f"Zip contents ({len(namelist)} files): {namelist}")

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
missing = [m for m in required_modules if m not in namelist]
print(f"Missing required local modules: {missing} (should be empty list)")

with open('/tmp/submit_v7_build_result.json', 'w') as f:
    json.dump({
        "zip_path": str(zip_path),
        "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
        "train_duration_seconds": t_train_duration,
        "files": namelist,
        "missing_required_modules": missing,
        "seeds": SEEDS,
    }, f, indent=2)

print(f"\nBUILD COMPLETE. Total time: {(time.time()-t0_train)/60:.1f} min")
