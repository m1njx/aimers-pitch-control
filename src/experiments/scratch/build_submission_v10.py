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
sys.path.insert(0, '~/LG_data/scratch')
import config
from preprocessing import PitchPreprocessor

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
SUBMIT_DIR = BASE_DIR / 'work/submit_v10'
MODEL_DIR = SUBMIT_DIR / 'model'

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.025, 0.0, -0.05  # 168번에서 nested-선택된 L2 목적함수 전용 shift
DECAY = 0.7  # 174번: recency 가중치, 5-seed nested-honest 869.90점 (GBDT-only 신규 SSOT)

if SUBMIT_DIR.exists():
    shutil.rmtree(SUBMIT_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("[Task 0] Preprocessing (GBDT L2-objective + recency-weighted, TabM 없음 — MPS/venv311 리스크 완전 배제)")
print("=" * 70)

t0_train = time.time()
df_train = pd.read_csv(config.TRAIN_PATH)
y_train = df_train[config.TARGET_COL].values.astype(np.float32)
print(f"Loaded train dataset: {df_train.shape[0]:,} rows")
print(f"numpy={np.__version__} pandas={pd.__version__} python={sys.version.split()[0]}")

prep = PitchPreprocessor()
prep.fit(df_train, as_of_season=None, is_final=True)
X_train = prep.transform(df_train)

base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
X_train['count_x_base'] = (cc_str + '_' + base_str)
cat_map_countbase = {v: i for i, v in enumerate(X_train['count_x_base'].unique())}
X_train['count_x_base'] = X_train['count_x_base'].map(cat_map_countbase).fillna(-1).astype(int)
prep.count_x_base_map = cat_map_countbase

CAT_COLS = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL, 'count_x_base']
cat_cols = [c for c in CAT_COLS if c in X_train.columns]
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

# --- recency sample weight: 전체 데이터 학습이라 "현재 시점" = 데이터의 최신 시즌(2024) ---
as_of_full = int(df_train['season'].max())
season_gap = (as_of_full - df_train['season']).clip(lower=0).values
sample_weight = np.power(DECAY, season_gap).astype(np.float64)
sample_weight = sample_weight / sample_weight.mean()
print(f"Recency sample_weight: decay={DECAY}, as_of={as_of_full}, "
      f"season별 평균가중치={pd.Series(sample_weight, index=df_train['season']).groupby(level=0).mean().to_dict()}")

print("\n" + "=" * 70)
print("[Task 1] Full Re-training: GBDT 3종 (L2/RMSE objective + recency weight, 5-seed, full data)")
print("=" * 70)

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
    print(f"\n--- [GBDT seed={seed}] Training LGBM(regression) / CatBoost(RMSE) / XGBoost(reg:squarederror) ---")
    t1 = time.time()
    m_lgb = lgb.LGBMRegressor(objective='regression', n_estimators=250, num_leaves=45, learning_rate=0.05,
                               min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                               random_state=seed, verbosity=-1, n_jobs=-1)
    m_lgb.fit(X_train, y_train, categorical_feature=cat_idx, sample_weight=sample_weight)
    m_lgb.booster_.save_model(str(MODEL_DIR / f'lgbm_model_seed{seed}.txt'))
    print(f"  LightGBM seed={seed} done in {time.time()-t1:.1f}s")

    t2 = time.time()
    m_cb = CatBoostRegressor(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                              loss_function='RMSE', random_seed=seed, verbose=0,
                              cat_features=cat_cols, thread_count=-1)
    m_cb.fit(X_tr_cb, y_train, sample_weight=sample_weight)
    m_cb.save_model(str(MODEL_DIR / f'catboost_model_seed{seed}.cbm'))
    print(f"  CatBoost seed={seed} done in {time.time()-t2:.1f}s")

    t3 = time.time()
    m_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, max_depth=5, learning_rate=0.05,
                              colsample_bytree=0.8, subsample=0.8, random_state=seed, n_jobs=-1)
    m_xgb.fit(X_tr_xgb, y_train, sample_weight=sample_weight)
    m_xgb.save_model(str(MODEL_DIR / f'xgb_model_seed{seed}.json'))
    print(f"  XGBoost seed={seed} done in {time.time()-t3:.1f}s")

joblib.dump(prep, MODEL_DIR / 'preprocessor_artifacts.pkl')
joblib.dump(prep.trackman_builder, MODEL_DIR / 'trackman_artifacts.pkl')
print("\nGBDT training complete. Preprocessor & Trackman artifacts saved.")

t_train_duration = time.time() - t0_train
print(f"\nFull training completed in {t_train_duration:.1f}s")

# =========================================================================
# WORK: Write script.py (GBDT-only, L2 objective + recency weight, no TabM/torch)
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
from catboost import CatBoostRegressor
import xgboost as xgb

t0 = time.time()
print("Starting DACON Submission Inference Pipeline (GBDT L2-objective + recency-weighted, 5-seed)...")

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.025, 0.0, -0.05

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

X_test_cb = X_test.copy()
for c in cat_cols:
    X_test_cb[c] = X_test_cb[c].astype(int).astype(str)
for c in [col for col in X_test_cb.columns if col not in cat_cols]:
    X_test_cb[c] = X_test_cb[c].astype(np.float32)

X_test_xgb = X_test.copy()
for c in cat_cols:
    X_test_xgb[c] = X_test_xgb[c].astype('category').cat.codes.astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

print("Predicting with GBDT 3-model ensemble (L2-objective, 5-seed bagged)...")
p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))
for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test)
    m_cb = CatBoostRegressor()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict(X_test_cb)
    m_xgb = xgb.XGBRegressor()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict(X_test_xgb)

n_seeds = len(SEEDS)
p_lgb = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)
p_final = np.clip(W_LGB * p_lgb + W_CB * p_cb + W_XGB * p_xgb, 1e-6, 1 - 1e-6)
print(f"GBDT ensemble done ({time.time()-t0:.1f}s elapsed)")

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

print("\n" + "=" * 70)
print("[Task 3] Writing requirements.txt (torch 불필요 — GBDT-only)")
print("=" * 70)
req_content = """lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=1.7.0
"""
with open(SUBMIT_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)
print(req_content)

print("\n" + "=" * 70)
print("[Task 4] Copying local pipeline modules")
print("=" * 70)
modules_to_copy = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
for m_file in modules_to_copy:
    shutil.copy(BASE_DIR / m_file, SUBMIT_DIR / m_file)
    print(f"  Copied: {m_file}")

print("\n" + "=" * 70)
print("[Task 5] Packaging submit_v10.zip")
print("=" * 70)
zip_path = BASE_DIR / 'work/submit_v10.zip'
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
print(f"Missing required local modules: {missing} (should be empty)")

with open('/tmp/submit_v10_build_result.json', 'w') as f:
    json.dump({
        "zip_path": str(zip_path), "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
        "train_duration_seconds": t_train_duration, "files": namelist, "missing_required_modules": missing,
        "seeds": SEEDS, "config": "L2-objective + recency(decay=0.7), GBDT-only, no TabM",
        "cv_nested_honest_skill": 869.90,
    }, f, indent=2)

print(f"\nBUILD COMPLETE. Total time: {(time.time()-t0_train)/60:.1f} min")
