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
SUBMIT_DIR = BASE_DIR / 'work/submit_v6'
MODEL_DIR = SUBMIT_DIR / 'model'
DUMMY_DIR = BASE_DIR / 'work/dummy_eval_v6'

SEEDS = [42, 100, 2024]
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
print("[Task 1] Full Re-training on Entire Train Dataset (3-seed bagging)")
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
print(f"\nFull Re-training (3-seed bagging, 9 models) completed in {t_train_duration:.1f} seconds!")

# =========================================================================
# WORK 2: Write script.py for Inference (3-seed bagged)
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
from pathlib import Path
import joblib

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')

t0 = time.time()
print("Starting DACON 6th Submission Inference Pipeline (3-seed bagged ensemble)...")

SEEDS = [42, 100, 2024]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

BASE_DIR = Path.cwd()
DATA_DIR = Path('/data') if Path('/data').exists() else BASE_DIR / 'data'
if not DATA_DIR.exists():
    DATA_DIR = Path('data')

OUTPUT_DIR = Path('/output') if Path('/output').exists() else BASE_DIR / 'output'
if not OUTPUT_DIR.exists():
    OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / 'model'

test_path = DATA_DIR / 'test.csv'
if not test_path.exists():
    test_path = Path('test.csv')

print(f"Loading test data from: {test_path}")
df_test = pd.read_csv(test_path)
print(f"Test data shape: {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

prep = joblib.load(MODEL_DIR / 'preprocessor_artifacts.pkl')
prep.trackman_builder = joblib.load(MODEL_DIR / 'trackman_artifacts.pkl')

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

print("Loading trained models and predicting (3-seed bagged per model type)...")

p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))

for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=str(MODEL_DIR / f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test)

    m_cb = CatBoostClassifier()
    m_cb.load_model(str(MODEL_DIR / f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]

    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(str(MODEL_DIR / f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]

n_seeds = len(SEEDS)
p_lgb = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)

# Weighted Ensemble (LGBM 15% + CatBoost 75% + XGBoost 10%), each model 3-seed prediction-bagged
p_final = np.clip(W_LGB * p_lgb + W_CB * p_cb + W_XGB * p_xgb, 1e-6, 1 - 1e-6)

sub = pd.DataFrame({
    'row_id': df_test['row_id'],
    'control_success': p_final
})

sub_path = OUTPUT_DIR / 'submission.csv'
sub.to_csv(sub_path, index=False)

elapsed = time.time() - t0
print(f"Inference completed & submission saved to {sub_path} in {elapsed:.2f} seconds!")
"""

with open(SUBMIT_DIR / 'script.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

print("script.py written successfully.")

# =========================================================================
# WORK 3: Write requirements.txt
# =========================================================================
print("\n" + "=" * 70)
print("[Task 3] Writing requirements.txt")
print("=" * 70)

import catboost
req_content = f"""lightgbm=={lgb.__version__}
catboost=={catboost.__version__}
xgboost=={xgb.__version__}
"""

with open(SUBMIT_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)

print("requirements.txt written:")
print(req_content)

# =========================================================================
# WORK 4: Packaging submit_v6.zip
# =========================================================================
print("\n" + "=" * 70)
print("[Task 4] Packaging submit_v6.zip")
print("=" * 70)

zip_path = BASE_DIR / 'work/submit_v6.zip'
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

print(f"Created zip archive: {zip_path} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")

with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = zipf.namelist()
    print("Zip Structure Content List:", namelist)

# =========================================================================
# WORK 5: Dummy Evaluation Rehearsal
# =========================================================================
print("\n" + "=" * 70)
print("[Task 5] Dummy Evaluation Rehearsal in work/dummy_eval_v6/")
print("=" * 70)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(DUMMY_DIR)

dummy_data_dir = DUMMY_DIR / 'data'
dummy_output_dir = DUMMY_DIR / 'output'
dummy_data_dir.mkdir(parents=True, exist_ok=True)
dummy_output_dir.mkdir(parents=True, exist_ok=True)

df_sample_test = df_train.head(5).copy().drop(columns=[config.TARGET_COL])
df_sample_test.to_csv(dummy_data_dir / 'test.csv', index=False)
df_train.head(5)[['row_id', config.TARGET_COL]].to_csv(dummy_data_dir / 'sample_submission.csv', index=False)

t0_rehearsal = time.time()
orig_cwd = os.getcwd()
os.chdir(DUMMY_DIR)
exec_code = open(DUMMY_DIR / 'script.py', 'r', encoding='utf-8').read()
g_ns = {}
exec(exec_code, g_ns)
t_rehearsal_duration = time.time() - t0_rehearsal
os.chdir(orig_cwd)

df_sub_res = pd.read_csv(dummy_output_dir / 'submission.csv')
sub_rows = df_sub_res.shape[0]
sub_cols = list(df_sub_res.columns)
mean_prob = float(df_sub_res['control_success'].mean())
std_prob = float(df_sub_res['control_success'].std()) if sub_rows > 1 else 0.0
min_prob = float(df_sub_res['control_success'].min())
max_prob = float(df_sub_res['control_success'].max())

print(f"\n--- Dummy Rehearsal Verification (inference only, models pre-trained) ---")
print(f"Rehearsal Execution Time: {t_rehearsal_duration:.2f} seconds")
print(f"Submission Shape        : {sub_rows} rows x {len(sub_cols)} columns")
print(f"Columns                 : {sub_cols}")
print(f"Probability Distribution -> Mean: {mean_prob:.6f}, Std: {std_prob:.6f}, Min: {min_prob:.6f}, Max: {max_prob:.6f}")

rehearsal_summary = {
    "zip_path": str(zip_path),
    "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
    "train_duration_seconds": t_train_duration,
    "rehearsal_inference_time_seconds": t_rehearsal_duration,
    "submission_rows": sub_rows,
    "submission_cols": sub_cols,
    "prob_mean": mean_prob,
    "prob_std": std_prob,
    "prob_min": min_prob,
    "prob_max": max_prob,
}
with open('/tmp/submit_v6_result.json', 'w', encoding='utf-8') as f:
    json.dump(rehearsal_summary, f, indent=2, ensure_ascii=False)

print("\nDone. Summary saved to /tmp/submit_v6_result.json")
