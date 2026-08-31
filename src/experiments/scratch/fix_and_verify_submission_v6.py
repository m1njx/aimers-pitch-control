"""
fix_and_verify_submission_v6.py
build_submission_v6.py가 만든 zip에 로컬 모듈(preprocessing.py 등)이 빠져있어
5차 제출과 동일한 ModuleNotFoundError 재발 위험이 있었음. 이를 수정하고,
101/102번 보고서와 동일한 "100% 격리 환경" 방법론으로 재검증한다.
재학습은 이미 완료된 9개 모델(work/submit_v6/model/)을 재사용하여 시간을 아낀다.
"""
import sys
import os
import shutil
import zipfile
import json
import time
import subprocess
import warnings
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
sys.path.insert(0, str(BASE_DIR))
import config

OUTPUTS_DIR = BASE_DIR / 'outputs'
SUBMIT_DIR = BASE_DIR / 'work/submit_v6'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# =========================================================================
# STEP 1: Audit current submit_v6 zip contents (before fix)
# =========================================================================
print("=" * 70)
print("[Step 1] Audit submit_v6.zip contents BEFORE fix")
print("=" * 70)
v6_zip_path = BASE_DIR / 'work/submit_v6.zip'
with zipfile.ZipFile(v6_zip_path, 'r') as zf:
    before_files = zf.namelist()
required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
missing_before = [m for m in required_modules if m not in before_files]
print(f"Files before fix: {before_files}")
print(f"Missing local modules (same class of bug as 5차 ModuleNotFoundError): {missing_before}")

# =========================================================================
# STEP 2: Rewrite script.py with __file__-based paths + explicit sys.path
# (v4's PROVEN working pattern, not v5's original cwd-based / no-sys.path pattern)
# =========================================================================
print("\n" + "=" * 70)
print("[Step 2] Rewrite script.py (robust __file__-based paths + sys.path insertion)")
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
print("Starting DACON 6th Submission Inference Pipeline (3-seed bagged ensemble)...")

SEEDS = [42, 100, 2024]
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

print("Loading trained models and predicting (3-seed bagged per model type)...")

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

# Weighted Ensemble (LGBM 15% + CatBoost 75% + XGBoost 10%), each model 3-seed prediction-bagged
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
print("script.py rewritten with __file__-based SCRIPT_DIR + explicit sys.path insertion.")

# =========================================================================
# STEP 3: Copy required local pipeline modules into SUBMIT_DIR root
# =========================================================================
print("\n" + "=" * 70)
print("[Step 3] Copy local pipeline modules into work/submit_v6/")
print("=" * 70)

modules_to_copy = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
for m_file in modules_to_copy:
    src_file = BASE_DIR / m_file
    shutil.copy(src_file, SUBMIT_DIR / m_file)
    print(f"  Copied: {m_file}")

# =========================================================================
# STEP 4: Re-zip
# =========================================================================
print("\n" + "=" * 70)
print("[Step 4] Re-package submit_v6.zip")
print("=" * 70)

if v6_zip_path.exists():
    v6_zip_path.unlink()

with zipfile.ZipFile(v6_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

with zipfile.ZipFile(v6_zip_path, 'r') as zf:
    after_files = zf.namelist()
missing_after = [m for m in required_modules if m not in after_files]
print(f"Files after fix ({len(after_files)}개): {sorted(after_files)}")
print(f"Missing local modules after fix: {missing_after}")
print(f"Zip size: {v6_zip_path.stat().st_size / (1024*1024):.2f} MB")

# =========================================================================
# STEP 5: TRUE 100%-isolated verification via a SEPARATE PYTHON SUBPROCESS
# (not exec() in the same contaminated sys.path — a real fresh interpreter,
#  matching 101/102's rigor, but even stronger since it's a real new process)
# =========================================================================
print("\n" + "=" * 70)
print("[Step 5] 100%-isolated verification (fresh subprocess, clean cwd/env)")
print("=" * 70)

iso_dir = Path('/tmp/clean_test_v6_verify')
if iso_dir.exists():
    shutil.rmtree(iso_dir)
iso_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(v6_zip_path, 'r') as zf:
    zf.extractall(iso_dir)

(iso_dir / 'data').mkdir(exist_ok=True)
(iso_dir / 'output').mkdir(exist_ok=True)

df_sample = pd.read_csv(config.TRAIN_PATH, nrows=5)
df_sample.drop(columns=[config.TARGET_COL]).to_csv(iso_dir / 'data/test.csv', index=False)
df_sample[['row_id', config.TARGET_COL]].to_csv(iso_dir / 'data/sample_submission.csv', index=False)

# Real subprocess: fresh interpreter, PYTHONPATH scrubbed, cwd = extracted dir only.
clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
t0_iso = time.time()
proc = subprocess.run(
    [sys.executable, 'script.py'],
    cwd=str(iso_dir),
    env=clean_env,
    capture_output=True,
    text=True,
    timeout=600,
)
t_iso_duration = time.time() - t0_iso

print(f"Subprocess return code: {proc.returncode}")
print(f"--- stdout ---\n{proc.stdout}")
if proc.returncode != 0:
    print(f"--- stderr ---\n{proc.stderr}")

iso_success = proc.returncode == 0 and (iso_dir / 'output' / 'submission.csv').exists()
df_sub_iso = None
if iso_success:
    df_sub_iso = pd.read_csv(iso_dir / 'output' / 'submission.csv')
    print(f"\nSubmission shape: {df_sub_iso.shape}, columns: {list(df_sub_iso.columns)}")
    print(f"Prob mean={df_sub_iso['control_success'].mean():.6f} std={df_sub_iso['control_success'].std():.6f}")

print(f"\nIsolated subprocess execution time: {t_iso_duration:.2f}s")
print(f"Isolated verification success: {iso_success}")

# =========================================================================
# STEP 6: CPU-6 approximate simulation (macOS has no taskset/cpuset;
# approximate by capping library thread counts to 6 and re-timing inference)
# =========================================================================
print("\n" + "=" * 70)
print("[Step 6] Approximate 6-CPU inference timing (macOS has no taskset — using thread-cap proxy)")
print("=" * 70)

script_6cpu = script_content.replace(
    "m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))\n    p_lgb_sum += m_lgb.predict(X_test)",
    "m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))\n    p_lgb_sum += m_lgb.predict(X_test, num_threads=6)"
).replace(
    "m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))\n    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]",
    "m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))\n    p_cb_sum += m_cb.predict_proba(X_test_cb, thread_count=6)[:, 1]"
)
with open(iso_dir / 'script.py', 'w', encoding='utf-8') as f:
    f.write(script_6cpu)

t0_6cpu = time.time()
proc6 = subprocess.run(
    [sys.executable, 'script.py'],
    cwd=str(iso_dir),
    env=clean_env,
    capture_output=True,
    text=True,
    timeout=600,
)
t_6cpu_duration = time.time() - t0_6cpu
print(f"6-thread-capped inference subprocess time: {t_6cpu_duration:.2f}s (return code {proc6.returncode})")
if proc6.returncode != 0:
    print(f"--- stderr ---\n{proc6.stderr}")

result = {
    "before_fix_files": before_files,
    "missing_before": missing_before,
    "after_fix_files": sorted(after_files),
    "missing_after": missing_after,
    "zip_size_mb": v6_zip_path.stat().st_size / (1024 * 1024),
    "isolated_subprocess_returncode": proc.returncode,
    "isolated_subprocess_time_sec": t_iso_duration,
    "isolated_success": iso_success,
    "isolated_stderr": proc.stderr if proc.returncode != 0 else None,
    "thread_capped_6cpu_time_sec": t_6cpu_duration,
    "thread_capped_returncode": proc6.returncode,
}
with open('/tmp/submit_v6_fix_verify_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\nDone. Full result saved to /tmp/submit_v6_fix_verify_result.json")
