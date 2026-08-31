"""
fix_v13_row_independence.py

CRITICAL BUG FOUND (2026-08-13, triggered by DACON's official notice about strict
row-independence enforcement): script.py's XGBoost branch re-derives categorical
codes via `.astype('category').cat.codes` on the TEST BATCH itself. This is
batch-dependent -- a row's XGB input features (and therefore its prediction)
depend on which OTHER rows happen to be present in the same test.csv call.
Verified empirically: full-batch vs single-row inference gave DIFFERENT
predictions for the same row_id.

ROOT CAUSE: at training time, `.astype('category').cat.codes` was applied to
the FULL train.csv (1.475M rows), which contains a dense contiguous 1..N range
for every one of these categorical columns (confirmed by direct inspection).
On a dense 1..N population, `.cat.codes` is exactly `value - 1` for every row
(since sorted rank of value V among {1..N} is V-1). At test time, a small
batch does NOT contain the full 1..N range, so `.cat.codes` produces a
different (batch-dependent) mapping -- both a regulation-4 violation AND a
genuine train/serve skew bug (the model receives codes it was never trained on).

FIX: replace `.astype('category').cat.codes` at inference with the fixed,
row-independent arithmetic `value - 1` (matching exactly what the model saw
during training on the full population). `count_x_base` is a special case --
it's already 0-indexed via a dict built at training time (`prep.count_x_base_map`),
so it does NOT get the -1 shift.

This script patches v12's script.py, rebuilds as v13 (same trained models,
zero retraining), and rigorously re-verifies row-independence: batch prediction
must EXACTLY equal the single-row prediction for every row_id.
"""
import sys, os, shutil, zipfile, json, subprocess
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import pandas as pd

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
SRC_DIR = BASE_DIR / 'work/submit_v12'
DST_DIR = BASE_DIR / 'work/submit_v13'

if DST_DIR.exists():
    shutil.rmtree(DST_DIR)
shutil.copytree(SRC_DIR, DST_DIR)

script_path = DST_DIR / 'script.py'
content = script_path.read_text()

OLD_BLOCK = """X_test_xgb = X_test.copy()
for c in cat_cols:
    X_test_xgb[c] = X_test_xgb[c].astype('category').cat.codes.astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)"""

NEW_BLOCK = """# --- FIX (v13, 2026-08-13): the previous `.astype('category').cat.codes` derived
# category codes from whichever values happen to be present in THIS test.csv batch,
# which is (a) a violation of the row-independence rule (DACON 공지: 평가 데이터의
# 다른 행이 현재 행의 추론에 영향을 주면 안 됨) and (b) a train/serve mismatch --
# at training time the same call ran on the FULL train.csv (dense 1..N range for
# every one of these columns), where `.cat.codes` is exactly `value - 1` for every
# row. We reproduce that exact fixed mapping here so each row's XGB features depend
# ONLY on that row's own already-encoded categorical value (verified empirically:
# batch-of-5 vs single-row-at-a-time now give bit-identical predictions).
X_test_xgb = X_test.copy()
for c in cat_cols:
    if c == 'count_x_base':
        # already 0-indexed via a fixed dict built from train.csv at training time
        X_test_xgb[c] = X_test_xgb[c].astype(np.float32)
    else:
        X_test_xgb[c] = (X_test_xgb[c] - 1).astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)"""

assert OLD_BLOCK in content, "OLD_BLOCK not found verbatim in v12 script.py -- aborting"
content = content.replace(OLD_BLOCK, NEW_BLOCK)
content = content.replace(
    'print("Starting DACON Submission Inference Pipeline (GBDT + asof_dec + shift extrapolation, 5-seed classification)...")',
    'print("Starting DACON Submission Inference Pipeline (GBDT + asof_dec + shift extrapolation, row-independence-fixed XGB encoding, 5-seed classification)...")'
)
script_path.write_text(content)
print("Patched script.py written to", script_path)

# ---- rebuild zip ----
zip_path = BASE_DIR / 'work/submit_v13.zip'
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(DST_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(DST_DIR)
            zipf.write(file_path, arcname)
with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = sorted(zipf.namelist())
print(f"Created {zip_path} ({zip_path.stat().st_size/(1024*1024):.2f} MB), {len(namelist)} files")

# ---- rigorous row-independence re-verification ----
print("\n=== Row-independence verification (batch vs single-row-at-a-time) ===")
df_real_test = pd.read_csv(BASE_DIR / 'open/data/test.csv')

work_root = Path('/tmp/v13_row_indep')
if work_root.exists():
    shutil.rmtree(work_root)
work_root.mkdir(parents=True)

batch_dir = work_root / 'batch'
shutil.copytree(DST_DIR, batch_dir)
(batch_dir / 'data').mkdir(exist_ok=True)
(batch_dir / 'output').mkdir(exist_ok=True)
df_real_test.to_csv(batch_dir / 'data/test.csv', index=False)
clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
proc = subprocess.run([sys.executable, 'script.py'], cwd=str(batch_dir), env=clean_env,
                       capture_output=True, text=True, timeout=600)
assert proc.returncode == 0, f"batch run failed:\n{proc.stderr}"
df_batch = pd.read_csv(batch_dir / 'output/submission.csv').set_index('row_id')['control_success']
print("Batch predictions:")
print(df_batch)

max_diff = 0.0
all_identical = True
for i, row_id in enumerate(df_real_test['row_id']):
    single_dir = work_root / f'single_{i}'
    shutil.copytree(DST_DIR, single_dir)
    (single_dir / 'data').mkdir(exist_ok=True)
    (single_dir / 'output').mkdir(exist_ok=True)
    df_real_test.iloc[[i]].to_csv(single_dir / 'data/test.csv', index=False)
    proc = subprocess.run([sys.executable, 'script.py'], cwd=str(single_dir), env=clean_env,
                           capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"single-row run failed for {row_id}:\n{proc.stderr}"
    p_single = pd.read_csv(single_dir / 'output/submission.csv').set_index('row_id')['control_success'][row_id]
    p_batch = df_batch[row_id]
    diff = abs(p_single - p_batch)
    max_diff = max(max_diff, diff)
    match = diff < 1e-9
    all_identical = all_identical and match
    print(f"  {row_id}: batch={p_batch:.10f}  single={p_single:.10f}  diff={diff:.2e}  {'OK' if match else '*** MISMATCH ***'}")

print(f"\nmax_diff = {max_diff:.2e}")
print(f"ALL IDENTICAL (row-independence verified): {all_identical}")

with open('/tmp/v13_row_independence_result.json', 'w') as f:
    json.dump({'all_identical': bool(all_identical), 'max_diff': float(max_diff)}, f, indent=2)
