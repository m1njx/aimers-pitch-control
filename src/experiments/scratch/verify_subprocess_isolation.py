import sys
import os
import shutil
import zipfile
import subprocess
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
FIXED_ZIP = BASE_DIR / 'work/submit_v5.zip'
ISO_DIR = Path('/tmp/clean_test_v5_subprocess_verify')

print("="*70)
print("[Subprocess Clean Verification] Testing submit_v5.zip in Subprocess")
print("="*70)

if ISO_DIR.exists():
    shutil.rmtree(ISO_DIR)
ISO_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(FIXED_ZIP, 'r') as zipf:
    zipf.extractall(ISO_DIR)

(ISO_DIR / 'data').mkdir(exist_ok=True)
(ISO_DIR / 'output').mkdir(exist_ok=True)

df_tr_sample = pd.read_csv(BASE_DIR / 'open/data/train.csv', nrows=5)
df_tr_sample.drop(columns=['control_success']).to_csv(ISO_DIR / 'data/test.csv', index=False)
df_tr_sample[['row_id', 'control_success']].to_csv(ISO_DIR / 'data/sample_submission.csv', index=False)

# Run script.py via subprocess inside ISO_DIR with clean PYTHONPATH
clean_env = os.environ.copy()
clean_env['PYTHONPATH'] = str(ISO_DIR)

cmd = [sys.executable, str(ISO_DIR / 'script.py')]
res = subprocess.run(cmd, cwd=str(ISO_DIR), env=clean_env, capture_output=True, text=True)

print(f"Subprocess Return Code: {res.returncode}")
print(f"Subprocess Stdout:\n{res.stdout}")
print(f"Subprocess Stderr:\n{res.stderr}")

assert res.returncode == 0, "Subprocess execution failed!"

sub_df = pd.read_csv(ISO_DIR / 'output/submission.csv')
print(f"Submission generated successfully! Shape: {sub_df.shape[0]} rows x {sub_df.shape[1]} cols")
print(f"Control Success Mean: {sub_df['control_success'].mean():.6f}")

print("\n🎉 Subprocess Clean Verification Passed 100%!")
