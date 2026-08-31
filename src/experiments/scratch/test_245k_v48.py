import time
import subprocess
import pandas as pd

sandbox_dir = '/tmp/v48_sandbox'
orig_test_csv = '~/LG_data/open/data/test.csv'

# Copy the real 245K test.csv
dest_test = '/tmp/v48_sandbox/data/test.csv'
import shutil
shutil.copy2(orig_test_csv, dest_test)

t0 = time.time()
res = subprocess.run([
    'python3',
    'script.py'
], cwd=sandbox_dir, capture_output=True, text=True)
el = time.time() - t0

if res.returncode != 0:
    print(f"FAILED: {res.stderr}")
else:
    print(f"SUCCESS on 245,789 rows in {el:.2f}s!")
    df_sub = pd.read_csv('/tmp/v48_sandbox/output/submission.csv')
    print(f"Shape: {df_sub.shape}")
    print(f"Nulls: {df_sub.isnull().sum().to_dict()}")
    print(df_sub.describe())
