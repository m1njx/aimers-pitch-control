import os
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
work_v52_dir = os.path.join(BASE_DIR, 'work', 'submit_v52')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v52.zip')
pokemon_zip = '~/pipeline_src/submit_v52.zip'

# 1. Check requirements.txt in v42 / create standard requirements.txt
req_src = os.path.join(work_v42_dir, 'requirements.txt')
req_dst = os.path.join(work_v52_dir, 'requirements.txt')

if os.path.exists(req_src):
    shutil.copy2(req_src, req_dst)
    print(f"Copied requirements.txt from v42: {req_dst}")
else:
    with open(req_dst, 'w') as f:
        f.write("numpy\npandas\nscipy\nscikit-learn\nlightgbm\ncatboost\nxgboost\ntorch\njoblib\n")
    print(f"Created standard requirements.txt at: {req_dst}")

with open(req_dst, 'r') as f:
    print("requirements.txt contents:\n" + f.read())

# 2. Clean temporary files
for root, dirs, files in os.walk(work_v52_dir, topdown=False):
    for f in files:
        if f.endswith('.pyc') or f == '.DS_Store' or f.startswith('._'):
            os.remove(os.path.join(root, f))
    for d in dirs:
        if d in ['__pycache__', 'output', 'data', 'catboost_info', '.ipynb_checkpoints']:
            shutil.rmtree(os.path.join(root, d))

# 3. Create zip file with requirements.txt at top level
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v52_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v52_dir)
            zf.write(full_path, rel_path)

print(f"Re-packaged submit_v52.zip ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

# 4. List files inside ZIP
print("\n--- Files in submit_v52.zip ---")
with zipfile.ZipFile(zip_path, 'r') as zf:
    namelist = zf.namelist()
    for name in sorted(namelist):
        print(f"  {name}")
assert 'requirements.txt' in namelist, "requirements.txt MUST be in zip!"
assert 'script.py' in namelist, "script.py MUST be in zip!"

# 5. Isolated sandbox execution test
sandbox_dir = '/tmp/v52_sandbox_verified'
if os.path.exists(sandbox_dir):
    shutil.rmtree(sandbox_dir)
os.makedirs(sandbox_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(sandbox_dir)

os.makedirs(os.path.join(sandbox_dir, 'data'), exist_ok=True)
shutil.copy2(os.path.join(BASE_DIR, 'open', 'data', 'test.csv'), os.path.join(sandbox_dir, 'data', 'test.csv'))

res = subprocess.run([
    'python3',
    'script.py'
], cwd=sandbox_dir, capture_output=True, text=True)

if res.returncode != 0:
    print(f"SANDBOX TEST FAILED:\n{res.stderr}")
    exit(1)

print("\n--- Sandbox Execution Output ---")
print(res.stdout)

# 6. Verify submission.csv
sub_file = os.path.join(sandbox_dir, 'output', 'submission.csv')
assert os.path.exists(sub_file), "submission.csv not generated!"
df_sub = pd.read_csv(sub_file)
assert df_sub.shape == (5, 2), f"Unexpected shape {df_sub.shape}"
assert list(df_sub.columns) == ['row_id', 'control_success'], f"Unexpected columns {df_sub.columns}"
assert df_sub.isna().sum().sum() == 0, "NaNs found!"
print("submission.csv head:\n", df_sub.head())

# 7. Copy to pokemon directory
shutil.copy2(zip_path, pokemon_zip)
print(f"\n✅ Successfully deployed verified submit_v52.zip to {pokemon_zip} ({os.path.getsize(pokemon_zip)/(1024*1024):.2f} MB)")
