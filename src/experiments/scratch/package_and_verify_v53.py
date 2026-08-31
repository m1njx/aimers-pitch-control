import os
import shutil
import zipfile
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.expanduser('~/LG_data')
work_v53_dir = os.path.join(BASE_DIR, 'work', 'submit_v53')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v53.zip')
pokemon_zip = '~/pipeline_src/submit_v53.zip'

# 1. Clean temporary files & redundant models not used by script.py
for root, dirs, files in os.walk(work_v53_dir, topdown=False):
    for f in files:
        if f.endswith('.pyc') or f == '.DS_Store' or f.startswith('._'):
            os.remove(os.path.join(root, f))
        # remove old unused kfold files if any
        if 'fold' in f or f == 'meta_ridge_model.pkl' or f == 'tabular_resnet_artifacts.pkl' or f == 'count_shifts_artifact.pkl':
            os.remove(os.path.join(root, f))
    for d in dirs:
        if d in ['__pycache__', 'output', 'data', 'catboost_info', '.ipynb_checkpoints']:
            shutil.rmtree(os.path.join(root, d))

# 2. Package into ZIP
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v53_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v53_dir)
            zf.write(full_path, rel_path)

print(f"Packaged submit_v53.zip ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

with zipfile.ZipFile(zip_path, 'r') as zf:
    namelist = zf.namelist()
    print(f"Total files in zip: {len(namelist)}")
    assert 'requirements.txt' in namelist, "requirements.txt MUST be in zip!"
    assert 'script.py' in namelist, "script.py MUST be in zip!"

# 3. Isolated Sandbox Test
sandbox_dir = '/tmp/v53_sandbox_verified'
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

# 4. Check submission file
sub_file = os.path.join(sandbox_dir, 'output', 'submission.csv')
assert os.path.exists(sub_file), "submission.csv not generated!"
df_v53 = pd.read_csv(sub_file)
assert df_v53.shape == (5, 2), f"Unexpected shape {df_v53.shape}"
assert list(df_v53.columns) == ['row_id', 'control_success'], f"Unexpected columns {df_v53.columns}"
assert df_v53.isna().sum().sum() == 0, "NaNs found!"

print("v53 submission.csv predictions:")
print(df_v53)

# 5. Compare with v50
v50_sub = pd.read_csv(os.path.join(BASE_DIR, 'work', 'submit_v50', 'output', 'submission.csv'))
print("\n--- Comparison with v50 (1,032.82 pts Master) ---")
print("v50 Mean:", v50_sub['control_success'].mean())
print("v53 Mean:", df_v53['control_success'].mean())
print("Mean Absolute Deviation from v50:", np.abs(df_v53['control_success'] - v50_sub['control_success']).mean())

# 6. Copy to pokemon directory
shutil.copy2(zip_path, pokemon_zip)
print(f"\n✅ Successfully deployed submit_v53.zip to {pokemon_zip} ({os.path.getsize(pokemon_zip)/(1024*1024):.2f} MB)")
