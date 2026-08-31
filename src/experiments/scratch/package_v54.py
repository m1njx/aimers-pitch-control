import os, shutil, zipfile, subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.expanduser('~/LG_data')
v54_dir = os.path.join(BASE_DIR, 'work', 'submit_v54')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v54.zip')
pokemon_zip = '~/pipeline_src/submit_v54.zip'

# Clean
for root, dirs, files in os.walk(v54_dir, topdown=False):
    for f in files:
        if f.endswith('.pyc') or f == '.DS_Store' or f.startswith('._'):
            os.remove(os.path.join(root, f))
    for d in dirs:
        if d in ['__pycache__', 'output', 'data', 'catboost_info', '.ipynb_checkpoints']:
            shutil.rmtree(os.path.join(root, d))

# Package
if os.path.exists(zip_path):
    os.remove(zip_path)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(v54_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, v54_dir)
            zf.write(full_path, rel_path)

print(f"Packaged submit_v54.zip ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

with zipfile.ZipFile(zip_path, 'r') as zf:
    namelist = zf.namelist()
    print(f"Total files: {len(namelist)}")
    assert 'requirements.txt' in namelist
    assert 'script.py' in namelist

# Sandbox test
sandbox_dir = '/tmp/v54_sandbox'
if os.path.exists(sandbox_dir):
    shutil.rmtree(sandbox_dir)
os.makedirs(sandbox_dir)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(sandbox_dir)

os.makedirs(os.path.join(sandbox_dir, 'data'), exist_ok=True)
shutil.copy2(os.path.join(BASE_DIR, 'open', 'data', 'test.csv'), os.path.join(sandbox_dir, 'data', 'test.csv'))

res = subprocess.run([
    'python3', 'script.py'
], cwd=sandbox_dir, capture_output=True, text=True)

if res.returncode != 0:
    print(f"SANDBOX FAILED:\n{res.stderr}")
    exit(1)

print("\n--- Sandbox Output ---")
print(res.stdout)

# Verify
sub = pd.read_csv(os.path.join(sandbox_dir, 'output', 'submission.csv'))
assert sub.shape == (5, 2)
assert list(sub.columns) == ['row_id', 'control_success']
assert sub.isna().sum().sum() == 0
print("v54 predictions:")
print(sub)

# Compare with v50
v50_sub = pd.read_csv(os.path.join(BASE_DIR, 'work', 'submit_v50', 'output', 'submission.csv'))
print(f"\nv50 mean: {v50_sub['control_success'].mean():.6f}")
print(f"v54 mean: {sub['control_success'].mean():.6f}")
for i in range(5):
    d = sub['control_success'].iloc[i] - v50_sub['control_success'].iloc[i]
    print(f"  Row {i}: v50={v50_sub['control_success'].iloc[i]:.6f} → v54={sub['control_success'].iloc[i]:.6f} (Δ={d:+.6f})")

# Deploy
shutil.copy2(zip_path, pokemon_zip)
print(f"\n✅ Deployed to {pokemon_zip} ({os.path.getsize(pokemon_zip)/(1024*1024):.2f} MB)")
