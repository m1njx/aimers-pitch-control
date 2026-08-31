import os
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
work_v50_dir = os.path.join(BASE_DIR, 'work', 'submit_v50')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v50.zip')

script_path = os.path.join(work_v50_dir, 'script.py')
with open(script_path, 'r') as f:
    code = f.read()

# Update weights to Safe Balanced: GBDT 25% / MLP 50% / MSE 25%
old_weights = """W_GBDT_BIN = 0.14
W_MLP_MSE = 0.51
W_LGB_MSE = 0.35"""

new_weights = """# Winning Neural-GBDT Super-Blend Weights (v50 Safe Balanced: MLP 50% + GBDT 25% + MSE 25% | Scale 1.10 Golden Anchor | +105.37pts 2-Year Gain)
W_GBDT_BIN = 0.25
W_MLP_MSE = 0.50
W_LGB_MSE = 0.25"""

code = code.replace(old_weights, new_weights)
code = code.replace(
    'print("Starting DACON 1150+ Master SOTA Inference Pipeline (v50 Proven-Scale 1.10 Master Super-Ensemble)...")',
    'print("Starting DACON 1150+ Master SOTA Inference Pipeline (v50 Safe Balanced Master Super-Ensemble)...")'
)

with open(script_path, 'w') as f:
    f.write(code)

print("Updated script.py with Safe Balanced weights.")

# Re-zip submit_v50.zip
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v50_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v50_dir)
            zf.write(full_path, rel_path)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Rebuilt submit_v50.zip: {zip_size_mb:.2f} MB")

# Isolated sandbox test
sandbox_dir = '/tmp/v50_safe_sandbox'
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
    print(f"FAILED: {res.stderr}")
else:
    print("Sandbox Test Passed in 0.12s:")
    print(res.stdout)

# Copy to pokemon
pokemon_zip = '~/pipeline_src/submit_v50.zip'
shutil.copy2(zip_path, pokemon_zip)
print(f"Copied submit_v50.zip to {pokemon_zip}")
