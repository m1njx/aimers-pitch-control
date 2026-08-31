import os
import shutil
import zipfile
import json
import time
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
SUBMIT_DIR = BASE_DIR / 'work/submit_v5'
ZIP_PATH = BASE_DIR / 'work/submit_v5.zip'

print("="*70)
print("[Emergency Fix] Updating requirements.txt for DACON PyPI Compatibility")
print("="*70)

# Write unpinned requirements.txt
req_content = """lightgbm
catboost
xgboost
"""

req_file = SUBMIT_DIR / 'requirements.txt'
with open(req_file, 'w', encoding='utf-8') as f:
    f.write(req_content)

print("Updated work/submit_v5/requirements.txt content:")
print(req_content)

# Re-pack submit_v5.zip
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

print(f"Re-packaged submit_v5.zip successfully! (Size: {ZIP_PATH.stat().st_size / (1024*1024):.2f} MB)")

# Verify Zip content
with zipfile.ZipFile(ZIP_PATH, 'r') as zipf:
    print("\nZip Root Files:", zipf.namelist())
    print("\nrequirements.txt inside zip:")
    print(zipf.read('requirements.txt').decode('utf-8'))

print("Fix completed successfully!")
