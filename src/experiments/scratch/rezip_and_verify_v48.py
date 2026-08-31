import os
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
work_v48_dir = os.path.join(BASE_DIR, 'work', 'submit_v48')
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v48.zip')

if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_v48_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_v48_dir)
            zf.write(full_path, rel_path)

print(f"Rezipped submit_v48.zip: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")

# Now run verify_and_sync_v48.py
