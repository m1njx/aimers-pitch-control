import os
import shutil
import zipfile
import torch
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
work_v48_dir = os.path.join(BASE_DIR, 'work', 'submit_v48')

if os.path.exists(work_v48_dir):
    shutil.rmtree(work_v48_dir)
os.makedirs(work_v48_dir, exist_ok=True)
os.makedirs(os.path.join(work_v48_dir, 'model'), exist_ok=True)

# Copy all files from v42
for item in os.listdir(work_v42_dir):
    src = os.path.join(work_v42_dir, item)
    dst = os.path.join(work_v48_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

# Now load the 15-seed SWA weights from ~/LG_data/work/submit_v48/model/mlp_artifacts.pkl
# Wait, let's load from the saved mlp_artifacts.pkl from previous run if exists, or check where it was saved
art_path = '~/LG_data/scratch/v48_mlp_artifacts_temp.pkl'
if not os.path.exists(art_path):
    # Check if work_v48 had it before rmtree or reload
    pass

print("Packaging v48 files...")
