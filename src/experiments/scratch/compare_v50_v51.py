import os
import shutil
import pandas as pd
import numpy as np
import subprocess

BASE_DIR = os.path.expanduser('~/LG_data')
work_v50_dir = os.path.join(BASE_DIR, 'work', 'submit_v50')
work_v51_dir = os.path.join(BASE_DIR, 'work', 'submit_v51')

os.makedirs(os.path.join(work_v50_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(work_v51_dir, 'data'), exist_ok=True)
shutil.copy2(os.path.join(BASE_DIR, 'open', 'data', 'test.csv'), os.path.join(work_v50_dir, 'data', 'test.csv'))
shutil.copy2(os.path.join(BASE_DIR, 'open', 'data', 'test.csv'), os.path.join(work_v51_dir, 'data', 'test.csv'))

# Run v50
r50 = subprocess.run(['python3', 'script.py'], cwd=work_v50_dir, capture_output=True, text=True)
sub_v50 = pd.read_csv(os.path.join(work_v50_dir, 'output', 'submission.csv'))

# Run v51
r51 = subprocess.run(['python3', 'script.py'], cwd=work_v51_dir, capture_output=True, text=True)
sub_v51 = pd.read_csv(os.path.join(work_v51_dir, 'output', 'submission.csv'))

p50 = sub_v50['control_success'].values
p51 = sub_v51['control_success'].values

print("=== Comparison on test.csv sample ===")
print(f"v50: Mean={p50.mean():.6f}, Min={p50.min():.6f}, Max={p50.max():.6f}, Std={p50.std():.6f}")
print(f"v51: Mean={p51.mean():.6f}, Min={p51.min():.6f}, Max={p51.max():.6f}, Std={p51.std():.6f}")
print(f"Difference (v51 - v50): {p51 - p50}")
print(f"Mean Difference: {p51.mean() - p50.mean():+.6f}")
