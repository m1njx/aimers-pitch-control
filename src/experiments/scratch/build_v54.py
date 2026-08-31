"""
v54: v50과 동일한 25개 모델, 블렌드 비율 + 캘리브레이션만 변경
- GBDT_bin 10%, MLP 30%, MSE 60% (v50: 25/50/25)
- SCALE=1.29, SHIFT=+0.006 (v50: 1.10, -0.0035)
- 카운트 쉬프트 제거
"""
import os, shutil, zipfile

BASE_DIR = os.path.expanduser('~/LG_data')
v50_dir = os.path.join(BASE_DIR, 'work', 'submit_v50')
v54_dir = os.path.join(BASE_DIR, 'work', 'submit_v54')

# 1. Copy v50 entirely
if os.path.exists(v54_dir):
    shutil.rmtree(v54_dir)
shutil.copytree(v50_dir, v54_dir, ignore=shutil.ignore_patterns('output', 'data', '__pycache__', '.DS_Store', 'catboost_info'))

# 2. Remove output/data dirs if copied
for d in ['output', 'data']:
    p = os.path.join(v54_dir, d)
    if os.path.exists(p):
        shutil.rmtree(p)

print(f"Copied v50 → v54")
print(f"Files in v54/model: {len(os.listdir(os.path.join(v54_dir, 'model')))}")
