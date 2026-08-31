"""
repackage_v6_after_requirements_fix.py
requirements.txt를 정확한 버전 고정(==)에서 하한선(>=)으로 완화한 뒤 재압축하고,
동일한 100%-격리 서브프로세스 방법론으로 재검증한다. 모델 재학습은 필요 없음.
"""
import sys, os, shutil, zipfile, json, time, subprocess
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
sys.path.insert(0, str(BASE_DIR))
import config

SUBMIT_DIR = BASE_DIR / 'work/submit_v6'
v6_zip_path = BASE_DIR / 'work/submit_v6.zip'

print("Current requirements.txt:")
print((SUBMIT_DIR / 'requirements.txt').read_text())

if v6_zip_path.exists():
    v6_zip_path.unlink()

with zipfile.ZipFile(v6_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

with zipfile.ZipFile(v6_zip_path, 'r') as zf:
    files = zf.namelist()
    req_content = zf.read('requirements.txt').decode('utf-8')
print(f"\nRe-zipped. Files ({len(files)}): {sorted(files)}")
print(f"requirements.txt inside zip:\n{req_content}")
print(f"Zip size: {v6_zip_path.stat().st_size / (1024*1024):.2f} MB")

# Re-verify with isolated subprocess (same methodology as 138번)
iso_dir = Path('/tmp/clean_test_v6_verify_reqfix')
if iso_dir.exists():
    shutil.rmtree(iso_dir)
iso_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(v6_zip_path, 'r') as zf:
    zf.extractall(iso_dir)

(iso_dir / 'data').mkdir(exist_ok=True)
(iso_dir / 'output').mkdir(exist_ok=True)
df_sample = pd.read_csv(config.TRAIN_PATH, nrows=5)
df_sample.drop(columns=[config.TARGET_COL]).to_csv(iso_dir / 'data/test.csv', index=False)
df_sample[['row_id', config.TARGET_COL]].to_csv(iso_dir / 'data/sample_submission.csv', index=False)

clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
t0 = time.time()
proc = subprocess.run([sys.executable, 'script.py'], cwd=str(iso_dir), env=clean_env,
                       capture_output=True, text=True, timeout=600)
elapsed = time.time() - t0

print(f"\nIsolated subprocess return code: {proc.returncode}, time: {elapsed:.2f}s")
print(f"stdout:\n{proc.stdout}")
if proc.returncode != 0:
    print(f"stderr:\n{proc.stderr}")

success = proc.returncode == 0 and (iso_dir / 'output' / 'submission.csv').exists()
if success:
    df_sub = pd.read_csv(iso_dir / 'output' / 'submission.csv')
    print(f"Submission OK: shape={df_sub.shape}, mean={df_sub['control_success'].mean():.6f}")

with open('/tmp/submit_v6_reqfix_result.json', 'w') as f:
    json.dump({
        "files": sorted(files),
        "requirements_txt": req_content,
        "zip_size_mb": v6_zip_path.stat().st_size / (1024*1024),
        "isolated_returncode": proc.returncode,
        "isolated_time_sec": elapsed,
        "isolated_success": success,
    }, f, indent=2)

print("\nDone.")
