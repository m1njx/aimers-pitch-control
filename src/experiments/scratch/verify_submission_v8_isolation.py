"""
verify_submission_v8_isolation.py
8차 제출 준비 최종 작업: 138/151번에서 확립한 100%-격리 서브프로세스 방법론으로
submit_v8.zip(GBDT 5-seed + TabM 5-seed 블렌딩) 검증.
결과 저장: outputs/158_submit_v8_isolation_check.md
"""
import sys, shutil, zipfile, json, time, subprocess, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
from datetime import datetime
import pandas as pd

import config

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

zip_path = BASE_DIR / 'work/submit_v8.zip'
with zipfile.ZipFile(zip_path, 'r') as zf:
    zip_files = sorted(zf.namelist())

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py', 'tabm_inference_model.py']
missing = [m for m in required_modules if m not in zip_files]

iso_dir = Path('/tmp/clean_test_v8_verify')
if iso_dir.exists():
    shutil.rmtree(iso_dir)
iso_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(iso_dir)

(iso_dir / 'data').mkdir(exist_ok=True)
(iso_dir / 'output').mkdir(exist_ok=True)

df_sample = pd.read_csv(config.TRAIN_PATH, nrows=5)
df_sample.drop(columns=[config.TARGET_COL]).to_csv(iso_dir / 'data/test.csv', index=False)
df_sample[['row_id', config.TARGET_COL]].to_csv(iso_dir / 'data/sample_submission.csv', index=False)

import os
clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
print(f"Running script.py in fully isolated subprocess (cwd={iso_dir}, PYTHONPATH removed)...")
t0 = time.time()
proc = subprocess.run(
    [sys.executable, 'script.py'],
    cwd=str(iso_dir),
    env=clean_env,
    capture_output=True,
    text=True,
    timeout=600,
)
t_elapsed = time.time() - t0

print(f"Return code: {proc.returncode}")
print(f"--- stdout ---\n{proc.stdout}")
if proc.returncode != 0:
    print(f"--- stderr ---\n{proc.stderr}")

success = proc.returncode == 0 and (iso_dir / 'output' / 'submission.csv').exists()
sub_shape = None
sub_cols = None
prob_stats = {}
cols_match_sample = False
row_ids_match = False
if success:
    df_sub = pd.read_csv(iso_dir / 'output' / 'submission.csv')
    sub_shape = list(df_sub.shape)
    sub_cols = list(df_sub.columns)
    prob_stats = dict(
        mean=float(df_sub['control_success'].mean()),
        std=float(df_sub['control_success'].std()),
        min=float(df_sub['control_success'].min()),
        max=float(df_sub['control_success'].max()),
    )
    cols_match_sample = sub_cols == ['row_id', 'control_success']
    row_ids_match = set(df_sub['row_id']) == set(df_sample['row_id'])

all_pass = success and not missing and cols_match_sample and row_ids_match

lines = [
    "# 158. 8차 제출 패키지(submit_v8.zip) 100%-격리 검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **구성**: GBDT 3종 5-seed(15/75/10) + TabM 5-seed 블렌딩(GBDT 52% + TabM 48%), 157번에서 확정된 nested-honest 888.43점 구성",
    "- **검증 방법**: 138/151번 방식 — 진짜 새 파이썬 서브프로세스, `PYTHONPATH` 제거, 완전히 새로운 임시 디렉토리\n",
    "---\n",
    "## 1. 로컬 모듈 포함 여부\n",
    f"- zip 내용물 ({len(zip_files)}개 파일)",
    f"- 필수 모듈 누락 여부: **`{missing if missing else '없음 (전부 포함, tabm_inference_model.py 포함 확인)'}`**\n",
    "## 2. 격리 서브프로세스 실행 결과\n",
    "| 항목 | 값 |",
    "|:---|:---:|",
    f"| 종료 코드 | `{proc.returncode}` (0=정상) |",
    f"| 실행 시간 | `{t_elapsed:.2f}초` |",
    f"| 10분(600초) 제한 대비 여유 | `{(600-t_elapsed)/600*100:.1f}%` |",
    f"| submission.csv 생성 | `{success}` |",
    f"| 행/열 형식 | `{sub_shape}` (기대: `[5, 2]`) |",
    f"| 컬럼명 | `{sub_cols}` (기대: `['row_id', 'control_success']`) |",
    f"| row_id 일치 | `{row_ids_match}` |",
    f"| 예측 확률 분포 | 평균 `{prob_stats.get('mean', 'N/A')}`, 표준편차 `{prob_stats.get('std', 'N/A')}`, 범위 `[{prob_stats.get('min', 'N/A')}, {prob_stats.get('max', 'N/A')}]` |",
    "\n---\n",
    "## 3. 최종 판정\n",
]
if all_pass:
    lines.append("> ## ✅ 8차 제출 준비 완료 (Ready for 8th Submission)\n")
    lines.append(
        "> `work/submit_v8.zip`은 로컬 모듈 전체 포함(TabM 클래스 정의 포함), 100%-격리 서브프로세스 정상 실행, "
        "제출 형식 100% 일치, 10분 제한 대비 충분한 여유를 모두 통과했습니다. "
        "GBDT+TabM 블렌딩은 5-seed 정식 표준으로 nested-honest 검증까지 통과한 구성(157번, +44.74점)입니다. "
        "실제 데이콘 업로드는 사용자의 명시적 지시가 있을 때 진행합니다."
    )
else:
    lines.append("> ## ❌ 검증 실패 — 제출 전 추가 조치 필요\n")
    if missing:
        lines.append(f"> - 누락된 모듈: {missing}")
    if not success:
        lines.append(f"> - 서브프로세스 실행 실패 (returncode={proc.returncode})")
        lines.append(f"> - stderr: ```\n{proc.stderr[-3000:]}\n```")
    if not cols_match_sample:
        lines.append(f"> - 컬럼 형식 불일치: {sub_cols}")
    if not row_ids_match:
        lines.append("> - row_id 불일치")

with open(OUTPUTS_DIR / '158_submit_v8_isolation_check.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport 158 written! all_pass={all_pass}")
