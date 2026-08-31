"""
verify_submission_v11_isolation.py
9차 제출(submit_v11.zip, GBDT 5-seed + asof_dec 분해피처) 100%-격리 검증.
v9 검증(138/151/159번)과 동일한 방법론: 진짜 새 서브프로세스, PYTHONPATH 제거,
완전히 새로운 임시 디렉토리. 실제 open/data/test.csv(season=2025, 5행)를 그대로
사용해 asof_dec의 실제 배포 경로(val_season=2025 분기)까지 검증.
결과 저장: outputs/171_submit_v11_isolation_check.md
"""
import sys, shutil, zipfile, json, time, subprocess, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
from datetime import datetime
import pandas as pd

import config

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

zip_path = BASE_DIR / 'work/submit_v11.zip'
with zipfile.ZipFile(zip_path, 'r') as zf:
    zip_files = sorted(zf.namelist())

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py', 'agent2_asof_decomp2.py']
missing = [m for m in required_modules if m not in zip_files]

iso_dir = Path('/tmp/clean_test_v11_verify')
if iso_dir.exists():
    shutil.rmtree(iso_dir)
iso_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(iso_dir)

(iso_dir / 'data').mkdir(exist_ok=True)
(iso_dir / 'output').mkdir(exist_ok=True)

# 실제 test.csv(season=2025) 그대로 사용 -- asof_dec의 val_season==2025 분기 경로를 실제로 태움
df_real_test = pd.read_csv(config.TEST_PATH)
df_real_test.to_csv(iso_dir / 'data/test.csv', index=False)

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
cols_match = False
row_ids_match = False
df_sub = None
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
    cols_match = sub_cols == ['row_id', 'control_success']
    row_ids_match = set(df_sub['row_id']) == set(df_real_test['row_id'])

all_pass = success and not missing and cols_match and row_ids_match

lines = [
    "# 171. 9차 제출 패키지(submit_v11.zip) 100%-격리 검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **구성**: GBDT 3종 5-seed(15/75/10, classification objective) + asof_dec 분해피처(46개 신규),"
    " 로컬 outer(2024) 805.74점(+206.77), 4개년도 독립홀드아웃 일관성 확인(170~192번 참고)",
    "- **검증 방법**: 138/151/159번 방식 — 진짜 새 파이썬 서브프로세스, `PYTHONPATH` 제거, 완전히 새로운 임시 디렉토리."
    " 실제 `open/data/test.csv`(season=2025, 5행)를 그대로 사용해 asof_dec의 실제 val_season=2025 배포 경로를 실제로 태움.\n",
    "---\n",
    "## 1. 로컬 모듈 포함 여부\n",
    f"- zip 내용물 ({len(zip_files)}개 파일): {zip_files}",
    f"- 필수 모듈 누락 여부: **`{missing if missing else '없음 (전부 포함, agent2_asof_decomp2.py 포함 확인)'}`**\n",
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
]
if df_sub is not None:
    lines.append(f"\n실제 예측값:\n```\n{df_sub.to_string()}\n```")
lines += ["\n---\n", "## 3. 최종 판정\n"]
if all_pass:
    lines.append("> ## ✅ 9차 제출 준비 완료 (Ready for 9th Submission)\n")
    lines.append(
        "> `work/submit_v11.zip`은 로컬 모듈 전체 포함(AsofDecomposer2 클래스 정의 포함), 100%-격리 서브프로세스 정상 실행, "
        "제출 형식 100% 일치, 10분 제한 대비 충분한 여유, 실제 test.csv(season=2025)에서 asof_dec 배포 경로까지 실사용 검증을 모두 통과했습니다. "
        "실제 데이콘 업로드는 사용자의 명시적 지시가 있을 때 진행합니다."
    )
else:
    lines.append("> ## ❌ 검증 실패 — 제출 전 추가 조치 필요\n")
    if missing:
        lines.append(f"> - 누락된 모듈: {missing}")
    if not success:
        lines.append(f"> - 서브프로세스 실행 실패 (returncode={proc.returncode})")
        lines.append(f"> - stderr: ```\n{proc.stderr[-3000:]}\n```")
    if not cols_match:
        lines.append(f"> - 컬럼 형식 불일치: {sub_cols}")
    if not row_ids_match:
        lines.append("> - row_id 불일치")

with open(OUTPUTS_DIR / '171_submit_v11_isolation_check.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

with open('/tmp/v11_isolation_result.json', 'w') as f:
    json.dump({'all_pass': all_pass, 'returncode': proc.returncode, 't_elapsed': t_elapsed,
                'prob_stats': prob_stats, 'missing': missing}, f, indent=2)

print(f"\nReport 171 written! all_pass={all_pass}")
