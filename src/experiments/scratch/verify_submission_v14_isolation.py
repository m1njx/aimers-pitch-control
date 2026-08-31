"""
verify_submission_v14_isolation.py
11차 제출 후보(submit_v14.zip, GBDT+asof_dec + SimpleMLP blend w=0.32) 검증:
(1) 100%-격리 서브프로세스 실행(171/201/203 방식)
(2) 배치 vs 개별행 예측값 비교로 전체 파이프라인(GBDT+MLP 결합) 레벨 행독립성 재확인
결과 저장: outputs/207_submit_v14_isolation_check.md
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

zip_path = BASE_DIR / 'work/submit_v14.zip'
with zipfile.ZipFile(zip_path, 'r') as zf:
    zip_files = sorted(zf.namelist())

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py', 'agent2_asof_decomp2.py']
missing = [m for m in required_modules if m not in zip_files]

df_real_test = pd.read_csv(config.TEST_PATH)
clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}

work_root = Path('/tmp/v14_verify')
if work_root.exists():
    shutil.rmtree(work_root)
work_root.mkdir(parents=True)

# ---- 1. batch run (full-format isolation check, matches 171/201/203) ----
batch_dir = work_root / 'batch'
batch_dir.mkdir()
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(batch_dir)
(batch_dir / 'data').mkdir(exist_ok=True)
(batch_dir / 'output').mkdir(exist_ok=True)
df_real_test.to_csv(batch_dir / 'data/test.csv', index=False)

print(f"Running script.py in fully isolated subprocess (cwd={batch_dir}, PYTHONPATH removed)...")
t0 = time.time()
proc = subprocess.run([sys.executable, 'script.py'], cwd=str(batch_dir), env=clean_env,
                       capture_output=True, text=True, timeout=600)
t_elapsed = time.time() - t0
print(f"Return code: {proc.returncode}")
print(f"--- stdout ---\n{proc.stdout}")
if proc.returncode != 0:
    print(f"--- stderr ---\n{proc.stderr}")

success = proc.returncode == 0 and (batch_dir / 'output' / 'submission.csv').exists()
df_sub = None
sub_shape = sub_cols = None
prob_stats = {}
cols_match = row_ids_match = False
if success:
    df_sub = pd.read_csv(batch_dir / 'output' / 'submission.csv')
    sub_shape = list(df_sub.shape)
    sub_cols = list(df_sub.columns)
    prob_stats = dict(mean=float(df_sub['control_success'].mean()), std=float(df_sub['control_success'].std()),
                       min=float(df_sub['control_success'].min()), max=float(df_sub['control_success'].max()))
    cols_match = sub_cols == ['row_id', 'control_success']
    row_ids_match = set(df_sub['row_id']) == set(df_real_test['row_id'])

all_pass = success and not missing and cols_match and row_ids_match

# ---- 2. row-independence: each row alone vs in the batch ----
row_indep_rows = []
max_diff = 0.0
if success:
    df_batch_idx = df_sub.set_index('row_id')['control_success']
    for i, row_id in enumerate(df_real_test['row_id']):
        single_dir = work_root / f'single_{i}'
        shutil.copytree(batch_dir, single_dir, ignore=shutil.ignore_patterns('output', '__pycache__'))
        (single_dir / 'data').mkdir(exist_ok=True)
        (single_dir / 'output').mkdir(exist_ok=True)
        df_real_test.iloc[[i]].to_csv(single_dir / 'data/test.csv', index=False)
        p = subprocess.run([sys.executable, 'script.py'], cwd=str(single_dir), env=clean_env,
                            capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, f"single-row run failed for {row_id}:\n{p.stderr}"
        p_single = pd.read_csv(single_dir / 'output/submission.csv').set_index('row_id')['control_success'][row_id]
        p_batch = df_batch_idx[row_id]
        diff = abs(p_single - p_batch)
        max_diff = max(max_diff, diff)
        row_indep_rows.append((row_id, p_batch, p_single, diff))
        print(f"  {row_id}: batch={p_batch:.10f}  single={p_single:.10f}  diff={diff:.2e}")

row_indep_pass = max_diff < 1e-6

lines = [
    "# 207. 11차 제출 후보(submit_v14.zip, GBDT+asof_dec+SimpleMLP 블렌딩) 검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **구성**: v13의 GBDT 3종 5-seed(15/75/10, 행독립성 수정판) 모델을 그대로 재사용(재학습 없음) + SimpleMLP 5-seed(전체 train.csv로 신규 학습)를 고정가중치 w_mlp=0.32로 블렌딩."
    " w_mlp=0.32는 agent6의 inner-only(2022,2023) 선택값을 outer(2024)에 5-seed 정식 배깅으로 1회 확인(`outputs/205_mlp_5seed_outer_confirmed.md`, gain=+8.56, 노이즈폭±31.75 안이라 확정은 아니나 3-fold 전부 양의 방향)한 뒤 그대로 사용.",
    "- **검증 방법**: 171/201/203번과 동일한 100%-격리 서브프로세스(진짜 새 파이썬 프로세스, PYTHONPATH 제거, 완전히 새로운 임시 디렉토리) + 실제 test.csv(5행) 배치 실행. **추가로 5행 전부를 개별 단일-행 배치로도 각각 실행해 배치 예측값과 비교**해서 전체 파이프라인(GBDT+MLP 결합) 레벨 행독립성을 재확인함(MLP 단독으로는 205번에서 이미 확인했지만, 최종 배포 스크립트 레벨에서 한번 더 확인).\n",
    "---\n",
    "## 1. 로컬 모듈 포함 여부\n",
    f"- zip 내용물 ({len(zip_files)}개 파일, MLP 아티팩트 6개 추가로 v13보다 6개 많음): {zip_files}",
    f"- 필수 모듈 누락 여부: **`{missing if missing else '없음'}`**\n",
    "## 2. 격리 서브프로세스 실행 결과\n",
    "| 항목 | 값 |",
    "|:---|:---:|",
    f"| 종료 코드 | `{proc.returncode}` (0=정상) |",
    f"| 실행 시간 | `{t_elapsed:.2f}초` |",
    f"| 10분(600초) 제한 대비 여유 | `{(600-t_elapsed)/600*100:.1f}%` |",
    f"| submission.csv 생성 | `{success}` |",
    f"| 행/열 형식 | `{sub_shape}` (기대: `[5, 2]`) |",
    f"| 컬럼명 | `{sub_cols}` |",
    f"| row_id 일치 | `{row_ids_match}` |",
    f"| 예측 확률 분포 | 평균 `{prob_stats.get('mean', 'N/A')}`, 표준편차 `{prob_stats.get('std', 'N/A')}`, 범위 `[{prob_stats.get('min', 'N/A')}, {prob_stats.get('max', 'N/A')}]` |",
]
if df_sub is not None:
    lines.append(f"\n실제 예측값:\n```\n{df_sub.to_string()}\n```")

lines += ["\n---\n", "## 3. 전체 파이프라인 행독립성 재검증 (배치 vs 개별행)\n"]
lines.append("| row_id | 배치예측 | 단일행예측 | 차이 |")
lines.append("|---|---|---|---|")
for row_id, p_b, p_s, d in row_indep_rows:
    lines.append(f"| {row_id} | {p_b:.10f} | {p_s:.10f} | {d:.2e} |")
lines.append(f"\n**max_diff = {max_diff:.2e}** — {'✅ PASS(부동소수점 오차 수준, 행독립적으로 확인됨)' if row_indep_pass else '❌ FAIL(배치의존성 발견됨)'}")

lines += ["\n---\n", "## 4. 최종 판정\n"]
if all_pass and row_indep_pass:
    lines.append("> ## ✅ 11차 제출 후보 준비 완료 (v14)\n")
    lines.append(
        "> `work/submit_v14.zip`은 100%-격리 서브프로세스 정상 실행, 제출 형식 100% 일치, 10분 제한 대비 충분한 여유, "
        "그리고 **전체 파이프라인(GBDT+MLP) 레벨 행독립성까지 실측 통과**했습니다. "
        "이 프로젝트 최초의 DL 컴포넌트 포함 제출 후보이며, 로컬 outer(2024) 확인상 GBDT 단독 대비 +8.56점(노이즈폭 ±31.75 안, 확정 아님)입니다. "
        "실제 데이콘 업로드는 사용자의 명시적 지시가 있을 때 진행합니다."
    )
else:
    lines.append("> ## ❌ 검증 실패 — 제출 전 추가 조치 필요\n")
    if missing:
        lines.append(f"> - 누락된 모듈: {missing}")
    if not success:
        lines.append(f"> - 서브프로세스 실행 실패 (returncode={proc.returncode})")
        lines.append(f"> - stderr: ```\n{proc.stderr[-3000:]}\n```")
    if not row_indep_pass:
        lines.append(f"> - 행독립성 실패: max_diff={max_diff:.2e}")

with open(OUTPUTS_DIR / '207_submit_v14_isolation_check.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

with open('/tmp/v14_isolation_result.json', 'w') as f:
    json.dump({'all_pass': bool(all_pass and row_indep_pass), 'returncode': proc.returncode, 't_elapsed': t_elapsed,
               'prob_stats': prob_stats, 'missing': missing, 'row_indep_max_diff': max_diff}, f, indent=2)

print(f"\nReport 207 written! all_pass={all_pass and row_indep_pass}")
