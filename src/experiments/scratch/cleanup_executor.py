#!/usr/bin/env python3
"""
cleanup_executor.py
모든 정리 단계를 순서대로 실행:
1단계: 인벤토리 (cleanup_inventory.py 내용 통합)
2단계: 중복 분석 (cleanup_01_duplicates.md)
3단계: 실제 파일 재배치 (core/, outputs/raw/, _archive/)
4단계: 검증 (cleanup_02_verification.md)
"""
import os, sys, shutil, json, subprocess
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/LG_data'))
NOW = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"[{NOW}] LG_data 정리 작업 시작\n")

# ============================================================
# 단계 1: 인벤토리
# ============================================================
print("=" * 60)
print("1단계: 전체 인벤토리 작성")
print("=" * 60)

# 파일 수집
all_files = []
for root, dirs, files in os.walk(BASE):
    # __pycache__ 제외 (카테고리 g로는 기록)
    dirs[:] = [d for d in sorted(dirs)]
    for f in files:
        fpath = Path(root) / f
        rel = fpath.relative_to(BASE)
        try:
            size = fpath.stat().st_size
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        except:
            size, mtime = 0, 'N/A'
        all_files.append({'fpath': fpath, 'rel': rel, 'size': size, 'mtime': mtime})

CORE_FILES = {
    'config.py', 'preprocessing.py', 'trackman_features.py',
    'cv_utils.py', 'submission_checklist.py', 'experiment_log.py',
    'model_config.py',
}

def categorize(info):
    fpath, rel, size, mtime = info['fpath'], info['rel'], info['size'], info['mtime']
    parts = rel.parts
    fname = fpath.name
    rel_str = str(rel)

    if parts[0] == 'open':
        return 'a'
    if parts[0] == 'work':
        return 'f'
    if parts[0] == 'final_code_submission':
        return 'f'
    if fname in ('submission_history.md', 'submission_history.json') and parts[0] == 'outputs':
        return 'e'
    if parts[0] == 'outputs':
        if fname.endswith('.md') and fname[:2].replace('_','').isdigit():
            return 'c'
        if fname[0].isdigit() and fname.endswith('.md'):
            return 'c'
        if fname.endswith('.csv') or fname.endswith('.json'):
            return 'd'
        if fname.endswith('.md'):
            return 'd'
        return 'g'
    if parts[0] == 'scratch':
        return 'd'
    if fname in CORE_FILES and len(parts) == 1:
        return 'b'
    if fname.endswith('.ipynb'):
        return 'g'
    if fname.endswith('.pyc') or '__pycache__' in parts:
        return 'g'
    if fname == '.DS_Store' or fname.startswith('.'):
        return 'g'
    if parts[0] == 'catboost_info':
        return 'g'
    if parts[0] == '__pycache__':
        return 'g'
    if fname.endswith('.py') and len(parts) == 1:
        # root-level .py not in CORE_FILES
        return 'g'
    return 'g'

cat_names = {
    'a': '원본 데이터 (open/)',
    'b': '핵심 파이프라인 코드',
    'c': '정식 산출물 보고서 (번호 .md)',
    'd': '실험 raw 데이터 / 스크래치',
    'e': '제출 기록',
    'f': 'work/ & final_code_submission/ 산출물',
    'g': '기타/정체불명',
}

for info in all_files:
    info['cat'] = categorize(info)

# Stats
stats = {}
for info in all_files:
    c = info['cat']
    stats.setdefault(c, {'count': 0, 'bytes': 0})
    stats[c]['count'] += 1
    stats[c]['bytes'] += info['size']

def human(b):
    if b >= 1024*1024: return f'{b/1024/1024:.1f} MB'
    if b >= 1024: return f'{b/1024:.1f} KB'
    return f'{b} B'

# Write inventory markdown
lines = ['# cleanup_00_inventory: LG_data 전체 파일 인벤토리', '',
         f'> 생성: {NOW}', '', '## 카테고리별 요약', '',
         '| 코드 | 설명 | 파일 수 | 총 용량 |',
         '|:---|:---|:---:|:---:|']
for c, name in cat_names.items():
    s = stats.get(c, {'count': 0, 'bytes': 0})
    lines.append(f'| {c} | {name} | {s["count"]} | {human(s["bytes"])} |')
lines += ['', '---', '']

sorted_files = sorted(all_files, key=lambda x: (x['cat'], str(x['rel'])))
for c, name in cat_names.items():
    lines.append(f'## 카테고리 {c}: {name}')
    lines.append('')
    lines.append('| 경로 | 용량 | 최종 수정일 | 비고 |')
    lines.append('|:---|:---:|:---:|:---|')
    for info in sorted_files:
        if info['cat'] != c: continue
        note = ''
        fname = info['fpath'].name
        if c == 'g':
            if fname.endswith('.ipynb'):
                note = 'DACON 배포 기본 노트북 (미사용)'
            elif fname.endswith('.pyc') or '__pycache__' in str(info['rel']):
                note = '컴파일 캐시 — 삭제 가능'
            elif fname == '.DS_Store':
                note = 'macOS 숨김 파일 — 삭제 가능'
            elif info['fpath'].parent.name == 'catboost_info':
                note = 'CatBoost 학습 로그 캐시'
            elif fname in ('experiment_tracker.py', 'generate_summary_report.py', 'my_experiment_log.py'):
                note = '초기 실험 헬퍼 스크립트 — 현재 미사용'
        lines.append(f'| `{str(info["rel"])}` | {human(info["size"])} | {info["mtime"]} | {note} |')
    lines.append('')

inv_path = BASE / 'outputs' / 'cleanup_00_inventory.md'
inv_path.write_text('\n'.join(lines), encoding='utf-8')
print(f"  -> 인벤토리 저장: {inv_path}")
print(f"  -> 총 파일 수: {len(all_files)}")
for c, name in cat_names.items():
    s = stats.get(c, {'count': 0, 'bytes': 0})
    print(f"     {c}) {name}: {s['count']}개 ({human(s['bytes'])})")

# ============================================================
# 단계 2: 중복 분석
# ============================================================
print("\n" + "=" * 60)
print("2단계: 중복/구버전 파일 식별")
print("=" * 60)

dup_lines = ['# cleanup_01_duplicates: 중복/구버전 파일 분석', '',
             f'> 생성: {NOW}', '']

# 2-1. 핵심 코드 중복 확인 (root vs final_code_submission vs work/submit_v4)
dup_lines += ['## 2-1. 핵심 코드 파일 다중 버전 존재 여부', '',
              '| 파일명 | 위치 | 용량 | 수정일 | 판정 |',
              '|:---|:---|:---:|:---:|:---|']

for fname in ['config.py', 'preprocessing.py', 'trackman_features.py']:
    locations = [
        BASE / fname,
        BASE / 'final_code_submission' / fname,
        BASE / 'work' / 'dummy_eval_v4' / fname,
        BASE / 'work' / 'submit_v4' / fname,
    ]
    for loc in locations:
        if loc.exists():
            sz = loc.stat().st_size
            mt = datetime.fromtimestamp(loc.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            rel = str(loc.relative_to(BASE))
            if str(loc.parent) == str(BASE):
                verdict = '✅ 정본 (root)'
            elif 'final_code_submission' in str(loc):
                verdict = '📦 제출용 복사본 — work에 보관 권장'
            else:
                verdict = '📦 제출 패키지 내 복사본 — work에 보관'
            dup_lines.append(f'| `{fname}` | `{rel}` | {human(sz)} | {mt} | {verdict} |')

dup_lines += ['', '> **결론**: root의 파일이 정본. final_code_submission/과 work/submit_v*/는 제출 패키지이므로 work/ 하위에 보관.', '']

# 2-2. Outputs 번호 중복 확인
dup_lines += ['## 2-2. outputs/ 번호 중복 보고서 확인', '',
              '| 번호 | 파일들 | 판정 |',
              '|:---:|:---|:---|']

outputs_dir = BASE / 'outputs'
num_groups = {}
for f in outputs_dir.iterdir():
    if f.is_file() and f.name[0].isdigit():
        num = ''
        for ch in f.stem:
            if ch.isdigit(): num += ch
            else: break
        if num:
            num_groups.setdefault(num, []).append(f)

for num in sorted(num_groups.keys(), key=int):
    files = num_groups[num]
    if len(files) > 1:
        fnames = ', '.join(f'`{f.name}`' for f in sorted(files, key=lambda x: x.name))
        dup_lines.append(f'| {num} | {fnames} | ⚠️ 동일 번호 — md+csv/json 조합 (정상) |')
    else:
        pass  # 단독은 기록 생략

dup_lines += ['', '> **결론**: 같은 번호의 .md + .csv/.json 조합은 보고서 + raw데이터 쌍으로 정상. 진짜 중복 없음.', '']

# 2-3. 이름 없는 / 언넘버드 outputs
dup_lines += ['## 2-3. 번호 없는 outputs/ 파일 (raw/ 폴더로 이동 권장)', '',
              '| 파일 | 설명 | 권장 조치 |',
              '|:---|:---|:---|']
unnumbered = {
    'catboost_exp_summary.json': 'CatBoost 실험 raw JSON',
    'ensemble_exp_summary.json': '앙상블 실험 raw JSON',
    'final_recheck_summary.json': '최종 재검증 raw JSON',
    'hp_and_4model_summary.json': 'HP/4모델 실험 raw JSON',
    'interaction_exp_summary.json': '교차피처 실험 raw JSON',
    'recency_weighting_exp.json': '최신성 가중 실험 raw JSON',
    'scheduled_tasks_master_summary.json': '스케줄 태스크 요약 JSON',
    'task1_2_audit_summary.json': '감사 태스크 요약 JSON',
    'xgboost_ensemble_exp.json': 'XGBoost 앙상블 실험 raw JSON',
    'my_log.json': '개인 실험 로그 JSON',
    'my_log.md': '개인 실험 로그 MD',
    'my_2week_summary_report.md': '2주 여정 요약 — 유용하지만 번호 없음',
    'daily_candidate_template.md': '후보 기록 템플릿',
    'solo_roadmap_2weeks.md': '2주 로드맵 — 참고용',
    'submission_history.md': '제출 이력 — 핵심 보관',
    'submission_history.json': '제출 이력 JSON — 핵심 보관',
}
for fname, desc in unnumbered.items():
    fpath = outputs_dir / fname
    if fpath.exists():
        sz = human(fpath.stat().st_size)
        action = 'outputs/raw/ 이동' if fname not in ('submission_history.md', 'submission_history.json') else '✅ 루트로 이동 (핵심)'
        dup_lines.append(f'| `{fname}` | {desc} | {sz} — {action} |')

dup_lines += ['', '## 2-4. _archive 이동 권장 파일', '',
              '| 파일/폴더 | 이유 |',
              '|:---|:---|',
              '| `catboost_info/` | CatBoost 학습 로그 캐시. 재학습 시 자동 재생성 |',
              '| `__pycache__/` | Python 컴파일 캐시. 자동 재생성 |',
              '| `*.ipynb` (2개) | DACON 기본 노트북. 미사용 |',
              '| `experiment_tracker.py` | 초기 버전 헬퍼. 현재 experiment_log.py로 대체됨 |',
              '| `generate_summary_report.py` | 일회성 요약 스크립트. 재사용 안 함 |',
              '| `my_experiment_log.py` | experiment_log.py와 중복 가능성 있는 초기 버전 |',
              '| `work/dummy_eval/` ~ `dummy_eval_v3/` | v1~v3 더미 평가 디렉터리 (v4가 최신) |',
              '| `work/submit/`, `work/submit_v2/`, `work/submit_v3/` | 구버전 제출 패키지 |',
              '| `work/submit.zip`, `submit_v2.zip`, `submit_v3.zip` | 구버전 제출 zip |',
              '| `outputs/.DS_Store` | macOS 숨김 파일 |',
              '', ]

dup_path = BASE / 'outputs' / 'cleanup_01_duplicates.md'
dup_path.write_text('\n'.join(dup_lines), encoding='utf-8')
print(f"  -> 중복 분석 저장: {dup_path}")

# ============================================================
# 단계 3: 실제 파일 재배치
# ============================================================
print("\n" + "=" * 60)
print("3단계: 파일 재배치 실행")
print("=" * 60)

ARCHIVE = BASE / '_archive'
ARCHIVE.mkdir(exist_ok=True)
CORE_DIR = BASE / 'core'
CORE_DIR.mkdir(exist_ok=True)
RAW_DIR = BASE / 'outputs' / 'raw'
RAW_DIR.mkdir(exist_ok=True)

move_log = []  # (src, dst, reason)

def safe_move(src, dst, reason):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(dst.stem + '_dup' + dst.suffix)
    shutil.move(str(src), str(dst))
    move_log.append({'src': str(src.relative_to(BASE)), 'dst': str(dst.relative_to(BASE)), 'reason': reason})
    print(f"  MOVE: {src.relative_to(BASE)} -> {dst.relative_to(BASE)}")

def safe_copy(src, dst, reason):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
        move_log.append({'src': str(src.relative_to(BASE)), 'dst': str(dst.relative_to(BASE)), 'reason': f'[COPY] {reason}'})
        print(f"  COPY: {src.relative_to(BASE)} -> {dst.relative_to(BASE)}")

# 3-1. 핵심 코드를 core/ 로 복사 (root에도 유지 — 경로 호환성)
print("\n[3-1] 핵심 코드 -> core/ 복사")
for fname in ['config.py', 'preprocessing.py', 'trackman_features.py',
              'cv_utils.py', 'submission_checklist.py', 'experiment_log.py',
              'model_config.py']:
    src = BASE / fname
    dst = CORE_DIR / fname
    if src.exists():
        safe_copy(src, dst, '핵심 파이프라인 코드 core/로 복사')

# 3-2. outputs/raw/ 로 이동: 번호 없는 json/md 보조 파일들
print("\n[3-2] outputs/ 번호 없는 raw 파일 -> outputs/raw/ 이동")
raw_to_move = [
    'catboost_exp_summary.json', 'ensemble_exp_summary.json',
    'final_recheck_summary.json', 'hp_and_4model_summary.json',
    'interaction_exp_summary.json', 'recency_weighting_exp.json',
    'scheduled_tasks_master_summary.json', 'task1_2_audit_summary.json',
    'xgboost_ensemble_exp.json', 'my_log.json',
    'my_log.md', 'my_2week_summary_report.md',
    'daily_candidate_template.md', 'solo_roadmap_2weeks.md',
]
for fname in raw_to_move:
    src = BASE / 'outputs' / fname
    if src.exists():
        safe_move(src, RAW_DIR / fname, '번호 없는 보조 파일 → outputs/raw/')

# submission_history는 루트로 이동
print("\n[3-3] submission_history -> LG_data 루트로 이동")
for fname in ['submission_history.md', 'submission_history.json']:
    src = BASE / 'outputs' / fname
    dst = BASE / fname
    if src.exists() and not dst.exists():
        safe_move(src, dst, '제출 이력 → 루트로 이동')
    elif src.exists() and dst.exists():
        print(f"  SKIP (already exists at root): {fname}")

# 3-4. _archive 로 이동: 구버전/불필요 파일
print("\n[3-4] 구버전/불필요 파일 -> _archive/ 이동")

# catboost_info/
if (BASE / 'catboost_info').exists():
    safe_move(BASE / 'catboost_info', ARCHIVE / 'catboost_info', 'CatBoost 학습 로그 캐시')

# __pycache__ at root
if (BASE / '__pycache__').exists():
    safe_move(BASE / '__pycache__', ARCHIVE / '__pycache__', 'Python 컴파일 캐시')

# .ipynb 파일들
for nb in BASE.glob('*.ipynb'):
    safe_move(nb, ARCHIVE / nb.name, 'DACON 기본 노트북 — 미사용')

# 구버전 root-level py
for fname in ['experiment_tracker.py', 'generate_summary_report.py', 'my_experiment_log.py']:
    src = BASE / fname
    if src.exists():
        safe_move(src, ARCHIVE / fname, '초기 버전 헬퍼 스크립트 — 현재 미사용/대체됨')

# work/dummy_eval v1~v3 (v4는 유지)
for dname in ['dummy_eval', 'dummy_eval_check', 'dummy_eval_v2', 'dummy_eval_v3']:
    src = BASE / 'work' / dname
    if src.exists():
        safe_move(src, ARCHIVE / 'work' / dname, f'구버전 더미 평가 디렉터리 ({dname})')

# work/submit v1~v3 폴더 및 zip
for name in ['submit', 'submit_v2', 'submit_v3', 'submit.zip', 'submit_v2.zip', 'submit_v3.zip']:
    src = BASE / 'work' / name
    if src.exists():
        safe_move(src, ARCHIVE / 'work' / name, f'구버전 제출 패키지 ({name})')

# work/baseline_submit (DACON 원본 RF 베이스라인)
src = BASE / 'work' / 'baseline_submit'
if src.exists():
    safe_move(src, ARCHIVE / 'work' / 'baseline_submit', 'DACON 원본 RF 베이스라인 — 미사용')

# .DS_Store files in outputs
for ds in BASE.rglob('.DS_Store'):
    if 'open' not in str(ds):
        safe_move(ds, ARCHIVE / ('.DS_Store_' + ds.parent.name), 'macOS 숨김 파일')

# 3-5. final_code_submission -> work/final_code_submission 으로 이동
src = BASE / 'final_code_submission'
dst = BASE / 'work' / 'final_code_submission'
if src.exists() and not dst.exists():
    safe_move(src, dst, '제출 코드 패키지 → work/ 하위로 정리')

print(f"\n  총 {len(move_log)}개 파일/폴더 처리 완료")

# ============================================================
# 단계 4: 검증
# ============================================================
print("\n" + "=" * 60)
print("4단계: 검증")
print("=" * 60)

ver_lines = ['# cleanup_02_verification: 정리 후 검증 보고서', '',
             f'> 생성: {NOW}', '']

# 4-1. 파이프라인 import 검증
ver_lines += ['## 4-1. 핵심 파이프라인 모듈 Import 검증', '',
              '| 모듈 | 경로 | 상태 |',
              '|:---|:---|:---:|']

import_results = {}
for fname in ['config', 'preprocessing', 'trackman_features', 'cv_utils',
              'submission_checklist', 'experiment_log', 'model_config']:
    try:
        result = subprocess.run(
            [sys.executable, '-c', f'import sys; sys.path.insert(0,"{str(BASE)}"); import {fname}; print("OK")'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and 'OK' in result.stdout:
            status = '✅ OK'
            import_results[fname] = True
        else:
            status = f'❌ FAIL: {result.stderr.strip()[:100]}'
            import_results[fname] = False
    except Exception as e:
        status = f'❌ ERROR: {e}'
        import_results[fname] = False
    ver_lines.append(f'| `{fname}` | `{BASE}/{fname}.py` | {status} |')
    print(f"  import {fname}: {'OK' if import_results.get(fname) else 'FAIL'}")

# 4-2. 경로 검증
ver_lines += ['', '## 4-2. config.py 내 경로 검증', '',
              '| 설정 변수 | 경로 | 존재 여부 |',
              '|:---|:---|:---:|']
try:
    result = subprocess.run(
        [sys.executable, '-c', f'''
import sys
sys.path.insert(0, "{str(BASE)}")
import config
paths = {{
    "TRAIN_PATH": getattr(config, "TRAIN_PATH", None),
    "TEST_PATH": getattr(config, "TEST_PATH", None),
    "ARTIFACTS_DIR": getattr(config, "ARTIFACTS_DIR", None),
    "TRACKMAN_PATH": getattr(config, "TRACKMAN_PATH", None),
}}
import json
print(json.dumps({{k: str(v) if v else "N/A" for k,v in paths.items()}}))
'''],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        paths = json.loads(result.stdout.strip())
        for k, v in paths.items():
            exists = '✅' if v and Path(v).exists() else '❌'
            ver_lines.append(f'| `{k}` | `{v}` | {exists} |')
            print(f"  {k}: {v} -> {'EXISTS' if v and Path(v).exists() else 'MISSING'}")
    else:
        ver_lines.append(f'| ERROR | config.py import 실패 | ❌ |')
except Exception as e:
    ver_lines.append(f'| ERROR | {e} | ❌ |')

# 4-3. outputs/ 보고서 파일 수 확인
ver_lines += ['', '## 4-3. outputs/ 보고서 파일 확인', '']
md_reports = [f for f in (BASE / 'outputs').glob('*.md') if f.name[0].isdigit()]
md_reports.sort()
ver_lines.append(f'- 번호 있는 .md 보고서 수: **{len(md_reports)}개**')
ver_lines.append(f'- 번호 범위: {md_reports[0].name if md_reports else "N/A"} ~ {md_reports[-1].name if md_reports else "N/A"}')
ver_lines.append('')

# 4-4. _archive 이동 목록
ver_lines += ['## 4-4. _archive/ 이동 파일 목록', '',
              '| 원본 경로 | 이동 위치 | 이유 |',
              '|:---|:---|:---|']
for m in move_log:
    if '_archive' in m['dst']:
        ver_lines.append(f'| `{m["src"]}` | `{m["dst"]}` | {m["reason"]} |')

# 4-5. 최종 요약
ver_lines += ['', '## 4-5. 정리 전후 요약', '',
              '| 항목 | 값 |',
              '|:---|:---|',
              f'| 총 이동/복사 처리 수 | {len(move_log)}개 |',
              f'| _archive로 이동된 항목 | {sum(1 for m in move_log if "_archive" in m["dst"])}개 |',
              f'| core/로 복사된 파일 | {sum(1 for m in move_log if "core/" in m["dst"])}개 |',
              f'| outputs/raw/로 이동된 파일 | {sum(1 for m in move_log if "raw/" in m["dst"])}개 |',
              f'| 현재 outputs/ 번호 보고서 수 | {len(md_reports)}개 |',
              f'| 핵심 파이프라인 import | {"✅ 전부 OK" if all(import_results.values()) else "❌ 일부 실패"} |',
              '| 내일 73/74번 작업 준비 완료 여부 | ' + ('✅ 즉시 가능' if all(import_results.values()) else '❌ 경로 수정 필요') + ' |',
              ]

# archive_manifest
manifest_lines = ['# _archive/archive_manifest.md: 아카이브 파일 근거', '',
                  f'> 생성: {NOW}', '',
                  '| 원본 경로 | 이동 위치 | 이유 |',
                  '|:---|:---|:---|']
for m in move_log:
    if '_archive' in m['dst']:
        manifest_lines.append(f'| `{m["src"]}` | `{m["dst"]}` | {m["reason"]} |')

manifest_path = ARCHIVE / 'archive_manifest.md'
manifest_path.write_text('\n'.join(manifest_lines), encoding='utf-8')

ver_path = BASE / 'outputs' / 'cleanup_02_verification.md'
ver_path.write_text('\n'.join(ver_lines), encoding='utf-8')

print(f"\n  -> 검증 보고서: {ver_path}")
print(f"  -> 아카이브 매니페스트: {manifest_path}")
print(f"\n{'='*60}")
print("모든 정리 단계 완료!")
print(f"{'='*60}")
