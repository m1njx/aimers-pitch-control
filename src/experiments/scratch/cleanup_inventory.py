#!/usr/bin/env python3
"""
cleanup_inventory.py
LG_data 전체 파일 인벤토리 작성 및 카테고리 분류 스크립트
"""
import os, json
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/LG_data'))

categories = {
    'a': '원본 데이터 (open/)',
    'b': '핵심 파이프라인 코드',
    'c': '정식 산출물 보고서 (번호 .md)',
    'd': '실험 raw 데이터 (.csv/.json 보조)',
    'e': '제출 기록',
    'f': 'work/ 산출물',
    'g': '기타/정체불명',
}

CORE_FILES = {
    'config.py', 'preprocessing.py', 'trackman_features.py',
    'cv_utils.py', 'submission_checklist.py', 'experiment_log.py',
    'model_config.py',
}

SUBMISSION_FILES = {
    'submission_history.md', 'submission_history.json',
}

rows = []

for root, dirs, files in os.walk(BASE):
    # skip open/ internals but record them
    for f in files:
        fpath = Path(root) / f
        rel = fpath.relative_to(BASE)
        parts = rel.parts

        try:
            size = fpath.stat().st_size
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        except Exception:
            size, mtime = 0, 'N/A'

        # Categorize
        rel_str = str(rel)
        fname = f

        if parts[0] == 'open':
            cat = 'a'
        elif parts[0] == 'work':
            cat = 'f'
        elif fname in CORE_FILES and parts[0] not in ('work', 'open', 'final_code_submission', 'scratch'):
            cat = 'b'
        elif fname in SUBMISSION_FILES:
            cat = 'e'
        elif parts[0] == 'outputs':
            # numbered md reports
            stem = Path(fname).stem
            if fname.endswith('.md') and stem[:2].isdigit():
                cat = 'c'
            elif fname.endswith('.csv') or fname.endswith('.json'):
                cat = 'd'
            elif fname.endswith('.md'):
                cat = 'd'  # unnumbered mds = raw/misc
            else:
                cat = 'g'
        elif parts[0] == 'scratch':
            cat = 'd'  # experiment scripts = raw/misc
        elif fname.endswith('.ipynb'):
            cat = 'g'
        elif fname.endswith('.py') and parts[0] not in ('open',):
            if fname in CORE_FILES:
                cat = 'b'
            else:
                cat = 'g'
        elif fname == '.DS_Store' or fname.startswith('.'):
            cat = 'g'
        elif parts[0] == '__pycache__' or fname.endswith('.pyc'):
            cat = 'g'
        elif parts[0] == 'catboost_info':
            cat = 'g'
        elif parts[0] == 'final_code_submission':
            cat = 'f'
        else:
            cat = 'g'

        rows.append({
            'path': rel_str,
            'size_bytes': size,
            'size_human': f'{size/1024:.1f} KB' if size < 1024*1024 else f'{size/1024/1024:.1f} MB',
            'mtime': mtime,
            'cat': cat,
            'cat_name': categories[cat],
        })

# Sort by category then path
rows.sort(key=lambda r: (r['cat'], r['path']))

# Stats per category
stats = {}
for row in rows:
    c = row['cat']
    stats.setdefault(c, {'count': 0, 'total_bytes': 0})
    stats[c]['count'] += 1
    stats[c]['total_bytes'] += row['size_bytes']

# Write markdown
lines = ['# LG_data 전체 파일 인벤토리', '',
         f'> 생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '',
         '## 카테고리별 요약', '',
         '| 카테고리 | 설명 | 파일 수 | 총 용량 |',
         '|:---|:---|:---:|:---:|']
for c, name in categories.items():
    s = stats.get(c, {'count': 0, 'total_bytes': 0})
    tb = s['total_bytes']
    human = f'{tb/1024/1024:.1f} MB' if tb >= 1024*1024 else f'{tb/1024:.1f} KB'
    lines.append(f'| {c} | {name} | {s["count"]} | {human} |')

lines += ['', '---', '']
for c, name in categories.items():
    lines.append(f'## 카테고리 {c}: {name}')
    lines.append('')
    lines.append('| 경로 | 용량 | 최종 수정일 |')
    lines.append('|:---|:---:|:---:|')
    cat_rows = [r for r in rows if r['cat'] == c]
    for r in cat_rows:
        lines.append(f'| `{r["path"]}` | {r["size_human"]} | {r["mtime"]} |')
    lines.append('')

out_path = BASE / 'outputs' / 'cleanup_00_inventory.md'
out_path.write_text('\n'.join(lines), encoding='utf-8')
print(f"Inventory written: {out_path}")
print(f"Total files: {len(rows)}")
for c, name in categories.items():
    s = stats.get(c, {'count': 0, 'total_bytes': 0})
    print(f"  {c}) {name}: {s['count']} files")
