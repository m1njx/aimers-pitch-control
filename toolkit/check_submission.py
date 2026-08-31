#!/usr/bin/env python3
"""check_submission.py — 제출 zip 0점 방지 자동 점검 (대회 무관, 범용).

2026 Aimers 대회에서 실제로 겪은 사고를 전부 자동 검사로 바꾼 것.
각 검사 옆의 [사고] 는 그 검사가 없어서 실제로 잃은 것이다.

    python3 check_submission.py mysub.zip --data-dir ./data
    python3 check_submission.py mysub.zip --data-dir ./data --require test.csv sample_submission.csv
    python3 check_submission.py mysub/ --data-dir ./data --entry script.py --timeout 600

검사 항목
  1 zip 위생        __pycache__ / .DS_Store / 학습데이터 동봉        [용량·정보 유출]
  2 진입점 구조     entry 스크립트가 최상위에 있는가
  3 구문            모든 .py 에 ast.parse                            [사고: SyntaxError 로 슬롯 소실]
  4 **새로 푼** 실행  zip 을 새 폴더에 풀어 필수 데이터만 두고 실행    [사고: ModuleNotFoundError]
  5 출력 유효성     행수 일치 · 범위 · NaN · 중복 id
  6 행 정렬         출력 id 순서가 sample_submission 과 같은가        [사고: 위치대입으로 100% 오정렬]
  7 결정성          두 번 실행해 바이트 동일                          [재현 불가 방지]
  8 시간            제한 대비 여유

종료 코드 0 = 전부 통과. 하나라도 실패하면 1.
"""
from __future__ import annotations
import argparse, ast, hashlib, os, shutil, subprocess, sys, tempfile, time, zipfile

FORBIDDEN_NAMES = ('__pycache__', '.DS_Store', '.ipynb_checkpoints', '.git')
BIG_DATA_HINTS = ('train.csv', 'train.parquet')
OK, BAD = '  \033[32mPASS\033[0m', '  \033[31mFAIL\033[0m'
results: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, msg: str = '') -> bool:
    results.append((name, ok, msg))
    print(f'{OK if ok else BAD}  {name}' + (f'  — {msg}' if msg else ''), flush=True)
    return ok


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission', help='제출 .zip 또는 폴더')
    ap.add_argument('--data-dir', required=True, help='test/sample 파일이 있는 폴더')
    ap.add_argument('--require', nargs='*', default=['test.csv', 'sample_submission.csv'],
                    help='채점 서버가 제공하는 파일 (이것만 복사해 실행한다)')
    ap.add_argument('--entry', default='script.py')
    ap.add_argument('--out', default='output/submission.csv')
    ap.add_argument('--id-col', default=None, help='기본: sample_submission 의 첫 컬럼')
    ap.add_argument('--pred-col', default=None, help='기본: sample_submission 의 마지막 컬럼')
    ap.add_argument('--timeout', type=int, default=600, help='채점 서버 제한(초)')
    ap.add_argument('--lo', type=float, default=0.0)
    ap.add_argument('--hi', type=float, default=1.0)
    ap.add_argument('--keep', action='store_true', help='작업 폴더 보존')
    a = ap.parse_args()

    import pandas as pd, numpy as np

    src = os.path.abspath(a.submission)
    work = tempfile.mkdtemp(prefix='subcheck_')
    root = os.path.join(work, 'run')
    print(f'대상 : {src}\n작업 : {work}\n' + '─' * 64)

    # ---- 1 zip 위생 + 4 새로 풀기 ----
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            hits = [n for n in names if any(f in n for f in FORBIDDEN_NAMES)]
            big = [n for n in names if os.path.basename(n) in BIG_DATA_HINTS]
            rec('1 zip 위생', not hits and not big,
                f'{len(hits)}개 잡파일' + (f', 학습데이터 {big}' if big else '') if (hits or big) else
                f'{len(names)}개 항목, 잡파일 없음')
            z.extractall(root)
    else:
        shutil.copytree(src, root)
        rec('1 zip 위생', True, '폴더 입력 — 건너뜀')

    # ---- 2 진입점 ----
    entry = os.path.join(root, a.entry)
    rec('2 진입점 구조', os.path.isfile(entry), f'{a.entry} ' + ('있음' if os.path.isfile(entry) else '없음(최상위)'))
    if not os.path.isfile(entry):
        return finish(work, a.keep)

    # ---- 3 구문 ----
    bad = []
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith('.py'):
                p = os.path.join(dp, fn)
                try:
                    ast.parse(open(p, encoding='utf-8', errors='replace').read())
                except SyntaxError as e:
                    bad.append(f'{os.path.relpath(p, root)}:{e.lineno}')
    rec('3 구문 (ast.parse)', not bad, ', '.join(bad[:3]) if bad else f'{a.entry} 포함 전체 통과')

    # ---- 4 실행 (필수 데이터만) ----
    dd = os.path.join(root, 'data'); os.makedirs(dd, exist_ok=True)
    missing = []
    for fn in a.require:
        s = os.path.join(os.path.abspath(a.data_dir), fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(dd, fn))
        else:
            missing.append(fn)
    if missing:
        rec('4 새로 푼 실행', False, f'입력 파일 없음: {missing}')
        return finish(work, a.keep)

    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE'}
    t0 = time.time()
    r = subprocess.run([sys.executable, a.entry], cwd=root, env=env,
                       capture_output=True, text=True)
    el = time.time() - t0
    if r.returncode != 0:
        print((r.stdout or '')[-1500:]); print((r.stderr or '')[-2500:])
        rec('4 새로 푼 실행', False, f'종료코드 {r.returncode}')
        return finish(work, a.keep)
    rec('4 새로 푼 실행', True, f'{el:.1f}초')

    outp = os.path.join(root, a.out)
    if not os.path.isfile(outp):
        rec('5 출력 유효성', False, f'{a.out} 없음')
        return finish(work, a.keep)

    # ---- 5 출력 유효성 ----
    sub = pd.read_csv(os.path.join(dd, a.require[-1]), encoding='utf-8-sig')
    out = pd.read_csv(outp, encoding='utf-8-sig')
    idc = a.id_col or sub.columns[0]
    pc = a.pred_col or sub.columns[-1]
    v = pd.to_numeric(out[pc], errors='coerce') if pc in out.columns else pd.Series(dtype=float)
    probs = [
        (len(out) != len(sub), f'행수 {len(out)} != {len(sub)}'),
        (pc not in out.columns, f'예측 컬럼 "{pc}" 없음'),
        (idc not in out.columns, f'id 컬럼 "{idc}" 없음'),
        (v.isna().any(), f'NaN {int(v.isna().sum())}개'),
        (len(v) and (v.min() < a.lo or v.max() > a.hi), f'범위 이탈 [{v.min():.4g}, {v.max():.4g}]'),
        (idc in out.columns and out[idc].duplicated().any(), 'id 중복'),
    ]
    fails = [m for c, m in probs if c]
    rec('5 출력 유효성', not fails,
        '; '.join(fails) if fails else f'{len(out):,}행, [{v.min():.4g}, {v.max():.4g}]')

    # ---- 6 행 정렬 ----
    if idc in out.columns and len(out) == len(sub):
        same = out[idc].astype(str).tolist() == sub[idc].astype(str).tolist()
        setsame = set(out[idc].astype(str)) == set(sub[idc].astype(str))
        rec('6 행 정렬', same or setsame,
            'sample_submission 과 순서 동일' if same else
            ('순서는 다르나 id 집합 동일 — 채점 시 조인되면 OK, 위치대입 코드가 있으면 위험' if setsame
             else 'id 집합이 다르다'))

    # ---- 7 결정성 ----
    h1 = sha(outp)
    r2 = subprocess.run([sys.executable, a.entry], cwd=root, env=env, capture_output=True, text=True)
    h2 = sha(outp) if r2.returncode == 0 and os.path.isfile(outp) else 'ERR'
    rec('7 결정성', h1 == h2, f'{h1[:12]} vs {str(h2)[:12]}')

    # ---- 8 시간 ----
    n_out, n_real = len(out), None
    rec('8 시간 여유', el < a.timeout * 0.5,
        f'{el:.1f}초 / 제한 {a.timeout}초 (실제 test 가 더 크면 비례 증가 — {n_out:,}행 기준)')

    print('─' * 64)
    nf = sum(1 for _, ok, _ in results if not ok)
    print(f'{"✅ 전부 통과 — 제출 가능" if nf == 0 else f"🔴 {nf}건 실패 — 제출 금지"}')
    return finish(work, a.keep, 0 if nf == 0 else 1)


def finish(work, keep, code=1):
    if keep:
        print(f'작업 폴더 보존: {work}')
    else:
        shutil.rmtree(work, ignore_errors=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
