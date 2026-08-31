#!/usr/bin/env python3
"""check_row_independence.py — 행 독립성 감사 (대회 무관, 범용).

**"각 행은 독립적으로 예측해야 한다"** 는 규정이 있는 대회에서, 파이프라인 전체가
정말 `p_i = f(row_i, 학습자산)` 인지 **행동으로** 검증한다. 정적 코드 리뷰보다 강하다 —
어떤 경로로 프레임 통계가 새든 결과 값이 달라지기 때문이다.

    python3 check_row_independence.py mysub.zip --data-dir ./data --probe ./data/train.csv --n 3000

4변형을 돌려 행별로 비교한다:
  FULL     프로브 N행 그대로
  SHUFFLE  행 순서만 뒤섞음      -> 순서 의존(rolling / shift / 정렬) 검출
  SUBSET   앞 절반만            -> 프레임 통계 의존(groupby / 평균 / 분위수) 검출
  SOLO     무작위 k행을 1행씩    -> 가장 강한 검사

판정: 최대 편차 < 1e-6 이면 통과. 부동소수점 잡음은 1e-9 수준, 진짜 누출은 1e-3 안팎이라
      명확히 갈린다. (2026 Aimers 실측: 통과한 파이프라인이 7.9e-09)

⚠️ SOLO 가 특정 행에서만 크래시하면 **모델 결함이 아니라 CSV dtype 추론** 문제일 수 있다.
   (실측 사례: 값이 전부 숫자인 문자열 카테고리 `'123'` 이 1행 프레임에서 int64 로 파싱되어
    OneHotEncoder 가 미지 카테고리로 처리) — 이 스크립트가 자동으로 그 후보를 지목한다.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, tempfile, zipfile

import numpy as np
import pandas as pd

ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE'}


def numeric_looking_object_cols(df: pd.DataFrame) -> list[str]:
    """CSV 왕복 시 dtype 이 바뀔 수 있는 컬럼 (전부 숫자로만 이뤄진 문자열 카테고리)."""
    out = []
    for c in df.columns:
        if df[c].dtype.kind != 'O':
            continue
        vals = df[c].dropna().astype(str).unique()
        if len(vals) and all(v.strip().lstrip('-').replace('.', '', 1).isdigit() for v in vals):
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--probe', required=True, help='실제 행이 든 CSV (test 와 같은 컬럼이면 됨)')
    ap.add_argument('--test-name', default='test.csv')
    ap.add_argument('--sample-name', default='sample_submission.csv')
    ap.add_argument('--entry', default='script.py')
    ap.add_argument('--out', default='output/submission.csv')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--solo', type=int, default=8)
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix='indep_')
    root = os.path.join(work, 'run')
    if zipfile.is_zipfile(a.submission):
        with zipfile.ZipFile(a.submission) as z:
            z.extractall(root)
    else:
        shutil.copytree(os.path.abspath(a.submission), root)
    dd = os.path.join(root, 'data'); os.makedirs(dd, exist_ok=True)

    ref = pd.read_csv(os.path.join(a.data_dir, a.test_name), encoding='utf-8-sig')
    smp = pd.read_csv(os.path.join(a.data_dir, a.sample_name), encoding='utf-8-sig')
    idc, pc = smp.columns[0], smp.columns[-1]
    cols = list(ref.columns)

    src = pd.read_csv(a.probe, encoding='utf-8-sig')
    miss = [c for c in cols if c not in src.columns]
    if miss:
        print(f'🔴 프로브에 없는 컬럼: {miss}'); return 1
    probe = src.sample(n=min(a.n, len(src)), random_state=20260830)[cols].reset_index(drop=True)
    risky = numeric_looking_object_cols(ref[[c for c in cols if c in ref.columns]])
    print(f'프로브 {len(probe):,}행 · id={idc} · pred={pc}')
    if risky:
        print(f'⚠️ dtype 왕복 주의 컬럼: {risky}  (SOLO 크래시 시 여기부터 의심)')

    def run(frame: pd.DataFrame) -> pd.Series:
        frame.to_csv(os.path.join(dd, a.test_name), index=False, encoding='utf-8')
        pd.DataFrame({idc: frame[idc], pc: 0.5}).to_csv(
            os.path.join(dd, a.sample_name), index=False, encoding='utf-8')
        r = subprocess.run([sys.executable, a.entry], cwd=root, env=ENV,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or '')[-1200:])
        o = pd.read_csv(os.path.join(root, a.out), encoding='utf-8-sig')
        return o.set_index(idc)[pc]

    print('\n[1/4] FULL', flush=True);    full = run(probe)
    rng = np.random.default_rng(a.seed)
    print('[2/4] SHUFFLE', flush=True)
    perm = probe.sample(frac=1.0, random_state=a.seed + 7).reset_index(drop=True)
    shuf = run(perm)
    print('[3/4] SUBSET', flush=True)
    half = probe.iloc[: len(probe) // 2].reset_index(drop=True)
    sub = run(half)
    print(f'[4/4] SOLO ({a.solo}행)', flush=True)
    solo, crashed = {}, []
    for i in rng.choice(len(probe), size=min(a.solo, len(probe)), replace=False):
        row = probe.iloc[[i]].reset_index(drop=True)
        try:
            solo[row[idc].iloc[0]] = run(row).iloc[0]
        except RuntimeError as e:
            crashed.append((int(i), str(e).strip().splitlines()[-1][:120]))

    print('\n' + '=' * 62)
    print(f'{"검사":<28}{"max|dp|":>14}{"판정":>10}')
    print('-' * 62)
    worst = 0.0
    for nm, s in (('SHUFFLE 불변성', shuf), ('SUBSET 불변성', sub)):
        k = full.index.intersection(s.index)
        d = float(np.abs(full.loc[k].to_numpy() - s.loc[k].to_numpy()).max())
        worst = max(worst, d)
        print(f'{nm + f" (n={len(k):,})":<28}{d:>14.3e}{"PASS" if d < a.atol else "FAIL":>10}')
    if solo:
        ds = max(abs(full.loc[k] - v) for k, v in solo.items())
        worst = max(worst, ds)
        print(f'{f"SOLO 불변성 (n={len(solo)})":<28}{ds:>14.3e}{"PASS" if ds < a.atol else "FAIL":>10}')
    print('=' * 62)
    print(f'\n최대 편차 {worst:.3e} -> ' +
          ('✅ 행 독립 (부동소수점 잡음 수준)' if worst < a.atol else
           '🔴 프레임 의존 존재 — 규정 위반 소지'))
    if crashed:
        print(f'\n⚠️ SOLO {len(crashed)}행 크래시:')
        for i, m in crashed[:3]:
            print(f'   row {i}: {m}')
        if risky:
            for c in risky:
                print(f'   -> dtype 후보 `{c}`: 크래시 행 값 = '
                      f'{[probe.iloc[i][c] for i, _ in crashed[:3]]}')
        print('   모델 결함이 아니라 CSV dtype 추론일 수 있다(위 주의 컬럼 참조).')
    shutil.rmtree(work, ignore_errors=True)
    return 0 if worst < a.atol else 1


if __name__ == '__main__':
    sys.exit(main())
