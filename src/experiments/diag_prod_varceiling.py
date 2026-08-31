#!/usr/bin/env python3
"""diag_prod_varceiling.py — 증류 이득의 정체(분산 수축)가 프로덕션에 남아 있는가.

diag_distill_covvar.py 결과: student 이득 +85.2 의 내역은
    분산 항 +165.3 / 정보 항 -83.4 / bias 항 +3.3
즉 '약한 단일 LGB 의 과대 분산을 눌러준 것' 이고 정보는 오히려 줄었다.

그렇다면 남는 질문은 하나다 — 프로덕션 예측에도 아직 누를 분산이 남아 있는가?
프로덕션 블렌드 예측에 검증 라벨로 오라클 아핀(a*p+b)을 적합해서, '분산·bias 축'
전체에 남은 여지의 상한을 잰다. 오라클이므로 실제로 얻을 수 있는 값보다 낙관적이다.
이 상한이 LB 노이즈 바닥(±12) 미만이면 분산 축 개입은 실행 가치가 없다.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/diag_prod_varceiling.py
"""
import os, sys
import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict, skill      # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

BASE = os.path.join(LG, 'harness/cache')
DIST = os.path.join(LG, 'harness/cache_distill')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def main():
    print('=' * 96)
    print('프로덕션 블렌드에 남은 분산/캘리브레이션 여지 (오라클 = 낙관적 상한)')
    print('=' * 96)
    print(f'{"fold":>5} {"프로덕션":>10} {"+오라클아핀":>12} {"이득":>8} '
          f'{"+오라클단조":>12} {"이득":>8} | {"증류student":>12} {"기울기a":>8}')
    print('-' * 96)
    rows = []
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy')).astype(np.float64)
        ps = [predict(dict(PROD), dict(np.load(os.path.join(BASE, f'pred_{y}_{s}.npz'))))
              for s in SEEDS]
        p = np.mean(ps, 0)                       # 프로덕션과 동일한 예측 배깅
        k = skill(p, yv)
        a, b = np.polyfit(p, yv, 1)
        k_aff = skill(a * p + b, yv)
        k_iso = skill(IsotonicRegression(out_of_bounds='clip').fit_transform(p, yv), yv)

        st = [np.load(os.path.join(DIST, f'preds_{y}_{s}.npz'))['stud'] for s in SEEDS]
        k_st = skill(np.mean(st, 0).astype(np.float64), yv)

        rows.append((y, k, k_aff, k_iso, k_st, a))
        print(f'{y:>5} {k:10.1f} {k_aff:12.1f} {k_aff-k:+8.1f} {k_iso:12.1f} '
              f'{k_iso-k:+8.1f} | {k_st:12.1f} {a:8.3f}')

    print('-' * 96)
    aff = np.mean([r[2] - r[1] for r in rows])
    iso = np.mean([r[3] - r[1] for r in rows])
    gap = np.mean([r[4] - r[1] for r in rows])
    print(f'3폴드 평균: 오라클 아핀 여지 {aff:+.1f}   오라클 단조 여지 {iso:+.1f}   '
          f'증류 student - 프로덕션 {gap:+.1f}')
    print()
    print(f'  · 증류의 이득 성분은 분산 항 +165.3 이었다. 프로덕션에서 그 축 전체에 '
          f'남은 오라클 상한은 {aff:+.1f} 이다.')
    print(f'  · LB 노이즈 바닥 ±12 대비 {"미만 → 실행 가치 없음" if abs(aff) < 12 else "초과"}')
    print(f'  · 증류 student 는 프로덕션 대비 {gap:+.1f} '
          f'({"열세" if gap < 0 else "우세"})')


if __name__ == '__main__':
    main()
