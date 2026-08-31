#!/usr/bin/env python3
"""diag_distill_covvar.py — 증류 이득 +85.2 의 출처를 가른다 (재학습 없음).

질문
----
exp_distill.py 의 student 이득이
  (a) 캘리브레이션/수축 = 같은 순위를 더 잘 눌러 담은 것   -> Idea B 의 +25 예산에 묶인다
  (b) 정보 = Cov(p,y) 증가, 순위 자체가 좋아진 것          -> 예산에 안 묶인다
중 어디서 오는가.

분해
----
MSE = (E[p]-ybar)^2 + Var(p) - 2Cov(p,y) + Var(y)
Var(y) 는 base/student 공통이므로 상쇄된다. skill 단위(=100000/(ybar(1-ybar)))로
환산하면 델타가 정확히 세 항으로 쪼개진다:
    delta_skill = -K*[ d(bias^2) + d(Var p) - 2*d(Cov) ],  K = 100000/(ybar(1-ybar))
즉 '이득에 대한 기여' 는  bias 항 -K*d(bias^2),  분산 항 -K*d(Var p),  정보 항 +2K*d(Cov).

정보 판별 (더 결정적)
--------------------
base 의 '순위' 만으로 도달 가능한 최대 skill 을 오라클로 재보정해 구한다.
  - affine  : 검증 라벨로 최소제곱 a,b 적합 (Idea B 가 잰 바로 그 상한)
  - isotonic: 검증 라벨로 단조 회귀 적합 (순위 보존 변환의 절대 상한, 낙관적)
student 가 oracle-isotonic-base 를 넘으면 순위가 실제로 좋아진 것 = 정보다.
넘지 못하면 그 이득은 원리상 후처리로 얻을 수 있었던 것이고, 프로덕션은
이미 캘리브레이션이 닫힌 축이므로 이식해도 사라질 가능성이 크다.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/diag_distill_covvar.py
"""
import os, sys
import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import skill                      # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from scipy.stats import spearmanr               # noqa: E402

BASE = os.path.join(LG, 'harness/cache')
DIST = os.path.join(LG, 'harness/cache_distill')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def decompose(pb, ps, y):
    """base -> student 의 skill 델타를 bias / var / cov 세 항으로 쪼갠다."""
    yb = y.mean()
    K = 100000.0 / (yb * (1 - yb))
    d_bias = (ps.mean() - yb) ** 2 - (pb.mean() - yb) ** 2
    d_var = ps.var() - pb.var()
    d_cov = np.cov(ps, y, ddof=0)[0, 1] - np.cov(pb, y, ddof=0)[0, 1]
    return dict(bias=-K * d_bias, var=-K * d_var, cov=2 * K * d_cov)


def oracle_affine(p, y):
    """검증 라벨로 최적 아핀 a*p+b 를 적합 (오라클, 낙관적)."""
    a, b = np.polyfit(p, y, 1)
    return a * p + b


def oracle_isotonic(p, y):
    """검증 라벨로 최적 단조 변환을 적합 (순위 보존 변환의 상한, 매우 낙관적)."""
    return IsotonicRegression(out_of_bounds='clip').fit_transform(p, y)


def main():
    rows = []
    print('=' * 100)
    print('증류 이득 분해 — 기여도는 skill 점수 단위, 합 = 실제 델타')
    print('=' * 100)
    hdr = (f'{"fold":>5} {"seed":>8} {"delta":>8} | {"bias항":>8} {"분산항":>8} '
           f'{"정보항":>8} | {"base":>8} {"b+aff":>8} {"b+iso":>8} {"stud":>8} '
           f'{"stud-iso":>9} {"rho":>6}')
    print(hdr)
    print('-' * 100)

    for y_ in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y_}.npy')).astype(np.float64)
        for s in SEEDS:
            f = os.path.join(DIST, f'preds_{y_}_{s}.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            pb = z['base'].astype(np.float64)
            ps = z['stud'].astype(np.float64)
            assert len(pb) == len(yv), f'행 불일치 {len(pb)} vs {len(yv)}'

            kb, ks = skill(pb, yv), skill(ps, yv)
            dec = decompose(pb, ps, yv)
            k_aff = skill(oracle_affine(pb, yv), yv)
            k_iso = skill(oracle_isotonic(pb, yv), yv)
            rho = spearmanr(pb, ps).correlation
            rows.append(dict(fold=y_, seed=s, delta=ks - kb, **dec,
                             base=kb, aff=k_aff, iso=k_iso, stud=ks,
                             gap_iso=ks - k_iso, rho=rho))
            r = rows[-1]
            print(f'{y_:>5} {s:>8} {r["delta"]:+8.1f} | {r["bias"]:+8.1f} '
                  f'{r["var"]:+8.1f} {r["cov"]:+8.1f} | {kb:8.1f} {k_aff:8.1f} '
                  f'{k_iso:8.1f} {ks:8.1f} {r["gap_iso"]:+9.1f} {rho:6.4f}')

    if not rows:
        print('예측 캐시가 없다.'); return

    import pandas as pd
    R = pd.DataFrame(rows)
    print('\n' + '=' * 100)
    print('폴드별 평균')
    for y_ in FOLDS:
        v = R[R.fold == y_]
        if not len(v):
            continue
        print(f'  {y_}: delta {v.delta.mean():+7.1f}  = bias {v["bias"].mean():+7.1f} '
              f'+ var {v["var"].mean():+7.1f} + cov {v["cov"].mean():+7.1f}   '
              f'| oracle-affine 이득 {(v.aff - v.base).mean():+6.1f}  '
              f'oracle-iso 이득 {(v.iso - v.base).mean():+6.1f}  '
              f'student-iso {v.gap_iso.mean():+7.1f} (양수 {(v.gap_iso > 0).sum()}/{len(v)})')

    print('\n15셀 종합')
    for k, lab in [('bias', 'bias 항'), ('var', '분산 항'), ('cov', '정보 항(2*dCov)')]:
        v = R[k].values
        print(f'  {lab:>16}: 평균 {v.mean():+8.1f}   '
              f'(델타 평균 {R.delta.mean():+.1f} 대비 {100*v.mean()/R.delta.mean():+6.1f}%)')

    g = R.gap_iso.values
    se = g.std(ddof=1) / np.sqrt(len(g))
    print(f'\n  student - oracle_isotonic(base) : 평균 {g.mean():+.1f}  '
          f'sd {g.std(ddof=1):.1f}  SE {se:.1f}  t={g.mean()/se:.2f}  '
          f'양수 {(g > 0).sum()}/{len(g)}')
    print(f'  base 순위 -> 오라클 아핀 이득  : 평균 {(R.aff - R.base).mean():+.1f} '
          f'(= Idea B 가 잰 "+25 예산" 과 같은 종류의 양)')
    print(f'  base 순위 -> 오라클 단조 이득  : 평균 {(R.iso - R.base).mean():+.1f}')
    print(f'  base vs student Spearman rho   : 평균 {R.rho.mean():.4f}')

    print('\n판정')
    if g.mean() > 0 and (g > 0).sum() >= 13:
        print('  ✅ student 가 base 순위의 오라클 단조 상한을 넘는다 -> 순위가 실제로 '
              '좋아졌다 = 정보. Idea B 의 +25 예산에 묶이지 않는다.')
    elif (R.aff - R.base).mean() >= R.delta.mean() * 0.8:
        print('  ❌ 이득 대부분이 base 순위만으로 오라클 후처리로 도달 가능하다 '
              '-> 캘리브레이션/수축. 프로덕션은 캘리브레이션이 닫힌 축이므로 소멸 예상.')
    else:
        print('  ⚠️ 중간. 정보 항과 분산 항이 섞여 있다. 풀 파이프라인 이식으로만 갈린다.')
    R.to_csv(os.path.join(LG, 'harness/diag_distill_covvar.csv'), index=False)
    print('\n저장: harness/diag_distill_covvar.csv')


if __name__ == '__main__':
    main()
