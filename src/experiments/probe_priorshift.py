"""probe_priorshift.py — H1 게이트: 라벨 없이 다음 시즌의 정답률 r 을 추정할 수 있는가.

배경 (`outputs/526` §1)
-----------------------
시즌 정답률이 단조 하락한다(2019 .5647 -> 2024 .4861). 우리 arm 은 이 드리프트를
못 따라가서 과대예측량 m = mean(p) − r 이 −0.0166 -> +0.0031 로 계속 오른다.
2025 외삽 m ≈ +0.010. 전역 상수 시프트만으로 손익분기(m>+0.00345)를 넘으면 이득.

방법: Saerens–Latinne–Decaestecker EM (prior shift / label shift 적응).
모델이 사전확률 r_old 로 학습됐고 배포 분포의 사전확률이 r_new 로 바뀌었을 때,
**라벨 없이** 예측만으로 r_new 를 추정하고 사후확률을 재조정한다.

    w      = (r/(1−r)) · ((1−r_old)/r_old)
    p'_i   = p_i·w / (p_i·w + (1−p_i))
    r^(t+1)= mean(p'_i)                        (수렴할 때까지 반복)

⚠️ 규정: 라벨을 일절 쓰지 않는다. 다른 평가 행의 정답을 도출하지도 않는다.
   test 피처의 주변분포만 쓰는 표준 비지도 적응이라 `524` 의 리스크와 무관하다.
   (행 독립성: r 추정은 프레임 전체 통계이므로 **배포 시에는 train 에서 외삽한
   상수**로 고정해야 규정 4 를 지킨다. 이 프로브는 추정 가능성만 검증한다.)

사전 확정 판정 기준 (착수 전 고정)
---------------------------------
G1) 4폴드에서 |r_hat − r_true| 중앙값 ≤ 0.005  → EM 이 실제로 작동
G2) EM 보정 후 skill 이 4폴드 전부에서 개선     → 배포 가치
둘 다 통과해야 H1 을 다음 단계(세그먼트별, H2)로 넘긴다.

실행: venv311/bin/python3 harness/probe_priorshift.py
"""
import os
import sys

import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023, 2024]
SEEDS = [7, 123, 2025, 31415, 8675309]
EPS = 1e-6


def arm(y):
    return np.mean([predict(PROD, dict(np.load(os.path.join(CACHE, f'pred_{y}_{s}.npz'))))
                    for s in SEEDS
                    if os.path.exists(os.path.join(CACHE, f'pred_{y}_{s}.npz'))], axis=0)


def em_prior(p, r_old, iters=500, tol=1e-10):
    """라벨 없이 새 사전확률을 추정한다."""
    r = float(np.mean(p))
    for _ in range(iters):
        w = (r / (1 - r)) * ((1 - r_old) / r_old)
        q = p * w / (p * w + (1 - p))
        r_new = float(np.mean(q))
        if abs(r_new - r) < tol:
            break
        r = r_new
    return r


def apply_shift(p, r_old, r_new):
    w = (r_new / (1 - r_new)) * ((1 - r_old) / r_old)
    return np.clip(p * w / (p * w + (1 - p)), EPS, 1 - EPS)


def skill(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


def main():
    # r_old = 각 폴드의 학습 구간(seasons < Y) 실제 정답률
    import pandas as pd
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), encoding='utf-8-sig',
                     usecols=['season', 'control_success'])
    r_by_season = df.groupby('season').control_success.mean()
    n_by_season = df.groupby('season').control_success.size()

    print('=' * 84)
    print('H1 게이트 — 라벨 없이 다음 시즌 r 을 추정할 수 있는가 (SLD EM)')
    print('=' * 84)
    print(f'  {"Y":>6} {"r_old(학습)":>11} {"r_true":>8} {"r_hat(EM)":>10} '
          f'{"오차":>8} {"mean(p)":>9} {"naive오차":>9}')
    errs, naive = [], []
    rows = []
    for Y in FOLDS:
        y = np.load(os.path.join(CACHE, f'y_{Y}.npy'))
        V = y.mean() * (1 - y.mean())
        p = arm(Y)
        m = df.season < Y
        r_old = float((df.control_success[m] * 1.0).sum() / m.sum())
        r_true = float(y.mean())
        r_hat = em_prior(np.clip(p, EPS, 1 - EPS), r_old)
        e = r_hat - r_true
        e0 = float(p.mean()) - r_true
        errs.append(abs(e)); naive.append(abs(e0))
        rows.append((Y, y, V, p, r_old, r_true, r_hat))
        print(f'  {Y:>6} {r_old:11.4f} {r_true:8.4f} {r_hat:10.4f} {e:+8.4f} '
              f'{p.mean():9.4f} {e0:+9.4f}')

    med = float(np.median(errs))
    print(f'\n  |오차| 중앙값  EM {med:.4f}  vs  naive(mean p) {np.median(naive):.4f}')
    g1 = med <= 0.005
    print(f'  [G1] EM 오차 <= 0.005 : {"PASS" if g1 else "FAIL"}')

    print('\n' + '=' * 84)
    print('[G2] EM 보정이 skill 을 올리는가')
    print('=' * 84)
    print(f'  {"Y":>6} {"기준선":>10} {"EM 보정":>10} {"Δ":>9} {"오라클(r_true)":>14} ')
    d_em, d_or = [], []
    for Y, y, V, p, r_old, r_true, r_hat in rows:
        s0 = skill(p, y, V)
        s1 = skill(apply_shift(np.clip(p, EPS, 1 - EPS), r_old, r_hat), y, V)
        s2 = skill(apply_shift(np.clip(p, EPS, 1 - EPS), r_old, r_true), y, V)
        d_em.append(s1 - s0); d_or.append(s2 - s0)
        print(f'  {Y:>6} {s0:10.1f} {s1:10.1f} {s1-s0:+9.1f} {s2-s0:+14.1f}')
    d_em = np.array(d_em); d_or = np.array(d_or)
    g2 = bool((d_em > 0).all())
    print(f'\n  EM   평균 {d_em.mean():+.1f}  전폴드양수={g2}')
    print(f'  오라클 평균 {d_or.mean():+.1f}  (r 을 정확히 알 때의 상한)')
    print(f'  [G2] 전폴드 양수 : {"PASS" if g2 else "FAIL"}')

    print('\n' + '=' * 84)
    print(f'판정: H1 {"통과 -> H2(세그먼트별) 진행" if (g1 and g2) else "미통과"}')
    print('=' * 84)
    print('  참고: 블렌드 전달 배수 0.25~0.50 을 곱하면 LB 기대치가 된다.')
    print(f'  오라클 상한 기준 LB 기대: {d_or.mean()*0.25:+.1f} ~ {d_or.mean()*0.50:+.1f}')


if __name__ == '__main__':
    main()
