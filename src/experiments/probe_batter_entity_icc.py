#!/usr/bin/env python3
"""probe_batter_entity_icc.py — 게이트 G1a (outputs/520). 학습 없음, 비용 ~2분.

질문
----
`control_success` 에 **타자 고유 성분**이 표본노이즈 보정 후에도 남는가.
남지 않으면 타자 측 EB shrinkage 계열은 원리상 채울 것이 없다.

방법 (outputs/515 의 ICC 절차를 라벨에 적용)
-------------------------------------------
통제를 단계적으로 더하면서 (교대 투영으로 가법 통제항 제거) 잔차의
타자간 분산에서 표본노이즈 기여분을 뺀다:
    E[관측 엔티티간분산] = 참분산 + E[엔티티내분산 / n]
대조군으로 같은 절차의 **투수 ICC** 를 낸다.

판정 (사전 확정, outputs/520)
  타자 ICC(전 통제 후) >= 0.005 이면 통과, 미만이면 축 종결.

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/probe_batter_entity_icc.py
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
N_MIN = 30


def icc(resid, key, n_min=N_MIN):
    d = pd.DataFrame({'r': resid, 'k': key}).dropna()
    n = d.groupby('k')['r'].size()
    d = d[d.k.isin(n[n >= n_min].index)]
    if len(d) < 1000:
        return np.nan, 0, 0
    g = d.groupby('k')['r']
    n = g.size().values.astype(float)
    m = g.mean().values
    v_within = g.var(ddof=1).fillna(0.0).values
    w = n / n.sum()
    var_obs = np.sum(w * (m - np.sum(w * m)) ** 2)
    bias = np.sum(w * v_within / n)
    return max(var_obs - bias, 0.0) / d['r'].var(ddof=1), len(n), int(np.median(n))


def demean(y, controls, n_iter=6):
    """교대 투영으로 가법 통제항 제거."""
    r = y.astype(np.float64).copy()
    for _ in range(n_iter):
        for c in controls:
            r = r - pd.Series(r).groupby(c).transform('mean').values
    return r


def main():
    t0 = time.time()
    cols = ['season', 'pitcher_id', 'batter_id', 'balls_before', 'strikes_before',
            'outs_before', 'pitcher_hand', 'batter_hand', 'batter_team_id',
            'pitcher_team_id', 'inning', 'control_success',
            'asof_batter_n', 'asof_batter_success_rate']
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=lambda c: c.replace('﻿', '') in cols)
    df.columns = [c.replace('﻿', '') for c in df.columns]
    print(f'train {df.shape}  ({time.time()-t0:.0f}s)')
    print(f'  시즌 {sorted(df.season.unique())}  투수 {df.pitcher_id.nunique():,}명  '
          f'타자 {df.batter_id.nunique():,}명  base rate {df.control_success.mean():.4f}\n',
          flush=True)

    y = df['control_success'].values.astype(np.float64)
    ps = (df.season.astype(str) + '_' + df.pitcher_id.astype(str)).values
    bs = (df.season.astype(str) + '_' + df.batter_id.astype(str)).values
    cnt = (df.balls_before.fillna(0).astype(int).astype(str) + '-' +
           df.strikes_before.fillna(0).astype(int).astype(str)).values
    plat = (df.pitcher_hand.astype(str) + 'v' + df.batter_hand.astype(str)).values
    bteam = (df.season.astype(str) + '_' + df.batter_team_id.astype(str)).values
    pteam = (df.season.astype(str) + '_' + df.pitcher_team_id.astype(str)).values

    print('=' * 92)
    print('control_success 의 엔티티 ICC (표본노이즈 보정). 통제를 단계적으로 추가.')
    print('=' * 92)
    print(f'{"통제":<34} {"타자 ICC":>10} {"투수 ICC":>10} {"비율":>8}  {"타자수":>7} {"중앙n":>7}')
    print('-' * 92)

    ladder = [
        ('없음 (원자료)',                 [],                   []),
        ('+ 상대 엔티티(시즌)',            [ps],                 [bs]),
        ('+ 카운트',                       [ps, cnt],            [bs, cnt]),
        ('+ platoon',                      [ps, cnt, plat],      [bs, cnt, plat]),
        ('+ 소속팀(시즌)',                 [ps, cnt, plat, bteam], [bs, cnt, plat, pteam]),
    ]
    rows = []
    for name, cb, cp in ladder:
        rb = demean(y, cb) if cb else y - y.mean()
        ib, nb, medb = icc(rb, bs)
        rp = demean(y, cp) if cp else y - y.mean()
        ip, npi, medp = icc(rp, ps)
        rows.append((name, ib, ip, ib * np.var(rb, ddof=1)))
        print(f'{name:<34} {ib:10.5f} {ip:10.5f} {ib/ip if ip else np.nan:8.3f}  '
              f'{nb:7,} {medb:7,}', flush=True)

    final_b, final_p = rows[-1][1], rows[-1][2]
    print('-' * 92)
    r0 = df.control_success.mean(); V0 = r0 * (1 - r0)
    nmed0 = float(df.asof_batter_n.median())
    print('\n통제 단계별 타자 축 상한 (skill 점) 과 EB 로 회수 가능한 몫')
    print(f'{"통제":<34} {"오라클 상한":>11} {"rho":>7} {"EB 최대회수":>11} {"잔여(추정오차)":>13}')
    for name, ib, ip, t2 in rows:
        ceil = 1e5 * t2 / V0
        rho = t2 / (t2 + V0 / nmed0)
        print(f'{name:<34} {ceil:+11.1f} {rho:7.4f} {rho*ceil:+11.1f} {(1-rho)*ceil:+13.1f}')

    # 프로덕션이 이미 그 정보를 갖고 있는가 — asof_batter_success_rate 는 그 ICC 의 추정량이다
    print('\n프로덕션 타자 피처의 커버리지')
    m = df.asof_batter_success_rate.notna()
    print(f'  asof_batter_success_rate 결측 아님 {m.mean():.4f}, '
          f'asof_batter_n 중앙값 {df.asof_batter_n.median():.0f}')
    yb = pd.Series(y).groupby(bs).transform('mean').values
    print(f'  타자 시즌평균 성공률의 관측 표준편차 {pd.Series(yb).std():.5f}  '
          f'(참 성분 sd ≈ {np.sqrt(final_b * y.var()):.5f})')

    # ---- 축 전체의 상한을 skill 점으로 환산 ----
    r = df.control_success.mean()
    V = r * (1 - r)
    rb_full = demean(y, [ps, cnt, plat, bteam])
    tau2 = final_b * np.var(rb_full, ddof=1)
    print('\n타자 축 전체의 상한 (완전 정보 오라클)')
    print(f'  V = r(1-r) = {V:.5f}')
    print(f'  참 타자 분산 tau^2 = {tau2:.6e}  ->  상한 {1e5*tau2/V:+.1f} skill 점')
    nmed = float(df.asof_batter_n.median())
    rel = tau2 / (tau2 + V / nmed)
    print(f'  프로덕션 추정량 신뢰도 (n 중앙값 {nmed:.0f}): rho = {rel:.4f}')
    print(f'    이미 포착 {1e5*rel*tau2/V:+.1f} 점 / 잔여 {1e5*(1-rel)*tau2/V:+.1f} 점')
    print('    잔여분은 추정오차이지 수축상수로 되찾는 양이 아니다 (EB 최적이 정확히 rho*tau^2).')

    # ---- 타자 x 카운트 상호작용이 있는가 (조건부 EB 변형의 여지) ----
    cg = np.where(df.strikes_before.fillna(0).values > df.balls_before.fillna(0).values,
                  'ahead', 'notahead')
    r_int = demean(y, [bs, cnt, ps])
    bxc = pd.Series(bs).astype(str).values + '|' + cg
    i_int, n_int, med_int = icc(r_int, bxc)
    print(f'\n타자 x 카운트(ahead/notahead) 상호작용 ICC (가법항 제거 후): {i_int:.5f}'
          f'   셀 {n_int:,}  중앙n {med_int:,}')
    print(f'  -> 상호작용 상한 {1e5*i_int*np.var(r_int, ddof=1)/V:+.1f} skill 점')

    print('\n판정 (사전기준: 전 통제 후 타자 ICC >= 0.005)')
    if final_b >= 0.005:
        print(f'  ✅ 통과 — 타자 ICC {final_b:.5f}. 다음 단계 진행 가치 있음.')
        print('     ⚠️ 단, 이건 정보 존재 확인일 뿐 이득 추정치가 아니다(outputs/514).')
    else:
        print(f'  ❌ 미달 — 타자 ICC {final_b:.5f} < 0.005 (투수 {final_p:.5f}, '
              f'비율 {final_b/final_p if final_p else float("nan"):.1%}).')
        print('     타자 고유 성분이 사실상 없다. 타자 EB 축은 학습하지 않고 종결.')

    pd.DataFrame(rows, columns=['controls', 'batter_icc', 'pitcher_icc',
                                'batter_tau2']).to_csv(
        os.path.join(LG, 'harness/probe_batter_entity_icc.csv'), index=False)
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
