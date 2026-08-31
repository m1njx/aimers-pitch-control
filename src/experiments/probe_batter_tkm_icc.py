#!/usr/bin/env python3
"""probe_batter_tkm_icc.py — 타자 트랙맨 프로파일 축에 '정보가 존재하는가' (사망 판정 전용).

배경
----
팀 기록상 유일하게 안 건드린 정보축이 '타자 트랙맨 프로파일'이다. 그런데 트랙맨의
측정값은 전부 **투수 쪽 물리량**(구속/회전/무브먼트/릴리스/익스텐션)이다. 타자에게
붙일 수 있는 것은 '그 타자가 상대한 투구들의 집계'뿐이고, 그 집계의 대부분은 **누가
던졌는가**로 결정된다. 타자 고유 성분(상대 투수의 투구 선택 변화)이 실제로 존재하는지
부터 확인한다.

무엇을 재는가
------------
트랙맨 데이터만으로 잰다 — batter_id 매핑이 필요 없다(트랙맨에 batter_trackman_id 가 있다).
  1. 각 물리량에서 **(시즌, 투수) 평균을 제거**한다  -> 잔차
  2. 잔차를 타자별로 모아 **타자간 분산**을 구한다
  3. 표본 노이즈 보정: E[관측 타자간분산] = 참분산 + E[타자내분산 / n_b]
     참분산 = 관측 - 보정항.  ICC = 참분산 / 전체분산
비교 기준으로 **투수 ICC**(같은 절차, 투수 대신 타자를 제거)를 같이 낸다.

판정
----
타자 ICC 가 투수 ICC 대비 무시할 수준이면, 타자 트랙맨 프로파일은 원리상 노이즈이고
entity resolution 에 시간을 쓸 이유가 없다. **이 진단은 축을 죽이는 용도이지, 통과해도
이득 추정치가 아니다** (outputs/514: 단독 측정은 풀 파이프라인 예측력이 없다).

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/probe_batter_tkm_icc.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')

PHYS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
        'extension', 'rel_height', 'rel_side', 'zone_speed']


def icc(resid, key, n_min=30):
    """key 별 참 분산 성분 / 전체 분산. 표본 노이즈 보정 포함."""
    d = pd.DataFrame({'r': resid, 'k': key}).dropna()
    g = d.groupby('k')['r']
    n = g.size()
    keep = n[n >= n_min].index
    d = d[d.k.isin(keep)]
    if len(d) < 1000:
        return np.nan, 0, 0
    g = d.groupby('k')['r']
    n = g.size().values.astype(float)
    m = g.mean().values
    v_within = g.var(ddof=1).fillna(0.0).values
    w = n / n.sum()
    var_obs = np.sum(w * (m - np.sum(w * m)) ** 2)          # 가중 타자간 분산(관측)
    bias = np.sum(w * v_within / n)                          # 표본 노이즈 기여분
    var_true = max(var_obs - bias, 0.0)
    return var_true / d['r'].var(ddof=1), len(keep), int(np.median(n))


def main():
    t0 = time.time()
    cols = ['season', 'pitcher_trackman_id', 'batter_trackman_id',
            'pitcher_hand', 'batter_hand', 'pitch_type_group'] + PHYS
    tm = pd.read_csv(os.path.join(LG, 'open/data/trackman_history.csv'),
                     usecols=lambda c: c.replace('﻿', '') in cols)
    tm.columns = [c.replace('﻿', '') for c in tm.columns]
    print(f'trackman {tm.shape}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'  투수 {tm.pitcher_trackman_id.nunique():,}명  '
          f'타자 {tm.batter_trackman_id.nunique():,}명  시즌 {sorted(tm.season.unique())}')

    ps = tm['season'].astype(str) + '_' + tm['pitcher_trackman_id'].astype(str)
    bs = tm['season'].astype(str) + '_' + tm['batter_trackman_id'].astype(str)

    print('\n' + '=' * 88)
    print('투수 효과 제거 후 남는 분산 성분 (ICC, 표본노이즈 보정)')
    print(f'{"물리량":>20} {"타자 ICC":>10} {"투수 ICC":>10} {"비율":>8}   '
          f'{"타자수":>7} {"중앙n":>6}')
    print('-' * 88)
    rows = []
    for c in PHYS:
        x = tm[c].astype(np.float64)
        if x.notna().sum() < 10000:
            continue
        # 타자 ICC: (시즌,투수) 평균 제거 후 타자별 성분
        r_b = x - x.groupby(ps).transform('mean')
        ib, nb, medb = icc(r_b.values, bs.values)
        # 투수 ICC: (시즌,타자) 평균 제거 후 투수별 성분 (대조군)
        r_p = x - x.groupby(bs).transform('mean')
        ip, npi, medp = icc(r_p.values, ps.values)
        rows.append((c, ib, ip))
        print(f'{c:>20} {ib:10.5f} {ip:10.5f} {ib/ip if ip else np.nan:8.3f}   '
              f'{nb:7,} {medb:6}', flush=True)

    R = pd.DataFrame(rows, columns=['col', 'batter_icc', 'pitcher_icc'])
    print('-' * 88)
    print(f'{"평균":>20} {R.batter_icc.mean():10.5f} {R.pitcher_icc.mean():10.5f} '
          f'{(R.batter_icc/R.pitcher_icc).mean():8.3f}')

    print('\n판정')
    ratio = (R.batter_icc / R.pitcher_icc).mean()
    if R.batter_icc.mean() < 0.005:
        print(f'  ❌ 타자 ICC 평균 {R.batter_icc.mean():.5f} — 투수 효과를 빼면 타자 고유 성분이')
        print('     사실상 없다. 타자 트랙맨 프로파일은 원리상 노이즈. entity resolution 불필요.')
    elif ratio < 0.10:
        print(f'  ⚠️ 타자 성분이 투수 대비 {ratio:.1%} 수준. 존재하되 매우 약하다.')
        print('     프로덕션의 타자 asof 피처와 중복일 가능성이 높으므로 기대치 낮음.')
    else:
        print(f'  ✅ 타자 고유 성분이 투수 대비 {ratio:.1%} 로 실재한다. 다음 단계로 진행 가치 있음.')
        print('     ⚠️ 단, 이건 정보 존재 확인일 뿐 이득 추정치가 아니다(outputs/514).')

    R.to_csv(os.path.join(LG, 'harness/probe_batter_tkm_icc.csv'), index=False)
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
