#!/usr/bin/env python3
"""residual_scan.py — 조건부 잔차 구조를 전수 조사한다 (학습비용 0).

동기
----
지금까지 가설을 사람이 지어내고 검증했는데, 오늘만 6번 틀렸고 두 번은 단일 폴드
관찰에서 출발한 게 원인이었다. 방향을 바꿔서 **잔차가 어디에 남아 있는지 데이터가
가리키게** 한다.

`count_shift_probe.py` 가 카운트에 대해 한 것을 모든 후보 분할에 대해 일반화한다:
분할 c 마다 잔차 평균 s_c = mean(y - p | c) 를 가산항으로 적합하고, **빈도가중 평균을
제거해 전역 레벨 성분을 뺀 뒤**(그건 SHIFT 가 이미 담당하고 로컬 최적화가 위험하다)
남은 구조가 홀드아웃 폴드로 전이되는지 본다.

전이되는 분할이 있으면 = 모델이 그 조건에서 체계적으로 틀리고 있다 = 실제 개선 여지.
전부 0이면 = 사후 보정 축이 완전히 닫힌 것이고, 더 이상 이 방향을 뒤질 이유가 없다.

프로토콜: inner 3폴드 x 5시드 예측 배깅(프로덕션과 동일한 채점), LOFO 검증.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict, skill, CACHE

FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def bucket(v, edges):
    return np.digitize(np.nan_to_num(v, nan=0.0), edges).astype(str)


def main():
    cols = ['season', 'balls_before', 'strikes_before', 'outs_before', 'inning',
            'base_state', 'pitcher_hand', 'batter_hand', 'li', 'game_month',
            'asof_pitcher_n', 'asof_batter_n', 'score_diff_pitcher_team',
            'num_runners_on', 'top_bottom', 'asof_pitcher_success_rate',
            'asof_pitcher_fastball_rate', 'control_success']
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), usecols=cols)
    df.columns = [c.replace('﻿', '') for c in df.columns]

    D = {}
    for y in FOLDS:
        va = df[df.season == y].reset_index(drop=True)
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        assert len(yv) == len(va)
        ps = [predict(dict(PROD), dict(np.load(os.path.join(CACHE, f'pred_{y}_{s}.npz'))))
              for s in SEEDS]
        p = np.mean(ps, axis=0)                     # 프로덕션과 동일한 예측 배깅
        b = va.balls_before.fillna(0).astype(int).astype(str)
        st = va.strikes_before.fillna(0).astype(int).astype(str)
        plat = (va.pitcher_hand.astype(str) == va.batter_hand.astype(str)).astype(int).astype(str)
        segs = {
            'count (볼×스트라이크)': (b + '_' + st).values,
            'inning': va.inning.fillna(0).astype(int).clip(1, 10).astype(str).values,
            'base_state': va.base_state.astype(str).values,
            'outs': va.outs_before.fillna(0).astype(int).astype(str).values,
            'platoon(동일손 여부)': plat.values,
            'pitcher_hand × batter_hand':
                (va.pitcher_hand.astype(str) + '_' + va.batter_hand.astype(str)).values,
            'game_month': va.game_month.fillna(0).astype(int).astype(str).values,
            'top_bottom': va.top_bottom.astype(str).values,
            'num_runners': va.num_runners_on.fillna(0).astype(int).astype(str).values,
            'li 구간': bucket(va.li.values, [0.5, 0.8, 1.2, 2.0, 3.0]),
            'asof_pitcher_n 구간': bucket(va.asof_pitcher_n.values, [100, 300, 800, 1500, 3000]),
            'asof_batter_n 구간': bucket(va.asof_batter_n.values, [50, 150, 400, 900]),
            'score_diff 구간': bucket(va.score_diff_pitcher_team.values, [-4, -1, 0, 1, 4]),
            'asof 성공률 구간': bucket(va.asof_pitcher_success_rate.values,
                                  [0.45, 0.49, 0.52, 0.55, 0.58]),
            'fastball 비율 구간': bucket(va.asof_pitcher_fastball_rate.values,
                                    [0.35, 0.45, 0.55, 0.65]),
            'count × platoon': (b + '_' + st + '|' + plat).values,
            'count × asof_n구간': (b + '_' + st + '|' +
                                bucket(va.asof_pitcher_n.values, [300, 1500])).values,
        }
        D[y] = (yv, p, segs)

    names = list(D[FOLDS[0]][2].keys())
    print(f'프로덕션 배깅 예측 기준, LOFO 검증 (3폴드 x 5시드)')
    print(f'\n  {"분할":>26} {"셀수":>6} {"LOFO 평균":>10} {"양수":>6} {"폴드별 델타":>26}')
    results = []
    for nm in names:
        keys = sorted(set(k for y in FOLDS for k in np.unique(D[y][2][nm])))
        ds = []
        for held in FOLDS:
            sel = [y for y in FOLDS if y != held]
            num = {k: 0.0 for k in keys}; den = {k: 0.0 for k in keys}
            for y in sel:
                yv, p, segs = D[y]
                r = yv - p; g = segs[nm]
                for k in keys:
                    m = g == k
                    if m.any():
                        num[k] += r[m].sum(); den[k] += m.sum()
            sh = {k: (num[k] / den[k] if den[k] else 0.0) for k in keys}
            # 전역 레벨 성분 제거 — 선택 폴드 빈도로 가중
            tw = sum(den.values())
            mu = sum(sh[k] * den[k] for k in keys) / tw if tw else 0.0
            sh = {k: sh[k] - mu for k in keys}
            yv, p, segs = D[held]
            add = np.array([sh.get(k, 0.0) for k in segs[nm]])
            ds.append(skill(np.clip(p + add, 1e-6, 1 - 1e-6), yv) - skill(p, yv))
        d = np.array(ds)
        results.append((nm, len(keys), d))
        pf = ' '.join(f'{v:+.2f}' for v in d)
        print(f'  {nm:>26} {len(keys):>6} {d.mean():+10.2f} {(d>0).sum():>4}/3 {pf:>26}')

    print('\n' + '=' * 80)
    best = max(results, key=lambda r: r[2].mean())
    allpos = [r for r in results if (r[2] > 0).all()]
    print(f'  최고 분할: {best[0]}  LOFO 평균 {best[2].mean():+.2f}점')
    print(f'  3폴드 모두 양수인 분할: {len(allpos)}개' +
          (f' → {", ".join(r[0] for r in allpos)}' if allpos else ''))
    print(f'\n  ※ LB 노이즈 바닥 ±12점. 위 수치가 전부 그 1/10에도 못 미치면')
    print(f'    사후 보정 축에는 남은 구조가 없다는 뜻이다.')


if __name__ == '__main__':
    main()
