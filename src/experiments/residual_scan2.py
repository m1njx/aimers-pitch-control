#!/usr/bin/env python3
"""residual_scan2.py — 조건부 잔차 구조 확장 스캔 (1차에서 안 본 분할들).

1차(`residual_scan.py`)는 17개 분할을 봤고 최대가 count×platoon +3.03 이었다.
여기서는 1차에서 다루지 않은 분할을 본다: 요일/팀/각종 asof 비율/직전경기 성적/
승률기대/2-way 조합, 그리고 예측확률 구간(캘리브레이션 잔차의 직접 확인).

프로토콜은 1차와 동일: 프로덕션 배깅 예측, 전역 레벨 성분 제거(빈도가중), LOFO 검증.
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


def bk(v, edges):
    """구간화 결과를 pandas Series 로 준다 — 2-way 조합에서 문자열 연결을 하려면
    ndarray(<U21) 가 아니라 Series 여야 한다."""
    return pd.Series(np.digitize(np.nan_to_num(np.asarray(v, float), nan=0.0),
                                 edges)).astype(str)


def main():
    cols = ['season', 'balls_before', 'strikes_before', 'outs_before', 'inning',
            'game_dayofweek', 'pitcher_team_id', 'batter_team_id', 'top_bottom',
            'pitcher_hand', 'batter_hand', 'num_runners_on', 'run_total_before',
            'home_win_expectancy', 'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate',
            'asof_pitcher_strike_rate', 'asof_pitcher_reverse_rate',
            'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
            'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
            'asof_batter_success_rate', 'asof_pitcher_n', 'control_success']
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), usecols=cols)
    df.columns = [c.replace('﻿', '') for c in df.columns]

    D = {}
    for y in FOLDS:
        va = df[df.season == y].reset_index(drop=True)
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        p = np.mean([predict(dict(PROD), dict(np.load(os.path.join(CACHE, f'pred_{y}_{s}.npz'))))
                     for s in SEEDS], axis=0)
        b = va.balls_before.fillna(0).astype(int).astype(str)
        st = va.strikes_before.fillna(0).astype(int).astype(str)
        inn_b = bk(va.inning.values, [3, 6, 8])
        segs = {
            'game_dayofweek': va.game_dayofweek.fillna(0).astype(int).astype(str).values,
            'pitcher_team_id': va.pitcher_team_id.astype(str).values,
            'batter_team_id': va.batter_team_id.astype(str).values,
            'run_total 구간': bk(va.run_total_before.values, [1, 3, 6, 10]),
            'home_win_exp 구간': bk(va.home_win_expectancy.values, [.2, .4, .6, .8]),
            'asof middle_rate 구간': bk(va.asof_pitcher_middle_rate.values, [.2, .3, .4, .5]),
            'asof ball_rate 구간': bk(va.asof_pitcher_ball_rate.values, [.3, .35, .4, .45]),
            'asof strike_rate 구간': bk(va.asof_pitcher_strike_rate.values, [.55, .6, .65, .7]),
            'asof reverse_rate 구간': bk(va.asof_pitcher_reverse_rate.values, [.05, .1, .15]),
            'asof breaking_rate 구간': bk(va.asof_pitcher_breaking_rate.values, [.2, .3, .4]),
            'asof offspeed_rate 구간': bk(va.asof_pitcher_offspeed_rate.values, [.1, .2, .3]),
            'prev1 경기성적 구간': bk(va.asof_pitcher_prev1_game_success_rate.values,
                                [.4, .48, .55, .62]),
            'prev3 경기성적 구간': bk(va.asof_pitcher_prev3_game_success_rate.values,
                                [.45, .5, .55, .6]),
            'asof_batter_rate 구간': bk(va.asof_batter_success_rate.values, [.45, .5, .55]),
            'inning 구간 x top_bottom': (inn_b + '|' + va.top_bottom.astype(str)).values,
            'count x inning 구간': (b + '_' + st + '|' + inn_b).values,
            'pitcher_hand x count': (va.pitcher_hand.astype(str) + '|' + b + '_' + st).values,
            'num_runners x outs': (va.num_runners_on.fillna(0).astype(int).astype(str)
                                   + '|' + va.outs_before.fillna(0).astype(int).astype(str)).values,
            '예측확률 20분위': bk(p, list(np.quantile(p, np.linspace(.05, .95, 19)))),
        }
        D[y] = (yv, p, segs)

    names = list(D[FOLDS[0]][2].keys())
    print('확장 잔차 스캔 — 프로덕션 배깅 예측, LOFO 검증 (3폴드 x 5시드)')
    print(f'\n  {"분할":>24} {"셀수":>6} {"LOFO 평균":>10} {"양수":>6} {"폴드별":>24}')
    res = []
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
            tw = sum(den.values())
            mu = sum(sh[k] * den[k] for k in keys) / tw if tw else 0.0
            sh = {k: sh[k] - mu for k in keys}
            yv, p, segs = D[held]
            add = np.array([sh.get(k, 0.0) for k in segs[nm]])
            ds.append(skill(np.clip(p + add, 1e-6, 1 - 1e-6), yv) - skill(p, yv))
        d = np.array(ds)
        res.append((nm, len(keys), d))
        print(f'  {nm:>24} {len(keys):>6} {d.mean():+10.2f} {(d>0).sum():>4}/3 '
              f'{" ".join(f"{v:+.2f}" for v in d):>24}')

    print('\n' + '=' * 78)
    best = max(res, key=lambda r: r[2].mean())
    allpos = [r for r in res if (r[2] > 0).all()]
    print(f'  최고: {best[0]}  LOFO 평균 {best[2].mean():+.2f}점')
    print(f'  3폴드 모두 양수: {len(allpos)}개' +
          (f' → {", ".join(f"{r[0]}({r[2].mean():+.2f})" for r in allpos)}' if allpos else ''))
    print('\n  ※ 1차 스캔(17분할) 최고는 count x platoon +3.03 이었다.')
    print('    LB 노이즈 바닥 ±12점 대비 여전히 미미하면 사후 보정 축은 확정적으로 닫힌다.')


if __name__ == '__main__':
    main()
