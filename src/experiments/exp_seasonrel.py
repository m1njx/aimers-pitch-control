#!/usr/bin/env python3
"""exp_seasonrel.py — 가설 Q: asof 피처의 시즌상대 재표현.

진단 근거 (harness/ceiling_probe.py, outputs/507)
------------------------------------------------
피처 행렬을 고정한 채 라벨 출처만 바꾸면 skill 이 738.8 → 1041.3 (+302.5) 뛴다.
즉 병목은 정보량이 아니라 **시간 일반화**다. 격차를 구간별로 쪼개면:

    asof_n   0~100    격차  -46.9   (시즌내 이력이 없으면 격차도 없다)
    asof_n 100~500    격차 +472.8
    asof_n 500~1500   격차 +397.4
    asof_n 1500~      격차 +263.2

시즌내 이력이 쌓인 구간에서만 격차가 난다 → asof 피처를 '이번 시즌 기준으로 어떻게
읽어야 하는가'를 과거 학습 모델이 못 맞추고 있다는 뜻.

가설
----
리그 성공률이 2019 .5647 → 2024 .4861 로 크게 표류하는데, **`season` 은 피처
화이트리스트에서 제외돼 있어 모델은 이 행이 몇 년도인지 모른다.** 따라서
asof_rate=.52 가 "평균 이상"인지 "평균 이하"인지 구분할 수 없고, 5개 시즌의 서로
다른 매핑을 하나로 뭉개 학습한다. asof 비율을 **그 시즌 리그 수준 대비 편차**로
다시 표현하면 매핑이 정상화(stationary)되어 시간 일반화가 개선될 수 있다.

504와 무엇이 다른가
-------------------
`outputs/504` Phase A(era-offset)는 **라벨/예측 레벨에 시즌 오프셋**을 걸었고 단조
하락으로 기각됐다(`exp_era.py:108-113`). 여기는 **피처 재표현**이다. 전자는 모델이
맞춰야 할 목표를 옮기고, 후자는 모델이 읽는 입력의 의미를 통일한다. 별개 축이다.

규정 준수
---------
시즌별 리그율 테이블은 train 에서만 산출하고, 평가 시즌 값은 과거 추세로 투영한
**고정 상수**를 쓴다(`exp_era.project_level` 재사용). 추론 시 test.csv 의 배치 통계를
전혀 계산하지 않으므로 규정4에 저촉되지 않는다 — 프로덕션의 SHIFT 상수와 동일한 성격.

    export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 -u harness/exp_seasonrel.py --years 2022 2023
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))

import build_cache as bc
from exp_era import project_level

# 시즌 리그 수준에 따라 의미가 달라지는 비율형 asof 피처들
REL_COLS = ['asof_pitcher_success_rate', 'asof_pitcher_middle_rate',
            'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
            'asof_pitcher_reverse_rate', 'asof_batter_success_rate',
            'asof_batter_middle_rate',
            'asof_pitcher_prev1_game_success_rate',
            'asof_pitcher_prev3_game_success_rate',
            'asof_pitcher_prev5_game_success_rate']

LEVELS = None          # {season: 리그 성공률}, 평가 시즌은 투영 상수
_orig_build_features = bc.build_features


def patched_build_features(df, prep, dec, cat_map):
    X, X133 = _orig_build_features(df, prep, dec, cat_map)
    lvl = df['season'].map(LEVELS).astype(np.float64)
    lvl = lvl.fillna(np.mean(list(LEVELS.values()))).values
    add = {}
    for c in REL_COLS:
        if c not in df.columns:
            continue
        v = df[c].astype(np.float64).values
        # 편차와 비율 두 형태 — 전자는 가법, 후자는 승법 드리프트를 흡수
        add[f'rel_{c}_diff'] = (v - lvl).astype(np.float32)
        add[f'rel_{c}_ratio'] = (v / np.clip(lvl, 1e-6, None)).astype(np.float32)
    A = pd.DataFrame(add)
    A.index = X.index
    X = pd.concat([X, A], axis=1)
    A2 = A.copy(); A2.index = X133.index
    X133 = pd.concat([X133, A2], axis=1)
    return X, X133


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025, 31415, 8675309])
    ap.add_argument('--tag', default='seasonrel')
    a = ap.parse_args()

    global LEVELS
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    cache_dir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cache_dir, exist_ok=True)
    bc.build_features = patched_build_features
    bc.CACHE = cache_dir

    for y in a.years:
        tr = df[df.season < y]
        rates = tr.groupby('season')['control_success'].mean().to_dict()
        r_hat, _ = project_level(rates)
        LEVELS = dict(rates)
        LEVELS[y] = r_hat            # 평가 시즌은 과거 추세 투영 상수 (test.csv 미참조)
        actual = df[df.season == y]['control_success'].mean()
        print(f'\n=== eval {y}: 투영 r_hat={r_hat:.4f} (실제 {actual:.4f}, '
              f'오차 {r_hat-actual:+.4f}) ===', flush=True)
        print('  학습시즌 리그율 ' + ' '.join(f'{k}:{v:.4f}' for k, v in sorted(rates.items())),
              flush=True)
        bc.run_fold(df, y, a.seeds)

    print('\n[채점]', flush=True)
    from exp_capacity import score_dir
    base = score_dir(os.path.join(LG, 'harness/cache'), a.years, a.seeds)
    new = score_dir(cache_dir, a.years, a.seeds)
    print(f'  Q0 (현행)        inner={base["inner"]:9.1f}  '
          f'연도별={ {k: round(v,1) for k,v in base["season_mean"].items()} }  '
          f'seed_sd={base["seed_sd"]:.1f}')
    print(f'  Q1 (시즌상대 asof) inner={new["inner"]:9.1f}  '
          f'연도별={ {k: round(v,1) for k,v in new["season_mean"].items()} }  '
          f'seed_sd={new["seed_sd"]:.1f}')
    d = new['inner'] - base['inner']
    noise = float(np.mean([base['seed_sd'], new['seed_sd']]))
    print(f'\n  → 델타={d:+.1f}  노이즈(보수)={noise:.1f}  신뢰가능={bool(d > noise)}')
    print(f'  [참고] 평균의 표준오차={noise/np.sqrt(len(a.seeds)*len(a.years)):.1f}')
    print(f'총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
