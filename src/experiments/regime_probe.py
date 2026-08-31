#!/usr/bin/env python3
"""regime_probe.py — 2024에 체제 변화(regime break)가 있었는가?

왜 이 질문이 중요한가
--------------------
우리의 inner 선별은 2021/2022/2023 폴드로 한다. 그런데 **제출 대상은 2025**이고
프로덕션은 2019~2024로 학습한다. 만약 2024에 구조적 체제 변화가 있었고 2025가 그
체제를 공유한다면:

  - inner 폴드(2021~2023)는 **구조적으로 그 체제를 볼 수 없다.** 거기서 내린 모든
    판정은 "옛 체제에서 옛 체제를 맞추는" 문제에 대한 답이다.
  - 특히 **recency 재가중이 decay=1.0(무가중)으로 기각된 것**(`504` Phase B)은
    2022/2023 폴드에서 잰 결과다. 2024 체제가 특별하다면 그 기각은 2025에 적용되지
    않는다. "최근 시즌을 더 크게 쓰라"는 처방이 여전히 살아있을 수 있다.

측정 방법 — 전이 효율
--------------------
평가 시즌 s 마다, **피처 행렬을 고정한 채**(prep/dec 는 season<s 로 fit) 두 모델을 잰다.

  P  직전 시즌(s-1) 라벨로만 학습 → s 평가        [과거에서 미래로의 전이]
  C  s 라벨로 투수분리 교차적합 → s 평가          [같은 체제 내 상한]

  전이효율 = P / C

시즌마다 난이도와 베이스레이트가 달라 절대값은 비교 불가이므로 **비율**로 본다.
2024의 전이효율만 유독 낮으면 2023→2024 사이에 체제가 끊긴 것이다.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

LGB_P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
             seed=7, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8)


def skill(p, y):
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))


def recal(p, y):
    b = np.cov(p, y)[0, 1] / p.var()
    return skill(np.clip(y.mean() + b * (p - p.mean()), 1e-6, 1 - 1e-6), y)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    rates = df.groupby('season')['control_success'].mean()
    print('시즌별 리그 성공률')
    for s, v in rates.items():
        print(f'  {s}  {v:.4f}')

    rows = []
    for s in (2021, 2022, 2023, 2024):
        past = df[df.season < s]
        va = df[df.season == s].reset_index(drop=True)
        prev = df[df.season == s - 1]
        y = va['control_success'].values.astype(np.float64)

        prep = bc.PitchPreprocessor()
        prep.fit(past, as_of_season=s - 1, is_final=False,
                 trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
        bs = ((past['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (past['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (past['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        cs = (past['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              past['strikes_before'].fillna(0).astype(int).astype(str))
        cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
        dec = bc.AsofDecomposer2(); dec.fit(past, val_season=s)

        Xva, _ = bc.build_features(va, prep, dec, cat_map)
        Xpr, _ = bc.build_features(prev, prep, dec, cat_map)
        ypr = prev['control_success'].values.astype(np.float64)

        # P: 직전 시즌만으로 학습
        p_pred = lgb.train(LGB_P, lgb.Dataset(Xpr, label=ypr)).predict(Xva)
        # C: 같은 시즌, 투수분리 교차적합
        oof = np.zeros(len(va))
        g = va.pitcher_id.values
        for tr_i, va_i in GroupKFold(n_splits=5).split(Xva, y, g):
            oof[va_i] = lgb.train(LGB_P, lgb.Dataset(Xva.iloc[tr_i], label=y[tr_i])
                                  ).predict(Xva.iloc[va_i])

        kp, kc = recal(p_pred, y), recal(oof, y)
        rows.append((s, len(prev), len(va), kp, kc, kp / kc if kc > 0 else float('nan')))
        print(f'  [{s}] 완료 ({time.time()-t0:.0f}s)', flush=True)

    print('\n' + '=' * 78)
    print(f'  {"평가시즌":>8} {"학습행(s-1)":>12} {"P 직전시즌":>11} {"C 동시대":>10} '
          f'{"전이효율":>10} {"리그율변화":>11}')
    for s, npr, nva, kp, kc, eff in rows:
        d = rates[s] - rates[s - 1]
        print(f'  {s:>8} {npr:>12,} {kp:11.1f} {kc:10.1f} {eff:10.3f} {d:+11.4f}')
    print('=' * 78)

    effs = {r[0]: r[5] for r in rows}
    older = [effs[s] for s in (2021, 2022, 2023)]
    print(f'\n  2021~2023 전이효율 평균 {np.mean(older):.3f}  (범위 {min(older):.3f}~{max(older):.3f})')
    print(f'  2024 전이효율          {effs[2024]:.3f}')
    gapz = (effs[2024] - np.mean(older)) / (np.std(older, ddof=1) if len(older) > 1 else 1)
    print(f'  → 2024는 과거 대비 {gapz:+.2f} 표준편차')
    if effs[2024] < min(older) * 0.85:
        print('\n  판정: 2023→2024 사이에 체제가 끊겼을 가능성이 크다.')
        print('        inner 폴드(2021~2023)는 그 체제를 볼 수 없으므로,')
        print('        recency 재가중 기각(504 Phase B)은 2025에 적용되지 않을 수 있다.')
    else:
        print('\n  판정: 2024의 전이효율이 과거 시즌들과 다르지 않다.')
        print('        체제 변화 근거 없음. inner 폴드의 대표성은 유지된다.')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
