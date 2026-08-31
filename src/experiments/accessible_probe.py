#!/usr/bin/env python3
"""accessible_probe.py — 시간 일반화 격차(+302.5) 중 '합법 접근 가능분'을 분리한다.

배경 (outputs/507)
------------------
피처를 고정한 채 라벨만 동시대(2024)로 바꾸면 skill 이 738.8 → 1041.3 으로 +302.5 뛴다.
그러나 이 이득은 성격이 다른 두 갈래가 섞여 있다.

  (1) 투수 암기   — 모델이 (asof_n, asof_rate, 팀, 손, 구종비율) 좌표로 개별 투수를
                   사실상 식별하고, 그 투수의 **당해 성적을 2024 라벨로부터 외운다**.
                   테스트 시점에는 2025 라벨이 없으므로 **구조적으로 불가능**.
  (2) 매핑 학습   — "asof_rate 를 이번 시즌엔 얼마나 신뢰해야 하는가" 같은 **모집단
                   수준 함수**를 배운다. 저차원이므로 과거 시즌 추세로 근사할 여지가 있다.

(1)이 전부라면 이 축은 닫힌 것이고 v50 확정을 권고해야 한다.
(2)가 유의미하게 남으면 다음 가설은 그 함수의 외삽으로 좁혀진다.

설계 — 조작 변수는 '학습 행의 출처' 하나뿐
-----------------------------------------
피처 행렬은 네 조건 모두 완전히 동일하게 생성한다(prep/dec 를 season<2024 로 fit).

  A          과거 전체(<=2023) 라벨 학습            [실전 조건, 캐시 재사용]
  A_recent   직전 시즌(2023) 라벨만 학습            [가장 가까운 시즌으로 얻을 수 있는 몫]
  B_random   2024 라벨, 무작위 5-fold 교차적합      [(1)+(2) 합계 상한]
  B_grouped  2024 라벨, **투수 분리** 5-fold 교차적합 [(2)만: val 투수의 라벨을 일절 못 봄]

판정
----
  B_grouped ~= A          → 이득은 전부 투수 암기. 축 닫힘. v50 확정 권고.
  B_grouped >> A          → 모집단 매핑에 실재하는 여지. 외삽 가능성 탐색으로 진행.
  A_recent 가 A 를 크게 넘음 → 매핑이 해마다 변하고 최근 시즌이 더 대표적이라는 뜻.

모든 비교는 2모수 선형 재보정 후 값으로 한다. 안 하면 base-rate 이동이 격차에 섞인다.
※ B_* 는 진단 전용이다. 동시대 라벨은 실전에 존재하지 않으므로 제출에 쓸 수 없다.
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

CACHE = os.path.join(LG, 'harness/cache')
YEAR = 2024
NFOLD = 5
LGB_P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
             seed=7, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8)


def skill(p, y):
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))


def recal_skill(p, y):
    """레벨·스케일을 2모수로 흡수한 뒤의 skill = 순수 해상도."""
    b = np.cov(p, y)[0, 1] / p.var()
    return skill(np.clip(y.mean() + b * (p - p.mean()), 1e-6, 1 - 1e-6), y)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    tr_all = df[df.season < YEAR]
    va = df[df.season == YEAR].reset_index(drop=True)
    y = va.control_success.values.astype(np.float64)

    # ---- 모든 조건이 공유하는 피처 생성기 (season<2024 로만 fit) ----
    prep = bc.PitchPreprocessor()
    prep.fit(tr_all, as_of_season=YEAR - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    bs = ((tr_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cs = (tr_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          tr_all['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
    dec = bc.AsofDecomposer2(); dec.fit(tr_all, val_season=YEAR)

    Xva, _ = bc.build_features(va, prep, dec, cat_map)
    print(f'[피처] val {Xva.shape}  ({time.time()-t0:.0f}s)', flush=True)

    res = {}

    # ---- A : 캐시된 과거학습 lgb_bin ----
    seeds = [s for s in (7, 123, 2025)
             if os.path.exists(os.path.join(CACHE, f'pred_{YEAR}_{s}.npz'))]
    pa = np.mean([np.load(os.path.join(CACHE, f'pred_{YEAR}_{s}.npz'))['lgb_bin']
                  for s in seeds], axis=0)
    res['A  과거전체(<=2023)'] = pa

    # ---- A_recent : 직전 시즌만 ----
    r23 = df[df.season == YEAR - 1]
    X23, _ = bc.build_features(r23, prep, dec, cat_map)
    y23 = r23['control_success'].values.astype(np.float64)
    m = lgb.train(LGB_P, lgb.Dataset(X23, label=y23))
    res['A_recent 직전시즌(2023)만'] = m.predict(Xva)
    print(f'[A_recent] 학습행 {len(r23):,}  ({time.time()-t0:.0f}s)', flush=True)

    # ---- B_random : 2024 라벨, 무작위 5-fold ----
    rng = np.random.RandomState(0)
    fold = rng.randint(0, NFOLD, len(va))
    oof = np.zeros(len(va))
    for k in range(NFOLD):
        tm = fold != k
        oof[~tm] = lgb.train(LGB_P, lgb.Dataset(Xva[tm], label=y[tm])).predict(Xva[~tm])
    res['B_random 2024 무작위분할'] = oof
    print(f'[B_random] 완료 ({time.time()-t0:.0f}s)', flush=True)

    # ---- B_grouped : 2024 라벨, 투수 분리 5-fold ----
    groups = va.pitcher_id.values
    oof_g = np.zeros(len(va))
    for tr_i, va_i in GroupKFold(n_splits=NFOLD).split(Xva, y, groups):
        oof_g[va_i] = lgb.train(LGB_P, lgb.Dataset(Xva.iloc[tr_i], label=y[tr_i])
                                ).predict(Xva.iloc[va_i])
    res['B_grouped 2024 투수분리'] = oof_g
    n_bg = int(len(va) * (NFOLD - 1) / NFOLD)
    print(f'[B_grouped] 완료, 투수 {len(np.unique(groups))}명, 학습행 ~{n_bg:,} '
          f'({time.time()-t0:.0f}s)', flush=True)

    # ---- A_sub : 표본크기 통제군. B_grouped 와 같은 행 수를 과거에서 무작위 추출 ----
    # B_* 는 학습행이 20만인데 A 는 120만이다. 이 크기 차이를 격차로 오독하지 않도록
    # 과거 데이터를 같은 크기로 잘라 비교한다.
    sub = tr_all.sample(n=min(n_bg, len(tr_all)), random_state=0)
    Xsub, _ = bc.build_features(sub, prep, dec, cat_map)
    ysub = sub['control_success'].values.astype(np.float64)
    res[f'A_sub 과거 {n_bg//1000}k행만'] = lgb.train(
        LGB_P, lgb.Dataset(Xsub, label=ysub)).predict(Xva)
    print(f'[A_sub] 학습행 {len(sub):,} ({time.time()-t0:.0f}s)', flush=True)

    # ---- 결과 ----
    print('\n' + '=' * 70)
    print(f'  {"조건":30s} {"raw":>9s} {"선형재보정후":>12s} {"A대비":>9s}')
    base = recal_skill(pa, y)
    for k, p in res.items():
        r = recal_skill(p, y)
        print(f'  {k:30s} {skill(p, y):9.1f} {r:12.1f} {r-base:+9.1f}')
    print('=' * 70)

    bg = recal_skill(res['B_grouped 2024 투수분리'], y)
    br = recal_skill(res['B_random 2024 무작위분할'], y)
    asub = recal_skill(res[f'A_sub 과거 {n_bg//1000}k행만'], y)
    print(f'\n  전체 시간일반화 격차 (B_random − A)        = {br-base:+.1f}')
    print(f'  그중 투수 암기 몫  (B_random − B_grouped)  = {br-bg:+.1f}  <- 구조적으로 불가능')
    print(f'\n  [표본크기 통제] 같은 ~{n_bg//1000}k행 학습끼리 비교:')
    print(f'    B_grouped 동시대 라벨 = {bg:.1f}')
    print(f'    A_sub     과거 라벨   = {asub:.1f}')
    print(f'    동시대 프리미엄       = {bg-asub:+.1f}   <- 접근 가능 후보의 정직한 추정')
    if bg - asub < 30:
        print('\n  → 판정: 표본크기를 맞추면 동시대 라벨의 이점이 없다.')
        print('     +302.5는 전부 val 투수 자신의 라벨을 본 데서 나온 것이며,')
        print('     테스트 시점에 존재하지 않는다. 시간 일반화 축은 닫혔다.')
    else:
        print('\n  → 판정: 표본크기를 통제해도 동시대 프리미엄이 남는다.')
        print('     모집단 매핑의 외삽 가능성 탐색으로 진행.')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
