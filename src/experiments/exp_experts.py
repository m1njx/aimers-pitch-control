#!/usr/bin/env python3
"""exp_experts.py — 가설 S: 시즌 혼합이 손해라면, 시즌별 전문가를 앙상블한다.

측정 근거 (harness/accessible_probe.py, outputs/508)
---------------------------------------------------
2024 폴드에서 학습행 ~200k 로 맞춰 비교하면:

    A_recent  2023 단일시즌 245k       643.3
    B_grouped 2024 동시대 202k(투수분리) 623.7
    A_sub     과거혼합 202k            556.4
    A         과거전체 1.2M            738.8

**단일 시즌(643.3)이 같은 크기의 혼합(556.4)보다 +86.9 낫다.** 이는 동시대 프리미엄
(+67.3)보다도 크다. 즉 지금까지 "동시대의 값어치"로 해석한 것의 상당 부분이 실은
**"균질성의 값어치"** 일 수 있다. 그렇다면 과거 데이터만으로 접근 가능하다 — 규정 문제도,
평가시즌 라벨 문제도 없다.

가설
----
풀링 모델은 이질적인 5개 시즌의 조건부분포를 **하나의 평균 매핑**으로 뭉갠다. 시즌별로
따로 적합한 전문가(expert)들을 앙상블하면, 균질성의 이득을 유지하면서 데이터 양도
전부 쓸 수 있다.

era 계열(504 / Q / R)과 무엇이 다른가
------------------------------------
그쪽은 **한 모델이 시대를 조건으로 삼게** 만들려 했고 3번 다 실패했다(평가시즌의 체제를
과거에서 알 수 없으므로). 여기는 시대 정보를 쓰지 않는다. **모델 구조만 바꾼다** —
어느 전문가가 평가시즌에 맞는지 고르지 않고 전부 평균한다. 실패 원인이었던 "미래 체제
예측"이 아예 개입하지 않는다.

측정 항목
---------
  pool        전체 과거 풀링 (현행)
  season_t    시즌 t 단독 전문가 (개별 성적도 기록 → 균질성 vs 최근성 분리)
  eq          전문가 등가중 평균
  rec         전문가 최근성 가중 평균 (w ∝ 0.7^age)
  pool_eq     pool 과 eq 를 반반

  size_ctrl   pool 을 시즌 1개 크기로 subsample (균질성 효과에서 표본크기 교란 제거)

판정: `outputs/507/508` 규율대로 **3폴드(2021/2022/2023) 전부에서 pool 초과**여야 후보.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

LGB_P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
             seed=7, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8)
FOLDS = [2021, 2022, 2023]


def skill(p, y):
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))


def recal(p, y):
    """레벨·스케일을 2모수로 흡수한 뒤의 skill. 시즌간 베이스레이트 차이를 제거한다."""
    b = np.cov(p, y)[0, 1] / p.var()
    return skill(np.clip(y.mean() + b * (p - p.mean()), 1e-6, 1 - 1e-6), y)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    summary = {}
    for s in FOLDS:
        past = df[df.season < s]
        va = df[df.season == s].reset_index(drop=True)
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
        Xpa, _ = bc.build_features(past, prep, dec, cat_map)
        ypa = past['control_success'].values.astype(np.float64)
        seas = past['season'].values
        print(f'\n=== eval {s}: past {len(past):,} / val {len(va):,} '
              f'({time.time()-t0:.0f}s) ===', flush=True)

        # pool
        p_pool = lgb.train(LGB_P, lgb.Dataset(Xpa, label=ypa)).predict(Xva)

        # 시즌별 전문가
        experts, sizes = {}, {}
        for t in sorted(np.unique(seas)):
            m = seas == t
            experts[t] = lgb.train(LGB_P, lgb.Dataset(Xpa[m], label=ypa[m])).predict(Xva)
            sizes[t] = int(m.sum())
            print(f'  expert {t}: {sizes[t]:,}행  skill {recal(experts[t], y):.1f}', flush=True)

        ts = sorted(experts)
        eq = np.mean([experts[t] for t in ts], axis=0)
        w = np.array([0.7 ** (ts[-1] - t) for t in ts]); w = w / w.sum()
        rec = np.sum([wi * experts[t] for wi, t in zip(w, ts)], axis=0)
        pool_eq = 0.5 * p_pool + 0.5 * eq

        # 표본크기 통제: pool 을 시즌 1개 크기로 subsample
        n1 = int(np.mean(list(sizes.values())))
        idx = np.random.RandomState(0).choice(len(Xpa), size=min(n1, len(Xpa)), replace=False)
        p_ctrl = lgb.train(LGB_P, lgb.Dataset(Xpa.iloc[idx], label=ypa[idx])).predict(Xva)

        r = dict(pool=recal(p_pool, y), eq=recal(eq, y), rec=recal(rec, y),
                 pool_eq=recal(pool_eq, y), size_ctrl=recal(p_ctrl, y),
                 best_expert=max(recal(experts[t], y) for t in ts),
                 last_expert=recal(experts[ts[-1]], y), n1=n1)
        summary[s] = r
        print(f'  pool {r["pool"]:.1f} | eq {r["eq"]:.1f} | rec {r["rec"]:.1f} | '
              f'pool_eq {r["pool_eq"]:.1f} | size_ctrl({n1//1000}k) {r["size_ctrl"]:.1f}',
              flush=True)

    print('\n' + '=' * 88)
    print(f'  {"fold":>6} {"pool":>9} {"eq":>9} {"rec":>9} {"pool_eq":>9} '
          f'{"직전시즌단독":>12} {"size_ctrl":>10}')
    for s in FOLDS:
        r = summary[s]
        print(f'  {s:>6} {r["pool"]:9.1f} {r["eq"]:9.1f} {r["rec"]:9.1f} '
              f'{r["pool_eq"]:9.1f} {r["last_expert"]:12.1f} {r["size_ctrl"]:10.1f}')
    print('=' * 88)

    print(f'\n  {"방식":>12} {"pool 대비 평균":>14} {"양수 폴드":>10}')
    for k in ('eq', 'rec', 'pool_eq'):
        d = np.array([summary[s][k] - summary[s]['pool'] for s in FOLDS])
        print(f'  {k:>12} {d.mean():+14.1f} {(d>0).sum():>7}/3')

    print('\n[균질성 vs 표본크기] 같은 크기에서 단일시즌 vs 혼합')
    for s in FOLDS:
        r = summary[s]
        print(f'  {s}: 직전시즌단독 {r["last_expert"]:.1f}  vs  혼합 {r["size_ctrl"]:.1f}'
              f'  (차이 {r["last_expert"]-r["size_ctrl"]:+.1f})')
    d = np.array([summary[s]['last_expert'] - summary[s]['size_ctrl'] for s in FOLDS])
    print(f'  → 균질성 이득 평균 {d.mean():+.1f}  (양수 {(d>0).sum()}/3)')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
