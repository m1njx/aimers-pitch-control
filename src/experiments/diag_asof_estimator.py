#!/usr/bin/env python3
"""diag_asof_estimator.py — asof 채널의 추정기가 최적인가 (재학습 없는 선별 측정).

HANDOFF 🟡 두 항목을 한 번에 잰다.
  1. `AsofDecomposer2` 의 EB 수축 `eb_m=150` 이 정직한 하네스로 튜닝된 적이 없다.
  2. 추정기를 가정하지 말고 **학습**한다 — (자기 카운터) -> (실제 성공률) 매핑을
     train 에서 적합. 규정4 무관(자기 행 컬럼 + train 적합 고정표).

무엇을 재는가
------------
`cs_p_succ_eb` 는 (현시즌 누적 cur_rate, 과거 hist_rate) 를 고정 가중 m 으로 섞은
**손으로 가정한 추정기**다. 이걸 **단독 예측기로 직접 채점**하면 재학습 없이
  (a) m 의 최적값 곡선
  (b) 같은 입력으로 학습한 추정기가 EB 를 얼마나 넘는가 (= 가정의 손실)
를 잰다. 폴드당 decomposer 1회 적합이면 m 스윕은 공짜다.

⚠️ 이건 **선별 필터**이지 채택 근거가 아니다. 여기서 큰 여지가 나오면 그때
`exp_template.py` (MODE='features') 로 3폴드x5시드 풀 파이프라인을 돌린다.
(`outputs/513` 교훈: 단독 채널 이득은 프로덕션에서 소멸할 수 있다.)

    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/diag_asof_estimator.py
"""
import os, sys, time, gc, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'work/submit_v42'))
sys.path.insert(0, os.path.join(LG, 'harness'))
sys.path.insert(0, LG)

import lightgbm as lgb                                    # noqa: E402
from sklearn.model_selection import GroupKFold            # noqa: E402
from agent2_asof_decomp2 import AsofDecomposer2           # noqa: E402
from evaluate import skill                                # noqa: E402

FOLDS = [2021, 2022, 2023]
M_GRID = [0.0, 10, 25, 50, 100, 150, 250, 400, 700, 1200, 2500, 1e9]
USECOLS = ['season', 'pitcher_id', 'batter_id', 'control_success',
           'asof_pitcher_n', 'asof_pitcher_success_rate',
           'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
           'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
           'asof_pitcher_pitchmix_n', 'asof_pitcher_fastball_rate',
           'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
           'asof_batter_n', 'asof_batter_success_rate', 'asof_batter_middle_rate']

# 학습 추정기 입력 = decomposer 가 이미 내놓는 '자기 행' 카운터들만.
LEARN_COLS = ['cs_p_succ_rate', 'cs_p_succ_hist', 'cs_pit_cur_n', 'cs_pit_hist_n',
              'cs_b_succ_rate', 'cs_b_succ_hist', 'cs_bat_cur_n', 'cs_bat_hist_n']
LGB_P = dict(objective='regression', metric='rmse', learning_rate=0.05,
             num_leaves=31, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8, num_threads=2,
             deterministic=True, force_row_wise=True)


def eb(cur_rate, hist_rate, cur_n, m, fb):
    """AsofDecomposer2 와 문자 그대로 동일한 EB 식. m 만 바꿔 재계산한다."""
    cr = np.nan_to_num(cur_rate, nan=fb)
    hr = np.nan_to_num(hist_rate, nan=fb)
    return (np.nan_to_num(cur_n * cr) + m * hr) / (cur_n + m)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    df = df[USECOLS]
    print(f'train {df.shape}  ({time.time()-t0:.0f}s)', flush=True)

    sweep, learn = [], []
    for y in FOLDS:
        past = df[df.season < y].reset_index(drop=True)
        va = df[df.season == y].reset_index(drop=True)
        dec = AsofDecomposer2(); dec.fit(past, val_season=y)
        Xva = dec.transform(va).astype(np.float64)
        Xpa = dec.transform(past).astype(np.float64)
        yv = va['control_success'].values.astype(np.float64)
        yp = past['control_success'].values.astype(np.float64)
        fbp = dec.fallback_['p_succ']; fbb = dec.fallback_['b_succ']
        print(f'\n=== {y}: past {len(past):,}  va {len(va):,}  '
              f'cur_n 중앙값 p={np.nanmedian(Xva["cs_pit_cur_n"]):.0f} '
              f'b={np.nanmedian(Xva["cs_bat_cur_n"]):.0f}  ({time.time()-t0:.0f}s) ===',
              flush=True)

        # ---- (a) m 스윕: 투수 EB 단독 / 투수+타자 EB 평균 ----
        for m in M_GRID:
            p = eb(Xva['cs_p_succ_rate'].values, Xva['cs_p_succ_hist'].values,
                   Xva['cs_pit_cur_n'].values, m, fbp)
            b = eb(Xva['cs_b_succ_rate'].values, Xva['cs_b_succ_hist'].values,
                   Xva['cs_bat_cur_n'].values, m, fbb)
            sweep.append(dict(fold=y, m=m, k_pit=skill(np.clip(p, 1e-6, 1-1e-6), yv),
                              k_pb=skill(np.clip(0.5*(p+b), 1e-6, 1-1e-6), yv)))

        # ---- (b) 학습 추정기: 같은 입력으로 past 에서 적합 ----
        A = Xpa[LEARN_COLS].values.astype(np.float32)
        B = Xva[LEARN_COLS].values.astype(np.float32)
        pl = np.zeros(len(B))
        for ki, (tr, _) in enumerate(GroupKFold(n_splits=3).split(
                A, yp, past['pitcher_id'].values)):
            p_ = dict(LGB_P); p_['seed'] = 7 + ki
            pl += lgb.train(p_, lgb.Dataset(A[tr], label=yp[tr])).predict(B) / 3.0
        k_learn = skill(np.clip(pl, 1e-6, 1-1e-6), yv)

        s = pd.DataFrame([r for r in sweep if r['fold'] == y])
        best = s.loc[s.k_pb.idxmax()]
        cur = s[s.m == 150].iloc[0]
        learn.append(dict(fold=y, k_learn=k_learn, k_m150=cur.k_pb,
                          k_best=best.k_pb, m_best=best.m))
        print(f'  m=150 (현행) {cur.k_pb:8.1f} | 최적 m={best.m:<8.0f} {best.k_pb:8.1f} '
              f'({best.k_pb-cur.k_pb:+.1f}) | 학습 추정기 {k_learn:8.1f} '
              f'({k_learn-cur.k_pb:+.1f})', flush=True)
        del past, va, Xva, Xpa, A, B, dec; gc.collect()

    S = pd.DataFrame(sweep)
    print('\n' + '=' * 84)
    print('EB 수축 m 스윕 — 투수+타자 EB 평균의 단독 skill')
    print(f'{"m":>10} ' + ' '.join(f'{y:>10}' for y in FOLDS) + f'{"평균":>10} {"vs m=150":>10}')
    base = {y: S[(S.fold == y) & (S.m == 150)].k_pb.iloc[0] for y in FOLDS}
    for m in M_GRID:
        v = [S[(S.fold == y) & (S.m == m)].k_pb.iloc[0] for y in FOLDS]
        d = np.mean([v[i] - base[y] for i, y in enumerate(FOLDS)])
        star = '  <- 현행' if m == 150 else ''
        print(f'{m:>10.0f} ' + ' '.join(f'{x:10.1f}' for x in v) +
              f'{np.mean(v):10.1f} {d:+10.1f}{star}')

    L = pd.DataFrame(learn)
    print('\n' + '=' * 84)
    print(L.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    dl = (L.k_learn - L.k_m150).values
    db = (L.k_best - L.k_m150).values
    print(f'\n학습 추정기 − 현행(m=150): 폴드별 {np.round(dl,1)}  평균 {dl.mean():+.1f}  '
          f'양수 {(dl>0).sum()}/3')
    print(f'최적 m − 현행(m=150)     : 폴드별 {np.round(db,1)}  평균 {db.mean():+.1f}  '
          f'최적 m {L.m_best.tolist()}')
    print('\n해석 가이드')
    print('  · 최적 m 이 폴드마다 갈리면 m 은 튜닝 불가(이전 안 됨) -> 축 닫힘.')
    print('  · 학습 추정기가 3폴드 전부 크게 이기면 "가정된 EB" 가 손실 원인 -> 풀 파이프라인行.')
    print('  · 단, 이 채널 단독 이득은 프로덕션 유효가중치에 눌린다(outputs/513).')
    S.to_csv(os.path.join(LG, 'harness/diag_asof_estimator_sweep.csv'), index=False)
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
