#!/usr/bin/env python3
"""exp_pooleq.py — pool_eq 재검증 (3폴드 x 5시드 짝지은 비교) + 다양성 대조군.

왜 재검증인가
-------------
`exp_experts.py` 에서 pool_eq(풀링 0.5 + 시즌전문가평균 0.5)가 pool 대비 +15.0,
양수 2/3 로 나왔다. 그러나 그 실행은 **LGB 시드 1개(seed=7)** 뿐이었다. 오늘 확립한
기준은 5시드이며([[505]]), 폴드축뿐 아니라 시드축도 막아야 한다([[507]]).
2/3 라는 결과 자체가 시드 잡음일 수 있으므로 짝지은 15셀로 다시 잰다.

대조군 — 효과가 '시즌 구조'인가 '단순 다양성'인가
------------------------------------------------
pool_eq 의 이득이 시즌별 전문화 때문인지, 아니면 그냥 서로 다른 모델을 섞어서 생기는
평범한 앙상블 다양성 때문인지 갈라야 한다. 그래서 **시즌을 전혀 쓰지 않는** 대조군을 둔다:

    pool_hyper = 0.5 * pool + 0.5 * (같은 데이터, 다른 하이퍼파라미터 LGB)

pool_hyper 가 pool_eq 만큼 오르면 시즌 구조는 무의미하고 그냥 다양성 효과다.
그 경우 프로덕션은 이미 5개 이질 성분을 블렌딩하므로 추가 이득을 기대하기 어렵다.

⚠️ 이 진단은 **LGB 단독** 위에서 돈다. 프로덕션은 이미 lgb_bin/cb/xgb/lgb_mse/mlp
5성분 앙상블이라 다양성이 이미 상당하다. 여기서 이득이 나와도 풀 파이프라인에서
사라질 수 있다 — 통과 시 반드시 풀 파이프라인으로 재확인할 것.

판정 (사전 확정, 결과 보고 조정 금지)
------------------------------------
3폴드 전부 평균 양수 + 짝지은 15셀 t > 2.5.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
BASE_P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
              verbose=-1, n_estimators=300, min_child_samples=50,
              subsample=0.8, colsample_bytree=0.8)
# 대조군용: 같은 데이터, 다른 형태의 모델 (시즌 정보 일절 사용 안 함)
ALT_P = dict(BASE_P, learning_rate=0.03, num_leaves=63, min_child_samples=200,
             n_estimators=400, colsample_bytree=0.6)


def skill(p, y):
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (y.mean() * (1 - y.mean())))


def recal(p, y):
    b = np.cov(p, y)[0, 1] / p.var()
    return skill(np.clip(y.mean() + b * (p - p.mean()), 1e-6, 1 - 1e-6), y)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    cells = []
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
        ts = sorted(np.unique(seas))
        print(f'\n=== eval {s}: past {len(past):,} / val {len(va):,} '
              f'전문가 {len(ts)}개 ({time.time()-t0:.0f}s) ===', flush=True)

        for sd in SEEDS:
            p_pool = lgb.train(dict(BASE_P, seed=sd), lgb.Dataset(Xpa, label=ypa)).predict(Xva)
            ex = [lgb.train(dict(BASE_P, seed=sd),
                            lgb.Dataset(Xpa[seas == t], label=ypa[seas == t])).predict(Xva)
                  for t in ts]
            eq = np.mean(ex, axis=0)
            p_alt = lgb.train(dict(ALT_P, seed=sd), lgb.Dataset(Xpa, label=ypa)).predict(Xva)

            k_pool = recal(p_pool, y)
            k_pe = recal(0.5 * p_pool + 0.5 * eq, y)
            k_ph = recal(0.5 * p_pool + 0.5 * p_alt, y)
            cells.append((s, sd, k_pool, k_pe, k_ph))
            print(f'  seed {sd:>8}: pool {k_pool:8.1f} | pool_eq {k_pe:8.1f} '
                  f'({k_pe-k_pool:+7.1f}) | pool_hyper {k_ph:8.1f} ({k_ph-k_pool:+7.1f})'
                  f'  [{time.time()-t0:.0f}s]', flush=True)

    print('\n' + '=' * 84)
    print(f'  {"fold":>6} {"pool_eq 델타":>14} {"양수":>6}   {"pool_hyper 델타":>16} {"양수":>6}')
    d_pe_all, d_ph_all = [], []
    for s in FOLDS:
        rows = [c for c in cells if c[0] == s]
        dpe = np.array([c[3] - c[2] for c in rows])
        dph = np.array([c[4] - c[2] for c in rows])
        d_pe_all += list(dpe); d_ph_all += list(dph)
        print(f'  {s:>6} {dpe.mean():+14.1f} {(dpe>0).sum():>4}/{len(dpe)}   '
              f'{dph.mean():+16.1f} {(dph>0).sum():>4}/{len(dph)}')
    print('=' * 84)

    for name, d in (('pool_eq', np.array(d_pe_all)), ('pool_hyper', np.array(d_ph_all))):
        se = d.std(ddof=1) / np.sqrt(len(d))
        fold_ok = all(np.mean([c[3 if name == 'pool_eq' else 4] - c[2]
                               for c in cells if c[0] == s]) > 0 for s in FOLDS)
        t = d.mean() / se
        print(f'\n  {name}: 15셀 평균 {d.mean():+.1f}  sd {d.std(ddof=1):.1f}  SE {se:.1f}  '
              f't={t:.2f}  양수 {(d>0).sum()}/{len(d)}')
        print(f'    → 사전기준(3폴드 전부 양수 + t>2.5) '
              f'{"충족" if fold_ok and t > 2.5 else "미달"}')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
