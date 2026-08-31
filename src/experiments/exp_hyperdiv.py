#!/usr/bin/env python3
"""exp_hyperdiv.py — pool_hyper 를 '풀 파이프라인'에서 확인한다.

배경 (outputs/510 0절)
---------------------
LGB 단독 진단에서 하이퍼파라미터 다양성이 사전 기준을 크게 통과했다:

    pool_hyper  15셀 평균 +33.6  sd 16.3  SE 4.2  t=7.98  양수 15/15
                폴드별 2021 +34.2 / 2022 +50.2 / 2023 +16.5 (전부 5/5)

반면 시즌전문가(pool_eq)는 +23.2 지만 t=1.65, 2023 폴드 0/5 로 미달.
→ **효과의 정체는 시즌 구조가 아니라 평범한 앙상블 다양성이다.**

그런데 그 진단은 **LGB 하나** 위에서 돌았다. 프로덕션은 이미 lgb_bin/cb_bin/xgb_bin/
lgb_mse/mlp 5성분을 블렌딩하므로 다양성이 이미 상당하다. 이미 다양한 앙상블에 다양성을
더하면 수확체감이 크다 — **풀 파이프라인에서 사라질 가능성이 실재한다.** 그래서 확인한다.

설계 — 조작 변수 하나
--------------------
기존 베이스라인 캐시에서 **`lgb_bin` 만** 다음으로 교체하고 나머지 4성분·가중치·
캘리브레이션은 바이트 그대로 재사용한다:

    lgb_bin  ->  0.5 * lgb_bin(현행 하이퍼) + 0.5 * lgb_alt(다른 하이퍼)

이러면 `evaluate.predict` 를 손대지 않아도 되고, 블렌드 구조·가중치가 전혀 안 바뀐다.
추가 학습은 (폴드, 시드)당 ALT LGB 한 개뿐이라 비용도 작다.

판정 (사전 확정)
---------------
inner 3폴드 x 5시드, 프로덕션과 동일한 예측 배깅 채점, 짝지은 15셀.
**3폴드 전부 양수 + t > 2.5.** 결과를 보고 기준을 바꾸지 않는다.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

BASE = os.path.join(LG, 'harness/cache')
OUT = os.path.join(LG, 'harness/cache_hyperdiv')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]

# 현행 lgb_bin 하이퍼 (build_cache.py 와 동일)
CUR_P = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
             verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8)
# 다양성용 대안 하이퍼 (exp_pooleq.py 의 ALT_P 와 동일 — 진단과 같은 설정을 옮긴다)
ALT_P = dict(CUR_P, learning_rate=0.03, num_leaves=63, min_child_samples=200,
             n_estimators=400, colsample_bytree=0.6)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    for y in FOLDS:
        past = df[df.season < y]
        va = df[df.season == y].reset_index(drop=True)
        ypa = past['control_success'].values.astype(np.float64)

        prep = bc.PitchPreprocessor()
        prep.fit(past, as_of_season=y - 1, is_final=False,
                 trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
        bs = ((past['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (past['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (past['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        cs = (past['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              past['strikes_before'].fillna(0).astype(int).astype(str))
        cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
        dec = bc.AsofDecomposer2(); dec.fit(past, val_season=y)
        Xpa, _ = bc.build_features(past, prep, dec, cat_map)
        Xva, _ = bc.build_features(va, prep, dec, cat_map)
        print(f'\n=== eval {y}: past {len(past):,} ({time.time()-t0:.0f}s) ===', flush=True)

        for sd in SEEDS:
            dst = os.path.join(OUT, f'pred_{y}_{sd}.npz')
            if os.path.exists(dst):
                print(f'  seed {sd}: cached, skip', flush=True)
                continue
            src = dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz')))
            alt = lgb.train(dict(ALT_P, seed=sd),
                            lgb.Dataset(Xpa, label=ypa)).predict(Xva)
            out = dict(src)
            out['lgb_bin'] = 0.5 * src['lgb_bin'] + 0.5 * alt   # 유일한 변경점
            np.savez_compressed(dst, **out)
            print(f'  seed {sd}: alt LGB 학습+저장 ({time.time()-t0:.0f}s)', flush=True)

    # ---- 채점 ----
    from evaluate import PROD, predict, skill
    print('\n[채점] 프로덕션 배깅 + 짝지은 15셀', flush=True)
    print(f'  {"fold":>6} {"seed":>9} {"현행":>9} {"다양성":>9} {"델타":>9}')
    cells, per_fold, bag = [], {}, {}
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
        ds, pa, pb = [], [], []
        for sd in SEEDS:
            a = predict(dict(PROD), dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz'))))
            b = predict(dict(PROD), dict(np.load(os.path.join(OUT, f'pred_{y}_{sd}.npz'))))
            ka, kb = skill(a, yv), skill(b, yv)
            ds.append(kb - ka); pa.append(a); pb.append(b)
            print(f'  {y:>6} {sd:>9} {ka:9.1f} {kb:9.1f} {kb-ka:+9.1f}')
        per_fold[y] = ds; cells += ds
        bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)

    d = np.array(cells)
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se
    print('\n' + '=' * 60)
    for y in FOLDS:
        v = np.array(per_fold[y])
        print(f'  {y}: 시드평균 {v.mean():+7.1f} 양수 {(v>0).sum()}/5   '
              f'배깅 델타 {bag[y]:+7.1f}')
    print(f'\n  15셀 평균 {d.mean():+.1f}  sd {d.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(d>0).sum()}/15')
    ok = all(np.mean(per_fold[y]) > 0 for y in FOLDS) and t > 2.5
    bag_ok = all(v > 0 for v in bag.values())
    print(f'  배깅 기준 3폴드 전부 양수: {bag_ok}  (평균 {np.mean(list(bag.values())):+.1f})')
    print(f'  → 사전기준(3폴드 전부 양수 + t>2.5) {"충족 ✅" if ok else "미달"}')
    print(f'  ※ LB 노이즈 바닥 ±12점 대비 배깅 평균 '
          f'{np.mean(list(bag.values())):+.1f}점')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
