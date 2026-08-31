#!/usr/bin/env python3
"""exp_seasonfeat.py — 가설 R: `season` 을 피처로 넣어 최신 시즌 체제로 특화시킨다.

측정 근거 (harness/accessible_probe.py)
--------------------------------------
학습 행 수를 ~202k 로 맞춰 비교하면:

    B_grouped  평가시즌(2024) 라벨, 투수분리   623.7
    A_sub      과거 라벨 무작위 202k           556.4
    → 동시대 프리미엄 +67.3  (투수 암기가 아니다. 투수가 분리돼 있다)

즉 **모집단 수준에서 '평가 시즌 체제'로 학습하는 것 자체에 +67.3점의 값어치**가 있다.
또 A_recent(2023만, 245k) 643.3 > A_sub(과거혼합 202k) 556.4 이므로, 행당 가치는
최신 시즌이 확실히 높다. 다만 A(전체 1.2M) 738.8 이 여전히 최고라 데이터 양이 이긴다.

가설
----
현재 `season` 은 `config.MODEL_FEATURE_COLS` 화이트리스트에서 제외돼 있어 **모델은 이
행이 몇 년도인지 모른다.** 그래서 5개 시즌의 서로 다른 체제를 하나로 평균낸 매핑만
배운다. `season` 을 수치 피처로 넣으면 트리가 시즌으로 분기할 수 있고, 평가 시즌 행은
학습 최대 시즌보다 크므로 **자동으로 '가장 최근 시즌' 가지로 배정**된다.
→ 데이터 양(1.2M)은 그대로 쓰면서 예측은 최신 체제로 특화된다. A_recent 의 최신성과
A 의 데이터 양을 동시에 갖는 구조다.

504/507 과 무엇이 다른가
------------------------
- `504` era-offset: 라벨/예측 레벨에 시즌 오프셋. 기각.
- `507` 가설 Q: asof 비율을 시즌 리그수준 대비로 재표현. 3폴드에서 기각(-20.4).
- 여기: **시즌 자체를 분기 가능한 피처로 제공.** 어떤 변수를 어떻게 쓸지는 모델이 정한다.
  앞의 둘은 보정 형태를 사람이 고정했지만 이건 형태를 강제하지 않는다.

규정
----
`season` 은 각 행 자신의 속성이다. test.csv 의 다른 행이나 배치 통계를 일절 참조하지
않는다. 규정4 무관.

프로토콜
--------
`outputs/507` 교훈에 따라 **inner 3폴드(2021/2022/2023) x 5시드**로 판정한다.
2폴드로는 폴드 특유의 요행을 못 걸러낸다(가설 Q가 정확히 그렇게 통과할 뻔했다).

    export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 -u harness/exp_seasonfeat.py
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

_orig = bc.build_features


def patched(df, prep, dec, cat_map):
    X, X133 = _orig(df, prep, dec, cat_map)
    s = df['season'].astype(np.float32).values
    X = X.copy(); X['season_idx'] = s
    X133 = X133.copy(); X133['season_idx'] = s
    return X, X133


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2021, 2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025, 31415, 8675309])
    ap.add_argument('--tag', default='seasonfeat')
    a = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    cache_dir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cache_dir, exist_ok=True)
    bc.build_features = patched
    bc.CACHE = cache_dir

    for y in a.years:
        bc.run_fold(df, y, a.seeds)

    print('\n[채점] inner 3폴드 x 5시드, 짝지은 셀 비교', flush=True)
    from evaluate import PROD, predict as bp, skill
    base_dir = os.path.join(LG, 'harness/cache')
    print(f'  {"year":>6} {"seed":>9} {"R0":>9} {"R1":>9} {"델타":>9}')
    per_year, allc = {}, []
    for y in a.years:
        yv = np.load(os.path.join(base_dir, f'y_{y}.npy'))
        ds = []
        for s in a.seeds:
            fa = os.path.join(base_dir, f'pred_{y}_{s}.npz')
            fb = os.path.join(cache_dir, f'pred_{y}_{s}.npz')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            ka = skill(bp(dict(PROD), dict(np.load(fa))), yv)
            kb = skill(bp(dict(PROD), dict(np.load(fb))), yv)
            ds.append(kb - ka)
            print(f'  {y:>6} {s:>9} {ka:9.1f} {kb:9.1f} {kb-ka:+9.1f}')
        per_year[y] = ds
        allc += ds
    print()
    for y, ds in per_year.items():
        if ds:
            print(f'  {y}: 평균 {np.mean(ds):+7.1f}  양수 {sum(1 for v in ds if v>0)}/{len(ds)}')
    d = np.array(allc)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'\n  전체 {len(d)}셀: 평균 {d.mean():+.1f}  sd {d.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={d.mean()/se:.2f}  양수 {(d>0).sum()}/{len(d)}')
    ok = d.mean() > 0 and d.mean() / se > 2.5 and all(np.mean(v) > 0 for v in per_year.values() if v)
    print(f'  → 판정 {"ACCEPT 후보 (모든 폴드 양수 + t>2.5)" if ok else "미달"}')
    print(f'총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
