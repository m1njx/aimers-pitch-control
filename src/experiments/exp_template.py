#!/usr/bin/env python3
"""exp_template.py — 새 가설을 프로토콜대로 검증하는 템플릿. 복사해서 쓴다.

    cp harness/exp_template.py harness/exp_myidea.py
    # 아래 [작성] 두 군데만 채우고
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE venv311/bin/python3 -u harness/exp_myidea.py --tag myidea

왜 템플릿인가
-------------
2026-08-24~25 세션에서 판정을 6번 틀렸고, 원인은 매번 프로토콜을 다시 짜면서 한두
군데를 빠뜨린 것이었다. 여기엔 그때 확립한 규칙이 전부 박혀 있다(`outputs/511` 3절):

  - inner **3폴드(2021/2022/2023) x 5시드**, 짝지은 (폴드,시드) 셀
  - 채점은 **프로덕션과 동일한 예측 배깅** (시드별 skill 평균과 부호가 갈릴 수 있다)
  - 판정 기준 **사전 확정**: 3폴드 전부 양수 + t > 2.5
  - 성분 개입이면 **유효 가중치 상한**을 먼저 경고

두 가지 모드
-----------
  MODE='features'   피처 행렬을 바꾼다. 전 성분 재학습 (3폴드x5시드 약 40~50분)
  MODE='component'  성분 예측 하나만 바꾼다. 나머지는 캐시 재사용 (약 5~25분)

component 모드가 가능하면 그쪽을 쓴다 — 훨씬 싸고 조작 변수도 더 깨끗하다.
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

BASE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]

# v50 기준 성분별 유효 가중치. 성분 개입의 이득 상한 = 단독효과 x 유효가중치.
# 상한이 LB 노이즈 바닥(12점)에 못 미치면 실행할 가치가 없다.
EFFECTIVE_W = {'mlp': 0.50, 'cb_bin': 0.29, 'lgb_mse': 0.25,
               'lgb_bin': 0.08, 'xgb_bin': 0.03}

# ===========================================================================
# [작성 1] 무엇을 검증하는가 — 한 문단으로 적는다.
#   반드시 포함: (a) 어떤 측정에서 나온 가설인가, (b) 이미 닫힌 어떤 축과 다른가,
#   (c) 규정4(행 독립성) 저촉이 없는 이유.
#   단일 폴드 관찰에서 출발한 가설은 착수 전에 3폴드로 관찰부터 재확인할 것.
# ===========================================================================
HYPOTHESIS = """
(여기에 가설을 적는다)
"""

MODE = 'features'          # 'features' 또는 'component'
COMPONENT = 'mlp'          # MODE='component' 일 때 바꿀 성분

_orig = bc.build_features


def patched_features(df, prep, dec, cat_map):
    """[작성 2-A] MODE='features' 일 때. X(119)와 X133 을 바꿔 반환한다.

    주의: X 는 GBDT 분류 3종이, X133 은 lgb_mse 와 mlp 가 쓴다. 보통 둘 다 바꿔야 한다.
    CAT_COLS 에 있는 컬럼을 지우면 cast_cb/cast_xgb 가 깨지니 확인할 것.
    """
    X, X133 = _orig(df, prep, dec, cat_map)
    # 예) X = X.drop(columns=[...]);  X133 = X133.assign(new_feat=...)
    return X, X133


def new_component(y, seed, Xpa, ypa, Xva, Xpa133, Xva133, src):
    """[작성 2-B] MODE='component' 일 때. 교체할 성분의 val 예측을 반환한다.

    src 는 베이스라인 캐시 dict (lgb_bin/cb_bin/xgb_bin/lgb_mse/mlp 포함).
    예) 대안 모델을 학습해 0.5*src[COMPONENT] + 0.5*alt 를 반환.
    """
    raise NotImplementedError('component 모드를 쓰려면 이 함수를 채울 것')


# ===========================================================================
# 이하 수정 불필요 — 프로토콜 구현부
# ===========================================================================
def fold_data(df, y):
    past = df[df.season < y]
    va = df[df.season == y].reset_index(drop=True)
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
    return past, va, prep, dec, cat_map


def score(cache_dir):
    """프로덕션과 동일한 예측 배깅 + 짝지은 15셀."""
    from evaluate import PROD, predict, skill
    cells, per_fold, bag = [], {}, {}
    print(f'\n  {"fold":>6} {"seed":>9} {"기준":>9} {"변형":>9} {"델타":>9}')
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
        d, pa, pb = [], [], []
        for s in SEEDS:
            fa = os.path.join(BASE, f'pred_{y}_{s}.npz')
            fb = os.path.join(cache_dir, f'pred_{y}_{s}.npz')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            a = predict(dict(PROD), dict(np.load(fa)))
            b = predict(dict(PROD), dict(np.load(fb)))
            ka, kb = skill(a, yv), skill(b, yv)
            d.append(kb - ka); pa.append(a); pb.append(b)
            print(f'  {y:>6} {s:>9} {ka:9.1f} {kb:9.1f} {kb-ka:+9.1f}')
        if not d:
            continue
        per_fold[y] = d; cells += d
        bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)

    dd = np.array(cells)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    t = dd.mean() / se
    print('\n' + '=' * 64)
    for y in FOLDS:
        if y in per_fold:
            v = np.array(per_fold[y])
            print(f'  {y}: 시드평균 {v.mean():+8.1f}  양수 {(v>0).sum()}/{len(v)}   '
                  f'배깅 {bag[y]:+8.1f}')
    print(f'\n  {len(dd)}셀 평균 {dd.mean():+.1f}  sd {dd.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(dd>0).sum()}/{len(dd)}')
    bag_mean = float(np.mean(list(bag.values())))
    print(f'  배깅 3폴드 평균 {bag_mean:+.1f}  전부 양수 {all(v > 0 for v in bag.values())}')
    ok = all(np.mean(per_fold[y]) > 0 for y in per_fold) and t > 2.5
    print(f'\n  → 사전기준(3폴드 전부 양수 + t>2.5) {"충족 ✅" if ok else "미달"}')
    print(f'  → LB 노이즈 바닥 ±12점 대비 배깅 {bag_mean:+.1f}점')
    if ok and bag_mean < 12:
        print('  ⚠️ 통계적으로는 유의하나 크기가 LB 노이즈 바닥 미만이다. 제출 가치 재검토.')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='캐시 디렉토리 이름 (harness/cache_<tag>)')
    ap.add_argument('--years', type=int, nargs='+', default=FOLDS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    a = ap.parse_args()

    print(HYPOTHESIS.strip())
    if MODE == 'component':
        w = EFFECTIVE_W.get(COMPONENT)
        print(f'\n⚠️ 유효 가중치 검사: {COMPONENT} = {w}')
        print(f'   단독 진단에서 +X 가 나와도 전체 이득 상한은 X x {w} 다.')
        print(f'   LB 노이즈 바닥(12점)을 넘으려면 단독 효과가 최소 '
              f'{12 / w:.0f}점은 나와야 한다.')

    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    cdir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cdir, exist_ok=True)

    if MODE == 'features':
        bc.build_features = patched_features
        bc.CACHE = cdir
        for y in a.years:
            bc.run_fold(df, y, a.seeds)
    else:
        for y in a.years:
            past, va, prep, dec, cat_map = fold_data(df, y)
            Xpa, Xpa133 = bc.build_features(past, prep, dec, cat_map)
            Xva, Xva133 = bc.build_features(va, prep, dec, cat_map)
            ypa = past['control_success'].values.astype(np.float64)
            print(f'\n=== eval {y}: past {len(past):,} ({time.time()-t0:.0f}s) ===',
                  flush=True)
            for s in a.seeds:
                dst = os.path.join(cdir, f'pred_{y}_{s}.npz')
                if os.path.exists(dst):
                    print(f'  seed {s}: cached, skip', flush=True)
                    continue
                src = dict(np.load(os.path.join(BASE, f'pred_{y}_{s}.npz')))
                out = dict(src)
                out[COMPONENT] = new_component(y, s, Xpa, ypa, Xva,
                                               Xpa133, Xva133, src)
                np.savez_compressed(dst, **out)
                print(f'  seed {s}: 완료 ({time.time()-t0:.0f}s)', flush=True)

    score(cdir)
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
