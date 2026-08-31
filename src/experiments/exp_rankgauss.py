#!/usr/bin/env python3
"""exp_rankgauss.py — 가설 T: MLP 입력을 분위수 변환(rank-gauss)한다.

가설
----
`build_cache.mlp_arrays` 는 수치 입력에 단순 z-score 만 적용한다:
`nz = nan_to_num((X133[num] - mean) / std)`. 그런데 입력에는 `asof_pitcher_n`,
`li`, `score_diff_pitcher_team`, `tkm_n_pitches` 처럼 **꼬리가 두꺼운** 피처가 섞여
있어 표준화 후에도 극단값이 첫 레이어의 가중치 갱신을 지배한다.

트리 모델은 단조변환에 불변이라 이 문제가 없지만 **신경망은 다르다.** 분위수 변환으로
각 피처를 정규분포로 사상하면 스케일이 균질해져 학습이 안정된다(tabular NN 의 표준 기법).

사전 점검 (`outputs/511` 3-3)
----------------------------
  유효 가중치      mlp = 0.50 — 5성분 중 최고. 상한 = 단독효과 x 0.50.
                  LB 노이즈 바닥(12)을 넘으려면 단독 +24점 필요.
  단일폴드 기반?   아니오. 관찰이 아니라 표준 기법에서 나온 가설.
  닫힌 축 중복?    아니오. MLP 용량(M1/M2/M3)과 아키텍처 다양성은 닫혔으나
                  **입력 표현은 한 번도 건드리지 않았다.**

설계 — 조작 변수 하나
--------------------
베이스라인 캐시에서 **`mlp` 만** 교체한다. 나머지 4성분·블렌드 가중치·캘리브레이션은
바이트 그대로 재사용. MLP 아키텍처·하이퍼파라미터·시드·에폭 전부 현행과 동일하고
**입력 변환만** 바뀐다.

  현행: (x - mean) / std
  변형: QuantileTransformer(output_distribution='normal') — train 에서만 fit

범주형 임베딩 경로는 손대지 않는다. 변환기는 폴드의 train 으로만 적합하고 val 에
그대로 적용하므로 누수 없음. 각 행은 자기 입력만 쓰므로 규정4 무관.

판정 (사전 확정, 결과 보고 변경 금지)
-----------------------------------
inner 3폴드(2021/2022/2023) x 5시드, 프로덕션 배깅 채점, 짝지은 15셀.
**3폴드 전부 양수 + t > 2.5**, 그리고 배깅 평균이 LB 노이즈 바닥(12점)을 넘을 것.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

BASE = os.path.join(LG, 'harness/cache')
OUT = os.path.join(LG, 'harness/cache_rankgauss')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def qt_arrays(Xtr133, Xva133):
    """현행 mlp_arrays 와 동일하되 수치 표준화만 분위수 변환으로 교체."""
    num = [c for c in Xtr133.columns if c not in bc.CAT_COLS]
    cat = [c for c in Xtr133.columns if c in bc.CAT_COLS]

    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                             subsample=200000, random_state=0)
    tr_num = np.nan_to_num(Xtr133[num].values.astype(np.float32), nan=0.0)
    va_num = np.nan_to_num(Xva133[num].values.astype(np.float32), nan=0.0)
    nz_tr = qt.fit_transform(tr_num).astype(np.float32)
    nz_va = qt.transform(va_num).astype(np.float32)
    nz_tr = np.nan_to_num(nz_tr, nan=0.0, posinf=0.0, neginf=0.0)
    nz_va = np.nan_to_num(nz_va, nan=0.0, posinf=0.0, neginf=0.0)

    vocabs = {c: {v: i for i, v in enumerate(Xtr133[c].astype(str).unique())} for c in cat}
    cards = [len(vocabs[c]) + 1 for c in cat]

    def codes(X):
        return np.stack([X[c].astype(str).map(vocabs[c]).fillna(len(vocabs[c]))
                         .astype(np.int64).values for c in cat], axis=1)

    return nz_tr, codes(Xtr133), nz_va, codes(Xva133), cards


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    print('유효 가중치 검사: mlp = 0.50 → 단독 +24점 이상이어야 LB 노이즈 바닥 통과\n',
          flush=True)

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
        _, Xpa133 = bc.build_features(past, prep, dec, cat_map)
        _, Xva133 = bc.build_features(va, prep, dec, cat_map)

        nz_tr, ca_tr, nz_va, ca_va, cards = qt_arrays(Xpa133, Xva133)
        print(f'\n=== eval {y}: past {len(past):,}  수치 {nz_tr.shape[1]} '
              f'범주 {ca_tr.shape[1]}  ({time.time()-t0:.0f}s) ===', flush=True)

        ds = torch.utils.data.TensorDataset(
            torch.tensor(nz_tr), torch.tensor(ca_tr),
            torch.tensor(ypa, dtype=torch.float32))

        for sd in SEEDS:
            dst = os.path.join(OUT, f'pred_{y}_{sd}.npz')
            if os.path.exists(dst):
                print(f'  seed {sd}: cached, skip', flush=True)
                continue
            t1 = time.time()
            # 아키텍처·하이퍼파라미터·에폭 전부 build_cache 현행과 동일
            torch.manual_seed(sd)
            net = bc.SimpleMLP_MSE(nz_tr.shape[1], cards)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
            crit = nn.MSELoss()
            dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
            net.train()
            for _ in range(5):
                for bn, bc_, by in dl:
                    opt.zero_grad(); crit(net(bn, bc_), by).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(torch.tensor(nz_va), torch.tensor(ca_va)).numpy().astype(np.float64)

            src = dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz')))
            out = dict(src)
            out['mlp'] = pred                      # 유일한 변경점
            np.savez_compressed(dst, **out)
            print(f'  seed {sd}: 완료 ({time.time()-t1:.0f}s)', flush=True)

    # ---- 채점 ----
    from evaluate import PROD, predict, skill
    print('\n[채점] 프로덕션 배깅 + 짝지은 15셀', flush=True)
    cells, per_fold, bag, solo = [], {}, {}, {}
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
        d, pa, pb, sa, sb = [], [], [], [], []
        for sd in SEEDS:
            A = dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz')))
            B = dict(np.load(os.path.join(OUT, f'pred_{y}_{sd}.npz')))
            a, b = predict(dict(PROD), A), predict(dict(PROD), B)
            ka, kb = skill(a, yv), skill(b, yv)
            d.append(kb - ka); pa.append(a); pb.append(b)
            sa.append(skill(np.clip(A['mlp'], 1e-6, 1-1e-6), yv))
            sb.append(skill(np.clip(B['mlp'], 1e-6, 1-1e-6), yv))
            print(f'  {y} {sd:>9}: {ka:8.1f} -> {kb:8.1f}  ({kb-ka:+7.1f})')
        per_fold[y] = d; cells += d
        bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)
        solo[y] = float(np.mean(sb) - np.mean(sa))

    dd = np.array(cells)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    t = dd.mean() / se
    print('\n' + '=' * 66)
    for y in FOLDS:
        v = np.array(per_fold[y])
        print(f'  {y}: 시드평균 {v.mean():+8.1f} 양수 {(v>0).sum()}/5   '
              f'배깅 {bag[y]:+8.1f}   [MLP 단독 {solo[y]:+8.1f}]')
    bm = float(np.mean(list(bag.values())))
    print(f'\n  15셀 평균 {dd.mean():+.1f}  sd {dd.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(dd>0).sum()}/15')
    print(f'  배깅 3폴드 평균 {bm:+.1f}  전부 양수 {all(v > 0 for v in bag.values())}')
    print(f'  MLP 단독 효과 평균 {np.mean(list(solo.values())):+.1f}  '
          f'(x0.50 = 예상 상한 {np.mean(list(solo.values()))*0.5:+.1f})')
    ok = all(np.mean(per_fold[y]) > 0 for y in FOLDS) and t > 2.5
    print(f'\n  → 사전기준(3폴드 전부 양수 + t>2.5) {"충족" if ok else "미달"}')
    print(f'  → LB 노이즈 바닥(12) 통과 {"예 ✅" if ok and bm > 12 else "아니오"}')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
