#!/usr/bin/env python3
"""exp_mlpdiv.py — 다양성 개입을 '유효 가중치가 가장 큰 성분'에 건다.

왜 MLP 인가 (outputs/511 의 교훈)
--------------------------------
LGB 단독 진단에서 하이퍼 다양성이 +33.6(t=7.98, 15/15)로 크게 통과했는데, 풀
파이프라인에 옮기자 **+0.3(t=1.12, 배깅 +0.1)로 소멸**했다. 원인은 명확하다:

    lgb_bin 의 유효 가중치 = W_LGB_BIN(0.20) x W_GBDT_BIN(0.40) ≈ **0.08**

8% 짜리 성분을 개선해봐야 전체에는 8% 만 반영되고, 게다가 나머지 성분이 이미 같은
분산을 덮고 있어 실제로는 그보다도 적게 남는다.

**따라서 성분 단위 개입은 유효 가중치를 곱해서 봐야 한다.** 이 렌즈로 보면 표적은
하나뿐이다:

    성분        유효 가중치        비고
    mlp         0.40~0.50        ← 가장 큼. lgb_bin 의 5~6배
    lgb_mse     0.20~0.25
    cb_bin      0.72 x 0.40 ≈ 0.29
    xgb_bin     0.08 x 0.40 ≈ 0.03
    lgb_bin     0.20 x 0.40 ≈ 0.08   ← 방금 실패한 표적

설계 — exp_hyperdiv 와 동일하게 조작 변수 하나
---------------------------------------------
베이스라인 캐시에서 **`mlp` 만** 교체하고 나머지 4성분·가중치·캘리브레이션은
바이트 그대로 재사용한다:

    mlp  ->  0.5 * mlp(현행 아키텍처) + 0.5 * mlp_alt(다른 아키텍처)

현행: hidden (128,64), dropout 0.12, lr 1e-3, wd 1e-5, 5 epoch
대안: hidden (256,128), dropout 0.22, lr 5e-4, wd 3e-5, 7 epoch

용량 스캔(M1/M2/M3)에서 용량 자체는 이득이 없었으므로([[505]]), 여기서 노리는 것은
용량이 아니라 **서로 다른 오차를 내는 두 모델의 평균**이다.

판정 (사전 확정): inner 3폴드 x 5시드, 프로덕션 배깅 채점, 짝지은 15셀,
**3폴드 전부 양수 + t > 2.5**. 결과 보고 기준 변경 금지.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

BASE = os.path.join(LG, 'harness/cache')
OUT = os.path.join(LG, 'harness/cache_mlpdiv')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]

ALT = dict(hidden=(256, 128), dropout=0.22, lr=5e-4, wd=3e-5, epochs=7)


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
        _, Xpa133 = bc.build_features(past, prep, dec, cat_map)
        _, Xva133 = bc.build_features(va, prep, dec, cat_map)
        nz_tr, ca_tr, art = bc.mlp_arrays(Xpa133)
        nz_va, ca_va, _ = bc.mlp_arrays(Xva133, art)
        print(f'\n=== eval {y}: past {len(past):,} ({time.time()-t0:.0f}s) ===', flush=True)

        ds = torch.utils.data.TensorDataset(
            torch.tensor(nz_tr), torch.tensor(ca_tr),
            torch.tensor(ypa, dtype=torch.float32))

        for sd in SEEDS:
            dst = os.path.join(OUT, f'pred_{y}_{sd}.npz')
            if os.path.exists(dst):
                print(f'  seed {sd}: cached, skip', flush=True)
                continue
            t1 = time.time()
            torch.manual_seed(sd + 999)
            net = bc.SimpleMLP_MSE(len(art['num_cols']), art['cards'],
                                   hidden=ALT['hidden'], dropout=ALT['dropout'])
            opt = torch.optim.Adam(net.parameters(), lr=ALT['lr'],
                                   weight_decay=ALT['wd'])
            crit = nn.MSELoss()
            dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
            net.train()
            for ep in range(ALT['epochs']):
                for bn, bc_, by in dl:
                    opt.zero_grad(); crit(net(bn, bc_), by).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                alt = net(torch.tensor(nz_va), torch.tensor(ca_va)).numpy().astype(np.float64)

            src = dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz')))
            out = dict(src)
            out['mlp'] = 0.5 * src['mlp'] + 0.5 * alt        # 유일한 변경점
            np.savez_compressed(dst, **out)
            print(f'  seed {sd}: ALT MLP 완료 ({time.time()-t1:.0f}s)', flush=True)

    from evaluate import PROD, predict, skill
    print('\n[채점] 프로덕션 배깅 + 짝지은 15셀', flush=True)
    cells, per_fold, bag = [], {}, {}
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
        d, pa, pb = [], [], []
        for sd in SEEDS:
            a = predict(dict(PROD), dict(np.load(os.path.join(BASE, f'pred_{y}_{sd}.npz'))))
            b = predict(dict(PROD), dict(np.load(os.path.join(OUT, f'pred_{y}_{sd}.npz'))))
            ka, kb = skill(a, yv), skill(b, yv)
            d.append(kb - ka); pa.append(a); pb.append(b)
            print(f'  {y} {sd:>9}: {ka:8.1f} -> {kb:8.1f}  ({kb-ka:+7.1f})')
        per_fold[y] = d; cells += d
        bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)

    dd = np.array(cells)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    t = dd.mean() / se
    print('\n' + '=' * 60)
    for y in FOLDS:
        v = np.array(per_fold[y])
        print(f'  {y}: 시드평균 {v.mean():+7.1f} 양수 {(v>0).sum()}/5   배깅 {bag[y]:+7.1f}')
    print(f'\n  15셀 평균 {dd.mean():+.1f}  sd {dd.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(dd>0).sum()}/15')
    ok = all(np.mean(per_fold[y]) > 0 for y in FOLDS) and t > 2.5
    print(f'  배깅 3폴드 평균 {np.mean(list(bag.values())):+.1f}  '
          f'전부 양수 {all(v > 0 for v in bag.values())}')
    print(f'  → 사전기준 {"충족 ✅" if ok else "미달"}   (LB 노이즈 바닥 ±12)')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
