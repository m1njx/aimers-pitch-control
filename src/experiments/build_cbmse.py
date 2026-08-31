"""build_cbmse.py — 우리 5성분에 없는 `cb_mse`(CatBoost Direct-RMSE)를 폴드 캐시로 추가한다.

왜
--
팀 `OPEN_QUESTIONS.md` §3.2(a): 우리 자신의 보고서
`322_advanced_nonlinear_calibration_and_stacking_sota.md` 의 제약 메타스태킹 최적해가

| 컴포넌트 | 우리 보고서 최적 가중 | v42 실제 배포 |
| CatBoost **Direct RMSE** | **36.5% (최고 기여)** | **0% — 배포본에 없음** |
| SimpleMLP MSE | 14.0% | 40% |

이고, `submit_v42.zip` 모델 전수 확인 결과 **CB-RMSE 아티팩트가 실제로 없다**(25개 전부
Classifier/binary 계열). 그쪽 감사는 우리 보고서 기준 **LB 12~16점이 미청구**라고 본다.

우리는 과거 이 컴포넌트를 우리 피처셋에서 기각했지만(−6.6/−6.7 HARMFUL),
그건 **다른 설정**에서의 판정이다. 지금 우리 5성분 블렌드에 붙여 우리 게이트로 다시 잰다.

⚠️ 전이비 정정(2026-08-27): 정직한 홀드아웃 로컬은 LB 와 상관 0.9967, 전이비 ~1.0.
   따라서 **로컬 Δ 를 액면가로 읽되**, 블렌드 전달배수 0.25~0.50 을 곱해야 팀 블렌드 기준이 된다.

사전 확정 판정 기준 (착수 전 고정)
  G1) 6성분 최적가중 블렌드가 현행 5성분 PROD 대비 **3폴드 전부 양수**
  G2) 평균 Δ ≥ +12 (로컬 = LB arm 기준). 미달이면 REJECT.
  ⚠️ 최적가중은 in-fold 라 상한이다. 통과 시 LOFO 전방검증 필수.

실행: venv311/bin/python3 harness/build_cbmse.py --years 2022 2023 2024 --seeds 7 123 2025
"""
import argparse
import os
import sys
import time

import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc  # noqa: E402

OUT = os.path.join(LG, 'harness/cache_cbmse')
os.makedirs(OUT, exist_ok=True)


def main():
    from catboost import CatBoostRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023, 2024])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025])
    a = ap.parse_args()

    df = bc.load_train() if hasattr(bc, 'load_train') else None
    import pandas as pd
    if df is None:
        df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), encoding='utf-8-sig')

    for year in a.years:
        todo = [s for s in a.seeds
                if not os.path.exists(os.path.join(OUT, f'cbmse_{year}_{s}.npy'))]
        if not todo:
            print(f'{year}: 전부 캐시됨', flush=True)
            continue
        tr = df[df.season < year]
        va = df[df.season == year].reset_index(drop=True)
        print(f'\n=== {year}: train {len(tr):,} val {len(va):,} ===', flush=True)
        t0 = time.time()

        prep = bc.PitchPreprocessor()
        prep.fit(tr, as_of_season=year - 1, is_final=False,
                 trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
        bs = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
              (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        cs = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              tr['strikes_before'].fillna(0).astype(int).astype(str))
        cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
        dec = bc.AsofDecomposer2()
        dec.fit(tr, val_season=year)

        Xtr, _ = bc.build_features(tr, prep, dec, cat_map)
        Xva, _ = bc.build_features(va, prep, dec, cat_map)
        ytr = tr['control_success'].values.astype(np.float64)
        Xtr_cb, Xva_cb = bc.cast_cb(Xtr), bc.cast_cb(Xva)
        print(f'  피처 {Xtr.shape[1]}  ({time.time()-t0:.0f}s)', flush=True)

        for seed in todo:
            t1 = time.time()
            # cb_bin 과 동일 설정에서 손실만 RMSE 로 (= Direct-RMSE)
            m = CatBoostRegressor(iterations=300, learning_rate=0.06, depth=6,
                                  loss_function='RMSE', cat_features=bc.CAT_COLS,
                                  random_seed=seed, verbose=0, thread_count=6)
            m.fit(Xtr_cb, ytr)
            p = np.clip(m.predict(Xva_cb), 1e-6, 1 - 1e-6)
            np.save(os.path.join(OUT, f'cbmse_{year}_{seed}.npy'), p)
            print(f'  seed {seed}: cb_mse 완료 ({time.time()-t1:.0f}s)  '
                  f'평균 {p.mean():.4f} sd {p.std():.4f}', flush=True)


if __name__ == '__main__':
    main()
