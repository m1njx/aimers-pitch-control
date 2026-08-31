#!/usr/bin/env python3
"""exp_ebm10.py — AsofDecomposer2 의 EB 수축 eb_m 을 150 -> 10 으로 (풀 파이프라인).

가설
----
(a) 어떤 측정에서 나왔나: `harness/diag_asof_estimator.py`. `cs_*_succ_eb` 를 단독
    예측기로 직접 채점하면 skill 이 m 에 대해 매끈한 단봉이고 **최적이 m≈8~10**,
    3폴드 전부 일치한다. 현행 m=150 은 단독 채널에서 평균 **−258.6** 만큼 손해다.

        m      2021      2022      2023      평균
        8    1251.9    1705.5     579.1    1178.9
       10    1252.6    1706.9     579.0    1179.5   <- 최적
       50    1180.6    1643.5     466.8    1096.9
      150    1028.7    1504.1     229.9     920.9   <- 현행

    cur_n 중앙값이 투수 507 / 타자 573 이므로 m=150 은 과거 시즌에 23% 가중을 준다.
    현시즌 누적 성공률이 그만큼 강한 신호인데 눌러버리고 있었다.
(b) 닫힌 축과 다른가: 지금까지 닫은 축은 캘리브레이션·블렌드가중치·시드수·잔차보정
    (전부 Idea B 의 +25 아핀 예산 안) 과 era/recency 재가중이다. 이건 **피처 자체의
    추정기 오설정**이고 어느 쪽도 아니다. eb_m 은 정직한 하네스로 튜닝된 적이 없다.
(c) 규정4(행 독립성): 위반 없음. `cs_*_eb` 는 **그 행 자신의** asof_* 컬럼과 train 에서
    고정된 경계표만 쓴다. 스코어링 배치의 다른 행을 보지 않는다. m 은 상수일 뿐이다.

⚠️ outputs/513 교훈: 단독 채널 +258.6 은 상한이 아니다. GBDT 는 이미
   `cs_*_rate`(cur_rate) 와 `cs_*_hist` 를 별도 컬럼으로 받고 있으므로 m 을 스스로
   부분 재구성할 수 있다. 그래서 풀 파이프라인으로만 갈린다.

판정 기준 (착수 전 확정, 결과 보고 변경 금지)
  3폴드 전부 시드평균 양수 AND 15셀 t > 2.5.
  채택 전 추가로 Cov/Var 분해(프로토콜 8번)를 확인한다 — 정보 항이 0 이하면 기각.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE nohup venv311/bin/python3 -u harness/exp_ebm10.py &
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc                                   # noqa: E402
from exp_template import score, FOLDS, SEEDS               # noqa: E402
from agent2_asof_decomp2 import AsofDecomposer2            # noqa: E402

NEW_M = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=float, default=NEW_M)
    ap.add_argument('--years', type=int, nargs='+', default=FOLDS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--score-only', action='store_true')
    a = ap.parse_args()

    print(__doc__.split('판정 기준')[0].strip())
    print(f'\n조작 변수: AsofDecomposer2.eb_m  150 -> {a.m}  (그 외 전부 동일)')
    print('판정 기준(사전 확정): 3폴드 전부 양수 + t>2.5, 이후 Cov/Var 분해 확인\n')

    tag = f'ebm{int(a.m)}'
    cdir = os.path.join(LG, f'harness/cache_{tag}')
    os.makedirs(cdir, exist_ok=True)

    if not a.score_only:
        t0 = time.time()
        df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
        df.columns = [c.replace('﻿', '') for c in df.columns]
        # run_fold 는 모듈 전역의 AsofDecomposer2 를 인자 없이 생성한다 -> 그것만 바꾼다.
        bc.AsofDecomposer2 = lambda m=a.m: AsofDecomposer2(eb_m=m)
        bc.CACHE = cdir
        for y in a.years:
            bc.run_fold(df, y, a.seeds)
            print(f'  [{y} 완료 {(time.time()-t0)/60:.1f}분]', flush=True)

    score(cdir)


if __name__ == '__main__':
    main()
