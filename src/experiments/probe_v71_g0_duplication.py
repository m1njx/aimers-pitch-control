#!/usr/bin/env python3
"""probe_v71_g0_duplication.py — v71 두 갈래의 제안 피처가 이미 프로덕션 행렬에 있는가.

게이트 G0 (outputs/520). 비용 ~3분. 학습 없음.

프로덕션 피처 행렬 X(119) / X133(133) 을 실제로 한 폴드 빌드해서 전 컬럼을 찍고,
제안된 4개 피처의 존재 여부와 **수식 동일성**을 수치로 검증한다.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    venv311/bin/python3 -u harness/probe_v71_g0_duplication.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc
import config

YEAR = 2023


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    past = df[df.season < YEAR]
    va = df[df.season == YEAR].reset_index(drop=True)

    prep = bc.PitchPreprocessor()
    prep.fit(past, as_of_season=YEAR - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    bs = ((past['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (past['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (past['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cs = (past['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          past['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
    dec = bc.AsofDecomposer2()
    dec.fit(past, val_season=YEAR)
    X, X133 = bc.build_features(va, prep, dec, cat_map)
    print(f'built X={X.shape} X133={X133.shape}  ({time.time()-t0:.0f}s)\n', flush=True)

    print('=' * 78)
    print(f'X (GBDT 분류 3종 입력) 전 {X.shape[1]} 컬럼')
    print('=' * 78)
    for i, c in enumerate(X.columns):
        print(f'  {i:3d} {c}')
    print('\n' + '=' * 78)
    print(f'X133 추가분 (lgb_mse + mlp 입력) {X133.shape[1] - X.shape[1]} 개')
    print('=' * 78)
    for c in [c for c in X133.columns if c not in X.columns]:
        print(f'      {c}')

    # ---------------- 갈래 1: 타자 EB ----------------
    print('\n' + '=' * 78)
    print('갈래 1 — 타자 측 EB shrinkage 가 이미 있는가')
    print('=' * 78)
    for c in ['cs_b_succ_eb', 'cs_b_succ_rate', 'cs_b_succ_hist', 'cs_b_succ_minus_hist',
              'cs_bat_cur_n', 'cs_bat_hist_n', 'cs_pb_succ_diff', 'cs_pb_succ_sum',
              'asof_batter_success_rate', 'asof_batter_n', 'asof_batter_middle_rate']:
        print(f'  {c:<28} X:{c in X.columns!s:<6} X133:{c in X133.columns!s}')
    if {'cs_b_succ_eb', 'cs_p_succ_eb', 'cs_pb_succ_diff'} <= set(X.columns):
        d = (X['cs_p_succ_eb'] - X['cs_b_succ_eb']).values.astype(np.float64)
        e = X['cs_pb_succ_diff'].values.astype(np.float64)
        print(f'\n  제안 eb_diff = cs_p_succ_eb - cs_b_succ_eb  vs 기존 cs_pb_succ_diff')
        print(f'    최대 절대차 {np.nanmax(np.abs(d - e)):.3e}   상관 {np.corrcoef(d, e)[0,1]:.10f}')

    # ---------------- 갈래 2: v_eff / VAA ----------------
    print('\n' + '=' * 78)
    print('갈래 2 — v_eff / VAA 가 이미 있는가')
    print('=' * 78)
    for c in ['phys_effective_velocity', 'phys_vaa_proxy', 'phys_haa_proxy',
              'phys_spin_efficiency', 'tkm_rel_speed_mean', 'tkm_extension_mean',
              'tkm_rel_height_mean', 'tkm_induced_vert_break_mean', 'tkm_horz_break_mean',
              'tkm_tunnel_dist_015s', 'tkm_deception_index', 'tkm_plate_break_divergence']:
        print(f'  {c:<32} X:{c in X.columns!s:<6} X133:{c in X133.columns!s}')

    ext = X['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
    v_rel = X['tkm_rel_speed_mean'].clip(lower=60.0)
    prop = (v_rel * 60.5 / (60.5 - ext)).values.astype(np.float64)
    cur = X133['phys_effective_velocity'].values.astype(np.float64)
    print(f'\n  제안 v_eff  vs 기존 phys_effective_velocity')
    print(f'    최대 절대차 {np.nanmax(np.abs(prop - cur)):.3e}   상관 {np.corrcoef(prop, cur)[0,1]:.10f}')

    # 제안 VAA (릴리스높이 + ivb + 무브먼트 기하) vs 기존 phys_vaa_proxy
    ivb = X['tkm_induced_vert_break_mean'] / 12.0
    vaa = (np.arctan((X['tkm_rel_height_mean'] - 2.5 + ivb) / (60.5 - ext)) * 180 / np.pi).values
    cur2 = X133['phys_vaa_proxy'].values.astype(np.float64)
    print(f'\n  제안 VAA  vs 기존 phys_vaa_proxy')
    print(f'    최대 절대차 {np.nanmax(np.abs(vaa - cur2)):.3e}   상관 {np.corrcoef(vaa, cur2)[0,1]:.10f}')

    # ---------------- G1b: 트랙맨 블록이 상황 셀의 결정론적 함수인가 ----------------
    print('\n' + '=' * 78)
    print('G1b — 트랙맨 블록의 정보 상한 (조인 키에 투수가 있는가)')
    print('=' * 78)
    jk = config.TRACKMAN_JOIN_KEYS
    print(f'  TRACKMAN_JOIN_KEYS = {jk}')
    print(f'  투수/타자 식별자 포함 여부: '
          f'{[k for k in jk if "pitcher" in k or "batter" in k] or "없음 (0비트)"}')
    cell = va[jk].astype(str).agg('_'.join, axis=1)
    print(f'  val {YEAR}: 조인 셀 {cell.nunique():,}개 / {len(va):,}행 '
          f'(셀당 중앙값 {int(cell.value_counts().median())}행)')
    # v_eff 가 셀 안에서 상수인가 = 셀의 결정론적 함수인가
    g = pd.DataFrame({'c': cell.values, 'v': cur}).groupby('c')['v']
    within = g.std(ddof=0).fillna(0.0)
    print(f'  v_eff 의 셀 내 표준편차: 최대 {within.max():.3e}, 평균 {within.mean():.3e}')
    print(f'    -> 셀 내 상수인가: {bool(within.max() < 1e-6)}')
    print(f'  조인 키 컬럼이 X 에 존재하는가:')
    for k in jk:
        print(f'    {k:<20} in X: {k in X.columns}')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
