#!/usr/bin/env python3
"""campaign_night2.py — 2026-08-25 야간 자율 캠페인 (약 5시간).

방침
----
정직하게만 기록한다. 각 단계는 실제 실행 로그에서 나온 수치만 쓰고, 실패하면 실패를
그대로 적는다(이 프로젝트는 조작 리포트 111건을 격리한 이력이 있다).
각 단계가 끝날 때마다 보고서를 즉시 갱신해 중간에 죽어도 결과가 남게 한다.

큐
--
  0. 진행 중인 pool_eq 재검증(별도 프로세스) 종료 대기 후 결과 수거
  1. outer 폴드(2024) 캐시를 5시드로 보강 (현재 3시드뿐)
  2. **피처 제거 실험** — 이번 세션에서 한 번도 안 해본 축

왜 제거인가
-----------
지금까지 시도는 전부 피처를 **더하는** 것이었고(임베딩·투수물리·시즌상대·season),
전부 실패했다. 반면 이 문제는 용량 추가를 강하게 처벌한다는 것이 반복 확인됐다
(`505` L2/L3 붕괴, `506` 21피처가 5피처보다 나쁨).

특히 `tkm_*` 17개는 조인 키가 [month, dayofweek, inning, top_bottom, balls, strikes,
outs] 뿐이라 **투수 정보가 0비트인 상황별 리그 평균**이다(`506`). 그런데 그 상황은
count/inning/game_month 원피처가 이미 담고 있다. 즉 **중복 인코딩 + 용량 비용**일
가능성이 있다. 그 위에 얹힌 물리 파생 7개(터널링 3 + 사버메트릭 4)도 같은 입력에서
나오므로 함께 본다.

변형
----
  ABL_TKM   tkm_* 수치 17개 제거 (tkm_match 플래그는 유지)
  ABL_PHYS  tkm에서 파생된 물리 7개 제거 (터널링 3 + phys 4)
  ABL_BOTH  위 둘 다 제거 (24개)
  ABL_FEAT  상황 상호작용 feat_* 10개 제거

판정 (사전 확정)
---------------
inner 3폴드(2021/2022/2023) x 5시드, 프로덕션과 동일한 **예측 배깅** 채점,
짝지은 15셀. **3폴드 전부 양수 + t > 2.5** 여야 후보. 결과를 보고 기준을 바꾸지 않는다.
"""
import os, sys, time, traceback, subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, '~/LG_data/harness')
LG = os.path.expanduser('~/LG_data')
REPORT = os.path.join(LG, 'outputs/510_overnight_ablation.md')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
T0 = time.time()

import build_cache as bc

TKM_NUM = ['tkm_rel_speed_mean', 'tkm_rel_speed_std', 'tkm_spin_rate_mean',
           'tkm_spin_rate_std', 'tkm_induced_vert_break_mean', 'tkm_induced_vert_break_std',
           'tkm_horz_break_mean', 'tkm_horz_break_std', 'tkm_extension_mean',
           'tkm_extension_std', 'tkm_rel_height_mean', 'tkm_rel_height_std',
           'tkm_rel_side_mean', 'tkm_rel_side_std', 'tkm_zone_speed_mean',
           'tkm_zone_speed_std', 'tkm_n_pitches']
PHYS = ['tkm_tunnel_dist_015s', 'tkm_plate_break_divergence', 'tkm_deception_index',
        'phys_effective_velocity', 'phys_vaa_proxy', 'phys_haa_proxy',
        'phys_spin_efficiency']
FEAT = ['feat_count_advantage', 'feat_full_count', 'feat_pitcher_ahead',
        'feat_pitcher_behind', 'feat_clutch_pressure', 'feat_scoring_position',
        'feat_platoon_fastball_inter', 'feat_platoon_breaking_inter',
        'feat_platoon_offspeed_inter', 'feat_late_inning_clutch']

VARIANTS = {
    'ABL_TKM': TKM_NUM,
    'ABL_PHYS': PHYS,
    'ABL_BOTH': TKM_NUM + PHYS,
    'ABL_FEAT': FEAT,
}

_orig = bc.build_features
DROP = []


def patched(df, prep, dec, cat_map):
    X, X133 = _orig(df, prep, dec, cat_map)
    d1 = [c for c in DROP if c in X.columns]
    d2 = [c for c in DROP if c in X133.columns]
    return X.drop(columns=d1), X133.drop(columns=d2)


def log(msg):
    print(f'[{(time.time()-T0)/60:6.1f}min] {msg}', flush=True)


def write(section):
    with open(REPORT, 'a') as f:
        f.write(section + '\n')


def bagged_paired(cache_dir, folds=FOLDS):
    """프로덕션과 동일한 예측 배깅 채점으로 (폴드, 시드) 짝지은 델타."""
    from evaluate import PROD, predict, skill
    base = os.path.join(LG, 'harness/cache')
    per_fold, cells = {}, []
    for y in folds:
        yv = np.load(os.path.join(base, f'y_{y}.npy'))
        ds = []
        for s in SEEDS:
            fa, fb = os.path.join(base, f'pred_{y}_{s}.npz'), os.path.join(cache_dir, f'pred_{y}_{s}.npz')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            ka = skill(predict(dict(PROD), dict(np.load(fa))), yv)
            kb = skill(predict(dict(PROD), dict(np.load(fb))), yv)
            ds.append(kb - ka)
        per_fold[y] = ds
        cells += ds
    # 배깅 기준 폴드 델타
    bag = {}
    for y in folds:
        yv = np.load(os.path.join(base, f'y_{y}.npy'))
        pa, pb = [], []
        for s in SEEDS:
            fa, fb = os.path.join(base, f'pred_{y}_{s}.npz'), os.path.join(cache_dir, f'pred_{y}_{s}.npz')
            if os.path.exists(fa) and os.path.exists(fb):
                pa.append(predict(dict(PROD), dict(np.load(fa))))
                pb.append(predict(dict(PROD), dict(np.load(fb))))
        if pa:
            bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)
    return per_fold, np.array(cells), bag


def main():
    write(f'\n\n# 510 — 야간 자율 캠페인 (2026-08-25 시작)\n')
    write('작성: `harness/campaign_night2.py` 가 각 단계 종료 시 자동 추가. 실행 로그 수치만 기록.\n')

    # ---- 0. pool_eq 대기 ----
    log('=== 0. pool_eq 재검증 종료 대기 ===')
    while subprocess.run(['pgrep', '-f', 'exp_pooleq.py'],
                         capture_output=True).returncode == 0:
        time.sleep(60)
    try:
        tail = subprocess.run(['tail', '-24', os.path.join(LG, 'harness/pooleq.log')],
                              capture_output=True, text=True).stdout
        log('pool_eq 종료. 결과 수거 완료')
        write('## 0. pool_eq 재검증 (3폴드 x 5시드 짝지은 비교)\n')
        write('```\n' + tail + '```\n')
    except Exception:
        write('## 0. pool_eq — 로그 수거 실패\n```\n' + traceback.format_exc() + '```\n')

    # ---- 1. outer(2024) 캐시 5시드 보강 ----
    log('=== 1. 2024 캐시 시드 보강 ===')
    try:
        r = subprocess.run(
            [os.path.join(LG, 'venv311/bin/python3'), '-u',
             os.path.join(LG, 'harness/build_cache.py'),
             '--years', '2024', '--seeds', '31415', '8675309'],
            cwd=LG, capture_output=True, text=True,
            env=dict(os.environ, OMP_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE'))
        n = len([f for f in os.listdir(os.path.join(LG, 'harness/cache'))
                 if f.startswith('pred_2024_')])
        log(f'2024 캐시 파일 {n}개 (rc={r.returncode})')
        write(f'## 1. outer(2024) 캐시 보강\n\n2024 예측 캐시 **{n}개 시드** 확보'
              f' (이전 3개). 이후 최종 확인에 5시드 사용 가능. rc={r.returncode}\n')
    except Exception:
        write('## 1. 2024 캐시 보강 실패\n```\n' + traceback.format_exc() + '```\n')

    # ---- 2. 피처 제거 실험 ----
    global DROP
    bc.build_features = patched
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    write('## 2. 피처 제거 실험\n')
    write('| 변형 | 제거 | 15셀 평균 | t | 양수 | 2021 | 2022 | 2023 | 판정 |')
    write('| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :--- |')

    for name, drop in VARIANTS.items():
        log(f'=== 2.{name}: {len(drop)}개 피처 제거 ===')
        try:
            DROP = drop
            cdir = os.path.join(LG, f'harness/cache_{name.lower()}')
            os.makedirs(cdir, exist_ok=True)
            bc.CACHE = cdir
            for y in FOLDS:
                bc.run_fold(df, y, SEEDS)
            per_fold, cells, bag = bagged_paired(cdir)
            se = cells.std(ddof=1) / np.sqrt(len(cells)) if len(cells) > 1 else float('nan')
            t = cells.mean() / se if se else float('nan')
            allpos = all(np.mean(v) > 0 for v in per_fold.values() if v)
            ok = allpos and t > 2.5
            log(f'{name}: 평균 {cells.mean():+.1f} t={t:.2f} '
                f'폴드 {[round(float(np.mean(v)),1) for v in per_fold.values()]}')
            write(f'| {name} | {len(drop)}개 | {cells.mean():+.1f} | {t:.2f} | '
                  f'{int((cells>0).sum())}/{len(cells)} | '
                  + ' | '.join(f'{bag.get(y, float("nan")):+.1f}' for y in FOLDS)
                  + f' | {"✅후보" if ok else "미달"} |')
        except Exception:
            log(f'{name} 실패')
            write(f'| {name} | {len(drop)}개 | 실행실패 | — | — | — | — | — | ❌ |')
            write('```\n' + traceback.format_exc() + '```')

    write(f'\n폴드 델타는 프로덕션과 동일한 **예측 배깅** 채점, 15셀 t는 짝지은 '
          f'(폴드,시드) 델타 기준.\n')
    write(f'판정 기준은 실행 전 확정: 3폴드 전부 양수 + t > 2.5.\n')
    log('=== 캠페인 종료 ===')
    write(f'\n캠페인 총 소요 {(time.time()-T0)/60:.0f}분.\n')


if __name__ == '__main__':
    main()
