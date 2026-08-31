#!/usr/bin/env python3
"""exp_ptk.py — 가설 P: 투수 단위 트랙맨 물리 프로파일.

가설
----
현행 트랙맨 피처 17개는 조인 키가 [month, dayofweek, inning, top_bottom, balls,
strikes, outs] 뿐이라 **투수가 키에 없다**. 같은 카운트·이닝이면 에이스든 신인이든
완전히 동일한 리그 평균값이 들어간다. 그런데 이 프로젝트는 그 값들 위에 터널링
거리·기만지수·유효구속·VAA/HAA 같은 정교한 물리 피처를 얹어 쓰고 있다
(build_cache.py:76-100). **물리 엔진은 이미 있는데 입력이 리그 평균이라 투수 식별
정보가 0비트다.**

공식 Q&A(2026-08-07/08-11)는 pitcher_id <-> pitcher_trackman_id 매칭과 시즌이전
트랙맨 요약피처를 명시 허용했다. 두 ID는 공간이 분리돼 있었으나
spike_id_match2.py 가 등판패턴 지문 + 팀/손 하드제약으로 매핑을 복원했다
(시즌간 독립매칭 일치율 98.8%, 만장일치 95.7%, train 행 커버리지 87.4%).

왜 임베딩(E1, REJECT)과 다른가
------------------------------
임베딩은 투수 **정체성**을 줬는데, 정체성은 asof 성공률과 거의 중복이다(둘 다
"이 투수가 얼마나 잘했나"). 물리 프로파일은 다른 양이다:
  1. **릴리스 산포(std)** 는 제구력의 물리적 실체 자체다. 적은 표본의 성공률
     평균으로는 복원되지 않는다.
  2. **콜드스타트에서 살아있다.** 성적 이력이 없는 투수도 트랙맨 이력은 있다.
     임베딩이 구조적으로 도울 수 없던 구간(val 행의 15.7%가 미학습 투수).
  3. 저차원 조밀 실수라 수축이 잘 먹는다. 792레벨 범주형과 실패양상이 다르다.

정직한 반론: 이력이 충분한 투수는 asof 성공률이 이미 물리를 적분해 담고 있다.
따라서 기대 이득은 저경험 구간(행의 12.7%)에 집중되고 희석된다. 효과가 작을 수
있다는 전제 하에 5시드로 잰다.

규정 준수
---------
(pitcher_id, season) 키로 **그 시즌 이전 시즌들만** 집계한 고정 테이블. 각 행은
자기 입력변수 + train/트랙맨 이력으로만 계산되고 test.csv 의 다른 행을 일절
참조하지 않는다 → 규정4 및 데이터설명서 5)/6) 저촉 없음. 현재 투구의 실측값은
쓰지 않는다.

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 -u harness/exp_ptk.py --years 2022 2023 --seeds 7 123 2025 31415 8675309
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))

import build_cache as bc

METRICS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
           'extension', 'rel_height', 'rel_side', 'zone_speed']
PTK_CACHE = os.path.join(LG, 'harness/cache_ptk')


def build_profile(id_map_path, trackman_path):
    """(pitcher_id, season) -> 그 시즌 '이전' 시즌들의 투수별 물리 프로파일."""
    m = pd.read_csv(id_map_path)
    m = m[m.votes == m.total]                      # 만장일치 매칭만 사용 (정밀도 우선)
    print(f'  매핑 {len(m)}명(만장일치)', flush=True)

    tk = pd.read_csv(trackman_path, usecols=['season', 'pitcher_trackman_id'] + METRICS)
    tk = tk.merge(m[['pitcher_id', 'pitcher_trackman_id']], on='pitcher_trackman_id',
                  how='inner')
    print(f'  트랙맨 {len(tk):,}행이 투수에 연결됨', flush=True)

    g = tk.groupby(['pitcher_id', 'season'])
    agg = g[METRICS].agg(['count', 'sum'])
    sq = g[METRICS].apply(lambda d: (d ** 2).sum())
    agg.columns = [f'{c}_{s}' for c, s in agg.columns]
    for c in METRICS:
        agg[f'{c}_sq'] = sq[c]
    agg = agg.reset_index().sort_values(['pitcher_id', 'season'])

    # 시즌 누적 후 1시즌 shift = "이 시즌 이전까지"의 집계
    num = [c for c in agg.columns if c not in ('pitcher_id', 'season')]
    cum = agg.groupby('pitcher_id')[num].cumsum()
    prior = cum.groupby(agg.pitcher_id).shift(1)
    prior['pitcher_id'] = agg.pitcher_id.values
    prior['season'] = agg.season.values

    out = pd.DataFrame({'pitcher_id': prior.pitcher_id, 'season': prior.season})
    n = prior[f'{METRICS[0]}_count']
    out['ptk_n_pitches'] = n.values
    for c in METRICS:
        cnt = prior[f'{c}_count'].replace(0, np.nan)
        mean = prior[f'{c}_sum'] / cnt
        var = (prior[f'{c}_sq'] / cnt - mean ** 2).clip(lower=0)
        out[f'ptk_{c}_mean'] = mean.values
        out[f'ptk_{c}_std'] = np.sqrt(var).values

    # 파생: 릴리스 포인트 산포 = 반복성(제구력)의 직접 물리 대리변수
    out['ptk_rel_scatter'] = np.sqrt(out.ptk_rel_height_std ** 2 + out.ptk_rel_side_std ** 2)
    out['ptk_break_scatter'] = np.sqrt(out.ptk_induced_vert_break_std ** 2
                                       + out.ptk_horz_break_std ** 2)
    out['ptk_speed_cv'] = out.ptk_rel_speed_std / out.ptk_rel_speed_mean.replace(0, np.nan)
    out = out[out.ptk_n_pitches.notna()]
    print(f'  프로파일 테이블 {len(out):,}행 x {out.shape[1]-2}피처', flush=True)
    return out


PROFILE = None
PTK_COLS = None
FILL = 'zero'
_orig_build_features = bc.build_features
_FILL_VALUES = None


def patched_build_features(df, prep, dec, cat_map):
    global _FILL_VALUES
    X, X133 = _orig_build_features(df, prep, dec, cat_map)
    key = df[['pitcher_id', 'season']].reset_index(drop=True)
    prof = key.merge(PROFILE, on=['pitcher_id', 'season'], how='left')
    prof['ptk_match'] = prof.ptk_n_pitches.notna().astype(np.float32)
    prof = prof[PTK_COLS].astype(np.float32)
    if FILL == 'zero':
        # 결측을 0으로 — 구속 0, 회전수 0 같은 물리적으로 불가능한 값이 들어간다.
        prof = prof.fillna(0.0)
    else:
        # 결측(미매칭 투수·데뷔시즌)은 리그 평균으로. ptk_match 플래그가 구분해 준다.
        # 표준화 후 0 근처에 놓이므로 MLP 입력으로 안전하다. 채움값은 첫 호출(train)
        # 에서 고정하고 val 에 그대로 재사용한다 — val 통계를 쓰면 누수다.
        if _FILL_VALUES is None:
            _FILL_VALUES = prof.drop(columns=['ptk_match']).mean()
        prof = prof.fillna(_FILL_VALUES).fillna(0.0)
    prof.index = X.index
    X = pd.concat([X, prof], axis=1)
    prof133 = prof.copy(); prof133.index = X133.index
    X133 = pd.concat([X133, prof133], axis=1)
    return X, X133


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025, 31415, 8675309])
    ap.add_argument('--fill', choices=['zero', 'mean'], default='zero')
    ap.add_argument('--lean', action='store_true',
                    help='산포 3개 + 표본수 + 플래그만 사용 (용량비용 최소화)')
    ap.add_argument('--tag', default='ptk')
    a = ap.parse_args()

    global PROFILE, PTK_COLS, FILL, _FILL_VALUES
    FILL = a.fill
    t0 = time.time()
    print(f'[1] 투수 물리 프로파일 구축  (fill={a.fill}, lean={a.lean})', flush=True)
    PROFILE = build_profile(os.path.join(LG, 'harness/pitcher_id_map.csv'),
                            os.path.join(LG, 'open/data/trackman_history.csv'))
    if a.lean:
        # 가설의 핵심만: 반복성(산포). mean 16개는 asof 성공률과 중복도가 높고
        # 용량비용만 물린다 — 이 문제는 용량 추가를 강하게 처벌한다(L2/L3 붕괴).
        PTK_COLS = ['ptk_rel_scatter', 'ptk_break_scatter', 'ptk_speed_cv',
                    'ptk_n_pitches', 'ptk_match']
    else:
        PTK_COLS = [c for c in PROFILE.columns if c.startswith('ptk_')] + ['ptk_match']
    print(f'  추가 피처 {len(PTK_COLS)}개 ({time.time()-t0:.0f}s)\n', flush=True)

    cache_dir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cache_dir, exist_ok=True)
    bc.build_features = patched_build_features
    bc.CACHE = cache_dir

    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    for y in a.years:
        _FILL_VALUES = None          # 폴드마다 train 기준으로 다시 고정
        bc.run_fold(df, y, a.seeds)

    print('\n[3] 채점', flush=True)
    from exp_capacity import score_dir
    base = score_dir(os.path.join(LG, 'harness/cache'), a.years, a.seeds)
    new = score_dir(cache_dir, a.years, a.seeds)
    print(f'  P0 (현행)            inner={base["inner"]:9.1f}  '
          f'연도별={ {k: round(v,1) for k,v in base["season_mean"].items()} }  '
          f'seed_sd={base["seed_sd"]:.1f}')
    print(f'  P1 (+투수물리프로파일) inner={new["inner"]:9.1f}  '
          f'연도별={ {k: round(v,1) for k,v in new["season_mean"].items()} }  '
          f'seed_sd={new["seed_sd"]:.1f}')
    delta = new['inner'] - base['inner']
    noise = float(np.mean([base['seed_sd'], new['seed_sd']]))
    print(f'\n  → 델타={delta:+.1f}  노이즈(보수)={noise:.1f}  '
          f'신뢰가능={bool(delta > noise)}')
    print(f'  [참고] 평균의 표준오차={noise/np.sqrt(len(a.seeds)*len(a.years)):.1f}')
    print(f'총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
