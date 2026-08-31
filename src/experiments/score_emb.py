#!/usr/bin/env python3
"""score_emb.py — 엔티티 임베딩 스캔 최종 채점 (재학습 없이 캐시만 읽음).

exp_embed.py 를 여러 번(시드를 나눠서) 돌린 뒤, 흩어져 쌓인 pred_{year}_{seed}.npz
를 전부 모아 한 번에 채점한다. 델타가 시드 노이즈를 넘는지가 유일한 판정 기준.
"""
import os, sys
import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
os.chdir(LG)

from exp_capacity import score_dir, CACHE

YEARS = [2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
LEVELS = ['E1', 'E1b', 'E1c', 'E1d']

dirs = {'E0': CACHE}
for lv in LEVELS:
    dirs[lv] = os.path.join(LG, f'harness/cache_emb_{lv}')

print('=== 엔티티 임베딩 최종 스캔 (inner 2022/2023 전용) ===')
res = {}
for k, d in dirs.items():
    n = sum(os.path.exists(os.path.join(d, f'pred_{y}_{s}.npz'))
            for y in YEARS for s in SEEDS)
    r = score_dir(d, YEARS, SEEDS)
    if r is None:
        print(f'  {k:4s}: 캐시 없음')
        continue
    res[k] = r
    yr = {y: round(v, 1) for y, v in r['season_mean'].items()}
    print(f'  {k:4s}: inner={r["inner"]:9.1f}  연도별={yr}  '
          f'seed_sd={r["seed_sd"]:.1f}  n={n}/{len(YEARS)*len(SEEDS)}')

if 'E0' in res and len(res) > 1:
    base = res['E0']['inner']
    best = max((k for k in res if k != 'E0'), key=lambda k: res[k]['inner'])
    delta = res[best]['inner'] - base
    sds = [res[k]['seed_sd'] for k in res if not np.isnan(res[k]['seed_sd'])]
    # 보수 기준: 기존 스캔(exp_capacity/exp_embed)과 동일하게 시드 sd 평균 그대로.
    # 판정은 이 값으로만 한다 — 과거 리포트와 비교 가능해야 하고, 느슨한 기준은
    # 이 프로젝트에서 반복적으로 거짓 양성을 만들어 왔다.
    noise = float(np.mean(sds))
    # 참고용: inner 는 (시드 x 연도) 평균이므로 평균의 표준오차는 이만큼 작다.
    se = noise / np.sqrt(len(SEEDS) * len(YEARS))
    print()
    print(f'  → 최고={best}  E0(현행) 대비 델타={delta:+.1f}  노이즈(보수, 기존기준)={noise:.1f}')
    print(f'  → 신뢰가능={bool(delta > noise)}   [참고: 평균의 표준오차={se:.1f}]')
