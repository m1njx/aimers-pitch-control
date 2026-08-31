#!/usr/bin/env python3
"""count_shift_probe.py — 프로덕션의 카운트별 미세보정(count_shifts)을 정직하게 재검증.

배경
----
`work/submit_v50/model/count_shifts_artifact.pkl` 은 볼-스트라이크 12개 상태별
가산 상수다. v40(2026-08-20)에서 도입돼 구조적 개선의 일부로 기록됐고, 그 뒤 set-A
전 버전에서 **바이트 동일**하다 — 즉 한 번도 재산출·재검증되지 않았다. 하네스도 이 항을
모델링하지 않아(evaluate.py 주석) 지금까지의 모든 하네스 판정에서 빠져 있었다.

세 가지를 묻는다
----------------
  1. 출하된 값이 홀드아웃 폴드에서 실제로 도움이 되는가?
  2. 지금 데이터로 다시 적합하면 값이 얼마나 달라지는가?
  3. 재적합한 값이 폴드를 넘어 전이되는가? (LOFO)

카운트별 최적 가산항은 해당 카운트 내 잔차 평균이다: s_c = mean(y - p | count=c).
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict, skill, CACHE

FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def main():
    shipped = joblib.load(os.path.join(LG, 'work/submit_v50/model/count_shifts_artifact.pkl'))
    print('출하된 count_shifts (v40 이후 불변)')
    print('  ' + '  '.join(f'{k}:{v:+.5f}' for k, v in sorted(shipped.items())))

    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'balls_before', 'strikes_before', 'control_success'])
    df.columns = [c.replace('﻿', '') for c in df.columns]

    D = {}
    for y in FOLDS:
        va = df[df.season == y].reset_index(drop=True)
        key = (va.balls_before.fillna(0).astype(int).astype(str) + '_' +
               va.strikes_before.fillna(0).astype(int).astype(str)).values
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        assert len(yv) == len(key)
        ps = []
        for s in SEEDS:
            f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
            if os.path.exists(f):
                ps.append(predict(dict(PROD), dict(np.load(f))))
        D[y] = (yv, key, ps)

    keys = sorted(shipped.keys())

    def fit_shifts(years):
        num = {k: 0.0 for k in keys}
        den = {k: 0.0 for k in keys}
        for y in years:
            yv, key, ps = D[y]
            p = np.mean(ps, axis=0)
            r = yv - p
            for k in keys:
                m = key == k
                num[k] += r[m].sum()
                den[k] += m.sum()
        return {k: (num[k] / den[k] if den[k] else 0.0) for k in keys}

    def apply_score(years, sh):
        out = []
        for y in years:
            yv, key, ps = D[y]
            add = np.array([sh.get(k, 0.0) for k in key]) if sh else 0.0
            out.append(np.mean([skill(np.clip(p + add, 1e-6, 1 - 1e-6), yv) for p in ps]))
        return float(np.mean(out))

    print('\n[재적합] 3폴드 전체로 다시 구한 최적 카운트별 가산항')
    refit = fit_shifts(FOLDS)
    print(f'  {"count":>6} {"출하값":>11} {"재적합":>11} {"차이":>11}')
    for k in keys:
        print(f'  {k:>6} {shipped[k]:+11.5f} {refit[k]:+11.5f} {refit[k]-shipped[k]:+11.5f}')

    print('\n[LOFO] 2폴드에서 적합 → 남긴 폴드 평가')
    print(f'  {"홀드아웃":>8} {"보정없음":>10} {"출하값":>10} {"재적합":>10} '
          f'{"출하−무":>9} {"재적합−무":>10}')
    d_ship, d_refit = [], []
    for held in FOLDS:
        sel = [y for y in FOLDS if y != held]
        sh = fit_shifts(sel)
        none_ = apply_score([held], None)
        ship_ = apply_score([held], shipped)
        ref_ = apply_score([held], sh)
        d_ship.append(ship_ - none_)
        d_refit.append(ref_ - none_)
        print(f'  {held:>8} {none_:10.1f} {ship_:10.1f} {ref_:10.1f} '
              f'{ship_-none_:+9.2f} {ref_-none_:+10.2f}')

    print(f'\n  출하값 평균 효과   {np.mean(d_ship):+.2f}점  (양수 {sum(1 for v in d_ship if v>0)}/3)')
    print(f'  재적합 평균 효과   {np.mean(d_refit):+.2f}점  (양수 {sum(1 for v in d_refit if v>0)}/3)')

    # ---- 전역 레벨 성분과 카운트 구조를 분리 ----
    # 재적합값이 전부 +0.004~+0.017 로 한쪽에 쏠려 있다. 이는 카운트 구조가 아니라
    # 하네스 블렌드의 전역 편향(하네스는 count_shifts 를 모델링하지 않아 프로덕션과
    # 절대 레벨이 다르다)을 흡수한 것이다. 전역 성분은 프로덕션의 SHIFT 가 이미 담당하고
    # 로컬에서 튜닝하면 2025 베이스레이트에서 반대로 갈 수 있다([[신호예산]]).
    # 따라서 **빈도가중 평균을 뺀 카운트 차등분**만 따로 전이되는지 본다.
    freq = {k: 0.0 for k in keys}
    for y in FOLDS:
        _, key, _ = D[y]
        for k in keys:
            freq[k] += (key == k).sum()
    tot = sum(freq.values())

    def center(sh):
        m = sum(sh[k] * freq[k] for k in keys) / tot
        return {k: sh[k] - m for k in keys}, m

    print('\n[분리] 카운트 차등분만 (빈도가중 평균 제거)')
    rc, rm = center(refit)
    sc, sm = center(shipped)
    print(f'  출하값 전역성분 {sm:+.5f} / 재적합 전역성분 {rm:+.5f}')
    print(f'  {"count":>6} {"출하 차등":>11} {"재적합 차등":>12}')
    for k in keys:
        print(f'  {k:>6} {sc[k]:+11.5f} {rc[k]:+12.5f}')

    print(f'\n  {"홀드아웃":>8} {"보정없음":>10} {"출하차등":>10} {"재적합차등":>11} '
          f'{"출하−무":>9} {"재적합−무":>10}')
    ds, dr = [], []
    for held in FOLDS:
        sel = [y for y in FOLDS if y != held]
        shc, _ = center(fit_shifts(sel))
        none_ = apply_score([held], None)
        a = apply_score([held], sc)
        b = apply_score([held], shc)
        ds.append(a - none_); dr.append(b - none_)
        print(f'  {held:>8} {none_:10.1f} {a:10.1f} {b:11.1f} {a-none_:+9.2f} {b-none_:+10.2f}')
    print(f'\n  출하 차등분 평균   {np.mean(ds):+.2f}점  (양수 {sum(1 for v in ds if v>0)}/3)')
    print(f'  재적합 차등분 평균 {np.mean(dr):+.2f}점  (양수 {sum(1 for v in dr if v>0)}/3)')
    print('\n  ※ LB 노이즈 바닥 ±12점. 차등분 효과가 이보다 훨씬 작으면')
    print('    count_shifts 는 사실상 전역 SHIFT 와 중복이며 카운트 구조는 없는 것이다.')


if __name__ == '__main__':
    main()
