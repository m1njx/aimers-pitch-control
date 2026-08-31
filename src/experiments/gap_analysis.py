#!/usr/bin/env python3
"""gap_analysis.py — LB 상위권과의 격차가 어디서 오는지 역분석.

질문
----
우리 1032점, 100위컷 1150, 1위 1421. 이 격차가 (a) 캘리브레이션 손실인지,
(b) 해상도(discrimination) 부족인지, (c) 애초에 존재하지 않는 신호인지를 가른다.

방법
----
1. **Murphy 분해**: Brier = 불확실성 - 해상도 + 신뢰도손실.
   Skill = 100000 x (해상도 - 신뢰도손실) / 불확실성.
   → 우리 점수를 "해상도가 벌어준 점수"와 "미보정으로 잃은 점수"로 쪼갠다.
   신뢰도손실이 크면 캘리브레이션에 남은 여지가 있다는 뜻.

2. **완전보정 상한**: val 폴드 자체에 isotonic 을 in-sample 로 적합한 점수.
   실전에선 불가능한 낙관적 상한이므로, 이걸로도 격차가 안 메워지면
   캘리브레이션 축은 완전히 닫힌 것.

3. **정보원별 해상도 상한(split-half 오라클)**: 변수집합 V 의 셀 확률을 val 의
   절반으로 추정해 나머지 절반에서 평가(양방향 평균). 이는 "V 안에 실제로 들어있는
   신호"의 정직한 상한이다. 우리 모델 점수와 **같은 폴드·같은 단위**로 비교되므로
   로컬-LB 스케일 문제([[dacon-signal-budget-and-recency-leak]]의 철회 사유)를 피한다.

4. **합법 대비 오라클 격차**: 같은 V 를 train(과거 시즌)으로 추정했을 때 점수와
   split-half 오라클 점수의 차이 = "과거로부터는 예측 불가능한 부분" = 우리에게
   구조적으로 닫힌 몫. 상위권도 이건 못 가져간다.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
CACHE = os.path.join(LG, 'harness/cache')

from evaluate import PROD, predict as blend_predict


def skill(p, y):
    base = y.mean() * (1 - y.mean())
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / base)


def murphy(p, y, nbins=50):
    """Brier = 불확실성 - 해상도 + 신뢰도손실. 점수 단위로 환산해 반환.

    빈 내부 y평균의 표본잡음이 해상도·신뢰도를 둘 다 위로 편향시키므로
    (편향 = nbins * unc / N) 보정값을 함께 낸다.
    """
    unc = y.mean() * (1 - y.mean())
    edges = np.quantile(p, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, nbins - 1)
    res = rel = 0.0
    for k in range(nbins):
        m = idx == k
        if not m.any():
            continue
        w = m.mean()
        res += w * (y[m].mean() - y.mean()) ** 2
        rel += w * (p[m].mean() - y[m].mean()) ** 2
    bias = nbins * unc / len(y)
    return dict(unc=unc,
                pts_resolution=100000 * (res - bias) / unc,
                pts_reliability=100000 * max(rel - bias, 0.0) / unc,
                pts_reliability_raw=100000 * rel / unc,
                pts_bias=100000 * bias / unc)


def isotonic_ceiling(p, y, rng=0):
    """교차적합 isotonic — 절반으로 적합해 나머지 절반 평가(양방향). 정직한 상한."""
    from sklearn.isotonic import IsotonicRegression
    r = np.random.RandomState(rng)
    half = r.rand(len(y)) < 0.5
    out = []
    for tr_m in (half, ~half):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(p[tr_m], y[tr_m])
        out.append(skill(ir.predict(p[~tr_m]), y[~tr_m]))
    return float(np.mean(out))


def linear_recal(p_raw, y):
    """2모수 선형 재보정 후 skill. 피처의 '정보량'을 보정 채널과 분리해 잰다.

    회귀-평균 수축(연도간 상관 r~0.46)을 프로브가 놓치면 과분산으로 음수 skill 이
    나오는데, 그건 정보가 없다는 뜻이 아니라 스케일이 틀렸다는 뜻이다.
    val 에서 2모수만 적합하므로(N=25만) 과적합은 무시할 수준.
    """
    v = p_raw.var()
    if v <= 0:
        return 0.0
    b = np.cov(p_raw, y)[0, 1] / v
    a = y.mean() - b * p_raw.mean()
    return skill(np.clip(a + b * p_raw, 1e-6, 1 - 1e-6), y)


def cell_codes(df, cols):
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + '|' + df[c].astype(str)
    return pd.factorize(s)[0]


def eb_rates(codes, y, ncell, prior, m):
    s = np.bincount(codes, weights=y, minlength=ncell)
    n = np.bincount(codes, minlength=ncell)
    return (s + m * prior) / (n + m)


def split_half_oracle(df_val, y, cols, m=50, rng=0):
    """V 의 셀 확률을 val 절반으로 추정 → 나머지 절반 평가(양방향). 신호 상한."""
    codes = cell_codes(df_val, cols)
    ncell = codes.max() + 1
    r = np.random.RandomState(rng)
    half = r.rand(len(y)) < 0.5
    out = []
    for tr_m in (half, ~half):
        va_m = ~tr_m
        prior = y[tr_m].mean()
        rates = eb_rates(codes[tr_m], y[tr_m], ncell, prior, m)
        seen = np.bincount(codes[tr_m], minlength=ncell) > 0
        p = np.where(seen[codes[va_m]], rates[codes[va_m]], prior)
        out.append(linear_recal(p, y[va_m]))
    return float(np.mean(out))


def past_estimate(df_tr, y_tr, df_val, y_val, cols, m=50):
    """같은 V 를 과거 시즌으로 추정 → val 평가. 합법적으로 얻을 수 있는 몫."""
    all_df = pd.concat([df_tr, df_val], ignore_index=True)
    codes_all = cell_codes(all_df, cols)
    ctr, cva = codes_all[:len(df_tr)], codes_all[len(df_tr):]
    ncell = codes_all.max() + 1
    prior = y_tr.mean()
    rates = eb_rates(ctr, y_tr, ncell, prior, m)
    seen = np.bincount(ctr, minlength=ncell) > 0
    p = np.where(seen[cva], rates[cva], prior)
    # 레벨 이동과 회귀-평균 수축은 별개 채널이므로 선형 재보정으로 흡수하고
    # 순수 정보량만 본다 (수축을 안 하면 과분산으로 음수가 나온다)
    return linear_recal(p, y_val)


def main():
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    for year in (2023, 2024):
        seeds = [s for s in (7, 123, 2025, 31415, 8675309)
                 if os.path.exists(os.path.join(CACHE, f'pred_{year}_{s}.npz'))]
        y = np.load(os.path.join(CACHE, f'y_{year}.npy'))
        ps = [blend_predict(dict(PROD), dict(np.load(os.path.join(CACHE, f'pred_{year}_{s}.npz'))))
              for s in seeds]
        p = np.mean(ps, axis=0)

        val = df[df.season == year].reset_index(drop=True)
        tr = df[df.season < year].reset_index(drop=True)
        ytr = tr.control_success.values.astype(float)
        assert len(val) == len(y)

        print('=' * 72)
        print(f'  eval_season {year}   n={len(y):,}  시드 {len(seeds)}개 평균 예측  '
              f'base_rate={y.mean():.4f}')
        print('=' * 72)

        sk = skill(p, y)
        M = murphy(p, y)
        print(f'\n[1] 우리 모델 skill = {sk:.1f}')
        print(f'    불확실성(=baseline brier) {M["unc"]:.6f}')
        print(f'    해상도가 벌어준 점수      +{M["pts_resolution"]:.1f}  (표본잡음 보정 후)')
        print(f'    미보정으로 잃은 점수      -{M["pts_reliability"]:.1f}  '
              f'(보정전 {M["pts_reliability_raw"]:.1f}, 표본잡음 {M["pts_bias"]:.1f})')

        iso = isotonic_ceiling(p, y)
        print(f'\n[2] 캘리브레이션 상한(교차적합 isotonic) = {iso:.1f}  '
              f'→ 남은 여지 {iso-sk:+.1f}점')

        print(f'\n[3] 정보원별 신호 상한 (split-half 오라클, 같은 폴드·같은 단위)')
        print(f'    {"변수집합":36s} {"오라클":>9s} {"과거추정":>9s} {"닫힌몫":>9s}')
        sets = [
            (['balls_before', 'strikes_before'], 'count'),
            (['pitcher_id'], 'pitcher'),
            (['batter_id'], 'batter'),
            (['pitcher_id', 'balls_before', 'strikes_before'], 'pitcher x count'),
            (['pitcher_id', 'batter_hand'], 'pitcher x batter_hand'),
            (['pitcher_id', 'inning'], 'pitcher x inning'),
            (['pitcher_id', 'game_month'], 'pitcher x month (시즌내 변동)'),
            (['pitcher_id', 'batter_id'], 'pitcher x batter (매치업)'),
        ]
        for cols, name in sets:
            try:
                o = split_half_oracle(val, y, cols)
                q = past_estimate(tr, ytr, val, y, cols)
                print(f'    {name:36s} {o:9.1f} {q:9.1f} {o-q:9.1f}')
            except Exception as e:
                print(f'    {name:36s} 실패 {e}')
        print()


if __name__ == '__main__':
    main()
