#!/usr/bin/env python3
"""spike_id_match.py — pitcher_id <-> pitcher_trackman_id 매핑 복원 가능성 판정 (feasibility spike).

배경
----
train.csv 의 pitcher_id(792명, 20000번대)와 trackman_history.csv 의
pitcher_trackman_id(906명, 50000번대)는 ID 공간이 완전히 분리돼 교집합이 0이다.
공식 Q&A(2026-08-07/08-11)는 두 ID의 "매칭"과 투수단위 과거 트랙맨 요약피처를
명시 허용했지만 매핑표는 제공하지 않는다. 두 파일은 같은 2019~2024 KBO 투구를
담고 있으므로, 투수별 등판 패턴 지문으로 매핑을 역산할 수 있는지 본다.

판정 방법 (정답표가 없으므로 자기일관성으로 검증)
-------------------------------------------------
전반기(2019~2021)만으로 매칭한 결과와 후반기(2022~2024)만으로 매칭한 결과가
독립적으로 같은 짝을 지목하면, 그 일치율이 매핑 신뢰도의 하한이다. 무작위 일치
확률은 1/906 ≈ 0.1% 이므로 일치율이 높으면 우연이 아니다.

지문: 투구수·이닝분포(선발/불펜 구분력이 큼)·월·요일·상대타자 손·볼카운트 분포.
팀은 인코딩이 서로 달라 지문에서 제외하고, 손(hand)은 하드 제약으로만 쓴다.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LG = os.path.expanduser('~/LG_data')
EARLY = [2019, 2020, 2021]
LATE = [2022, 2023, 2024]


def hist(df, gcol, col, bins, prefix):
    """투수별 col 분포를 bins 로 정규화한 히스토그램."""
    sub = df[df[col].isin(bins)]
    ct = pd.crosstab(sub[gcol], sub[col])
    ct = ct.reindex(columns=bins, fill_value=0)
    ct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    ct.columns = [f'{prefix}{b}' for b in bins]
    return ct


def fingerprint(df, gcol, hand_col, seasons):
    """투수별 지문 행렬 + 손 정보."""
    d = df[df.season.isin(seasons)]
    n = d.groupby(gcol).size().rename('n_pitch')
    parts = [
        hist(d, gcol, 'inning', list(range(1, 13)), 'inn'),
        hist(d, gcol, 'game_month', list(range(3, 12)), 'mon'),
        hist(d, gcol, 'game_dayofweek', list(range(0, 7)), 'dow'),
        hist(d, gcol, 'balls_before', [0, 1, 2, 3], 'b'),
        hist(d, gcol, 'strikes_before', [0, 1, 2], 's'),
        hist(d, gcol, 'outs_before', [0, 1, 2], 'o'),
    ]
    X = pd.concat(parts, axis=1).fillna(0.0)
    # 투구량은 로그 스케일로 (선발 vs 원포인트 릴리버 구분에 결정적)
    X['log_n'] = np.log1p(n.reindex(X.index).fillna(0))
    # 시즌별 등판 비중 (커리어 궤적)
    for s in seasons:
        cnt = d[d.season == s].groupby(gcol).size().reindex(X.index).fillna(0)
        X[f'sea{s}'] = cnt / n.reindex(X.index).replace(0, 1)
    hands = d.groupby(gcol)[hand_col].agg(lambda v: v.mode().iloc[0] if len(v.mode()) else '?')
    return X, hands.reindex(X.index), n.reindex(X.index).fillna(0)


def match(Xa, ha, na, Xb, hb, nb):
    """지문 거리 최소화 할당. 손이 다르면 금지(큰 비용)."""
    cols = [c for c in Xa.columns if c in Xb.columns]
    A = Xa[cols].to_numpy(float)
    B = Xb[cols].to_numpy(float)
    # 표준화: 분포 피처와 log_n 스케일 차이 보정
    mu = np.concatenate([A, B]).mean(0)
    sd = np.concatenate([A, B]).std(0)
    sd[sd == 0] = 1.0
    A = (A - mu) / sd
    B = (B - mu) / sd
    # 유클리드 거리
    C = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    hand_a = ha.to_numpy()
    hand_b = hb.to_numpy()
    C = C + 1e6 * (hand_a[:, None] != hand_b[None, :])
    r, c = linear_sum_assignment(C)
    return dict(zip(Xa.index[r], Xb.index[c])), C, r, c


def main():
    t0 = time.time()
    print('로딩...', flush=True)
    tr = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'balls_before',
        'strikes_before', 'outs_before', 'pitcher_id', 'pitcher_hand'])
    tk = pd.read_csv(os.path.join(LG, 'open/data/trackman_history.csv'), usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'balls_before',
        'strikes_before', 'outs_before', 'pitcher_trackman_id', 'pitcher_hand'])
    print(f'  train {len(tr):,} / trackman {len(tk):,}  ({time.time()-t0:.0f}s)', flush=True)

    print('\n손 표기 확인')
    print('  train   :', tr.pitcher_hand.value_counts().to_dict())
    print('  trackman:', tk.pitcher_trackman_id.notna().sum(), tk.pitcher_hand.value_counts().to_dict())

    res = {}
    for tag, seasons in [('EARLY', EARLY), ('LATE', LATE)]:
        Xa, ha, na = fingerprint(tr, 'pitcher_id', 'pitcher_hand', seasons)
        Xb, hb, nb = fingerprint(tk, 'pitcher_trackman_id', 'pitcher_hand', seasons)
        # 표본이 너무 적은 투수는 지문이 잡음이라 제외 (판정 신뢰도 확보용)
        keep_a = na[na >= 200].index
        keep_b = nb[nb >= 200].index
        Xa, ha = Xa.loc[keep_a], ha.loc[keep_a]
        Xb, hb = Xb.loc[keep_b], hb.loc[keep_b]
        m, C, r, c = match(Xa, ha, na.loc[keep_a], Xb, hb, nb.loc[keep_b])
        res[tag] = m
        print(f'\n{tag} {seasons}: train투수 {len(Xa)} / trackman투수 {len(Xb)} → 매칭 {len(m)}쌍'
              f'  (평균거리 {C[r, c][C[r, c] < 1e5].mean():.2f})', flush=True)

    common = set(res['EARLY']) & set(res['LATE'])
    agree = sum(res['EARLY'][p] == res['LATE'][p] for p in common)
    print('\n' + '=' * 60)
    print(f'양쪽 기간 모두 매칭된 투수: {len(common)}명')
    if common:
        print(f'독립 매칭 일치: {agree}명  →  일치율 {agree/len(common)*100:.1f}%')
        print(f'무작위 기대 일치율: {1/len(res["LATE"])*100:.2f}%')
    print('=' * 60)
    print(f'\n판정 기준: 일치율이 90% 이상이면 매핑 복원 가능, 50% 미만이면 폐기.')
    print(f'총 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
