#!/usr/bin/env python3
"""spike_id_match2.py — pitcher_id <-> pitcher_trackman_id 매핑 복원 (2단계 정밀).

v1(spike_id_match.py) 결과: 손 제약이 인코딩 불일치로 무효화된 상태에서도
전/후반기 독립매칭 일치율 72.4% (무작위 0.25%). 지문이 실제로 투수를 식별한다는 뜻.

v2 개선
-------
1. 손 제약 복원: train {2:Right, 1:Left} <-> trackman {'Right','Left'} (투구수 비율로 확인).
2. 지문 강화: (season, month, dayofweek) 등판 셀 패턴 추가 — 같은 날 던진 기록이
   가장 강한 식별자. 상대타자 손 비율(원포인트 릴리버 식별) 추가.
3. 2단계: 1차 전역매칭 결과에서 팀 코드 대응표(train team_id <-> trackman team명)를
   과반투표로 역산 → 2차는 시즌별로 (손 + 팀) 하드제약 하에 재매칭. 후보가 팀당
   수십 명으로 줄어 정확도가 크게 오른다.
4. 검증: 시즌별 독립 매칭이 서로 같은 짝을 지목하는 비율(자기일관성). 정답표가
   없으므로 이것이 신뢰도 하한이다.
"""
import os, sys, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LG = os.path.expanduser('~/LG_data')
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
MIN_PITCH = 150          # 지문이 잡음이 되는 소량 등판 투수는 제외
HAND_MAP = {2: 'Right', 1: 'Left'}


def hist(d, g, col, bins, prefix):
    sub = d[d[col].isin(bins)]
    ct = pd.crosstab(sub[g], sub[col]).reindex(columns=bins, fill_value=0)
    ct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    ct.columns = [f'{prefix}{b}' for b in bins]
    return ct


def fingerprint(d, g, seasons, with_cells=True):
    d = d[d.season.isin(seasons)]
    n = d.groupby(g).size().rename('n')
    parts = [
        hist(d, g, 'inning', list(range(1, 13)), 'inn'),
        hist(d, g, 'balls_before', [0, 1, 2, 3], 'b'),
        hist(d, g, 'strikes_before', [0, 1, 2], 's'),
        hist(d, g, 'outs_before', [0, 1, 2], 'o'),
        hist(d, g, 'top_bottom', sorted(d.top_bottom.dropna().unique()), 'tb'),
        hist(d, g, 'bhand', ['Right', 'Left'], 'bh'),
    ]
    if with_cells:
        # (season, month, dayofweek) 등판 셀 — 가장 강한 식별자
        d = d.assign(_cell=(d.season.astype(str) + '_' + d.game_month.astype(str)
                            + '_' + d.game_dayofweek.astype(str)))
        cells = sorted(d._cell.unique())
        parts.append(hist(d, g, '_cell', cells, 'c'))
    X = pd.concat(parts, axis=1).fillna(0.0)
    X['log_n'] = np.log1p(n.reindex(X.index).fillna(0))
    return X, n.reindex(X.index).fillna(0)


def assign(Xa, Xb, block_a, block_b):
    """block(손·팀 등)이 같은 쌍만 허용하는 할당. (매칭dict, 신뢰마진dict) 반환."""
    cols = [c for c in Xa.columns if c in Xb.columns]
    A, B = Xa[cols].to_numpy(float), Xb[cols].to_numpy(float)
    both = np.concatenate([A, B])
    mu, sd = both.mean(0), both.std(0)
    sd[sd == 0] = 1.0
    A, B = (A - mu) / sd, (B - mu) / sd
    C = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    ba, bb = np.asarray(block_a, object), np.asarray(block_b, object)
    banned = ba[:, None] != bb[None, :]
    C = C + 1e6 * banned
    r, c = linear_sum_assignment(C)
    out, margin = {}, {}
    for i, j in zip(r, c):
        if C[i, j] >= 1e5:
            continue                       # 허용 후보 자체가 없던 경우
        out[Xa.index[i]] = Xb.index[j]
        row = C[i].copy()
        row[j] = np.inf
        second = row.min()
        margin[Xa.index[i]] = float(second - C[i, j])   # 2등과의 거리차 = 확신도
    return out, margin


def main():
    t0 = time.time()
    print('로딩...', flush=True)
    tr = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'balls_before', 'strikes_before', 'outs_before',
        'pitcher_id', 'pitcher_hand', 'batter_hand', 'pitcher_team_id'])
    tk = pd.read_csv(os.path.join(LG, 'open/data/trackman_history.csv'), usecols=[
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'balls_before', 'strikes_before', 'outs_before',
        'pitcher_trackman_id', 'pitcher_hand', 'batter_hand', 'pitcher_team'])
    tr['phand'] = tr.pitcher_hand.map(HAND_MAP)
    tr['bhand'] = tr.batter_hand.map(HAND_MAP)
    tk['phand'] = tk.pitcher_hand
    tk['bhand'] = tk.batter_hand
    print(f'  train {len(tr):,} / trackman {len(tk):,} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  손 매핑 검증 train Right {(tr.phand=="Right").mean():.3f} '
          f'/ trackman Right {(tk.phand=="Right").mean():.3f}')
    print(f'  top_bottom train {sorted(tr.top_bottom.unique())} '
          f'/ trackman {sorted(tk.top_bottom.unique())}')

    # ---------- 1단계: 전역 매칭 (손 제약만) ----------
    print('\n[1단계] 전역 매칭, 손 제약', flush=True)
    Xa, na = fingerprint(tr, 'pitcher_id', SEASONS)
    Xb, nb = fingerprint(tk, 'pitcher_trackman_id', SEASONS)
    ka = na[na >= MIN_PITCH].index
    kb = nb[nb >= MIN_PITCH].index
    Xa, Xb = Xa.loc[ka], Xb.loc[kb]
    ha = tr.groupby('pitcher_id').phand.agg(lambda v: v.mode().iloc[0]).reindex(Xa.index)
    hb = tk.groupby('pitcher_trackman_id').phand.agg(lambda v: v.mode().iloc[0]).reindex(Xb.index)
    m1, _ = assign(Xa, Xb, ha, hb)
    print(f'  train {len(Xa)}명 / trackman {len(Xb)}명 → {len(m1)}쌍')

    # ---------- 팀 대응표 역산 ----------
    ta = tr.groupby(['pitcher_id', 'season']).pitcher_team_id.agg(
        lambda v: v.mode().iloc[0])
    tb = tk.groupby(['pitcher_trackman_id', 'season']).pitcher_team.agg(
        lambda v: v.mode().iloc[0])
    votes = defaultdict(Counter)
    for p, q in m1.items():
        for s in SEASONS:
            if (p, s) in ta.index and (q, s) in tb.index:
                votes[ta[(p, s)]][tb[(q, s)]] += 1
    team_map = {}
    for tid, cnt in votes.items():
        top, n_top = cnt.most_common(1)[0]
        team_map[tid] = top
        print(f'  팀 {tid} -> {top}  (지지 {n_top}/{sum(cnt.values())} '
              f'= {n_top/sum(cnt.values())*100:.0f}%)')
    consistent = all(len(set([team_map[t]])) == 1 for t in team_map)
    rev = Counter(team_map.values())
    print(f'  팀 대응 단사 여부: {"OK" if max(rev.values())==1 else "충돌 있음 " + str(rev)}')

    # ---------- 2단계: 시즌별 (손+팀) 하드제약 매칭 ----------
    print('\n[2단계] 시즌별 매칭, 손+팀 제약', flush=True)
    per_season = {}
    for s in SEASONS:
        Xa, na = fingerprint(tr, 'pitcher_id', [s], with_cells=True)
        Xb, nb = fingerprint(tk, 'pitcher_trackman_id', [s], with_cells=True)
        ka = na[na >= MIN_PITCH].index
        kb = nb[nb >= MIN_PITCH].index
        Xa, Xb = Xa.loc[ka], Xb.loc[kb]
        if not len(Xa) or not len(Xb):
            continue
        ba = [f'{tr[tr.pitcher_id==p].phand.mode().iloc[0]}|'
              f'{team_map.get(ta.get((p, s)), "?")}' for p in Xa.index]
        bb = [f'{tk[tk.pitcher_trackman_id==q].phand.mode().iloc[0]}|'
              f'{tb.get((q, s), "?")}' for q in Xb.index]
        m, mg = assign(Xa, Xb, ba, bb)
        per_season[s] = (m, mg)
        print(f'  {s}: train {len(Xa)} / tkm {len(Xb)} → {len(m)}쌍', flush=True)

    # ---------- 자기일관성 검증 ----------
    print('\n[검증] 시즌 간 독립매칭 일치율', flush=True)
    pair_votes = defaultdict(Counter)
    for s, (m, mg) in per_season.items():
        for p, q in m.items():
            pair_votes[p][q] += 1
    multi = {p: c for p, c in pair_votes.items() if sum(c.values()) >= 2}
    agree = sum(c.most_common(1)[0][1] for c in multi.values())
    total = sum(sum(c.values()) for c in multi.values())
    print(f'  2시즌 이상 등장한 투수: {len(multi)}명, 총 {total}표')
    print(f'  최빈짝 일치표 {agree} → 일치율 {agree/total*100:.1f}%')
    unanimous = sum(1 for c in multi.values() if len(c) == 1)
    print(f'  전 시즌 만장일치 투수: {unanimous}/{len(multi)} '
          f'= {unanimous/len(multi)*100:.1f}%')

    out = os.path.join(LG, 'harness/pitcher_id_map.csv')
    rows = [dict(pitcher_id=p, pitcher_trackman_id=c.most_common(1)[0][0],
                 votes=c.most_common(1)[0][1], total=sum(c.values()))
            for p, c in pair_votes.items()]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'\n매핑표 저장: {out} ({len(rows)}행)')
    print(f'총 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
