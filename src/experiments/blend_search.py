#!/usr/bin/env python3
"""blend_search.py — 학습비용 0인 마지막 축: 블렌드 가중치를 정직하게 재탐색한다.

왜 지금인가
-----------
하네스는 원래 이걸 위해 만들어졌다("캐시가 있으면 임의 설정 채점은 즉시"). 그런데
현재 프로덕션 가중치는 (a) LB 점수를 보고 고른 것이고 — LB 노이즈 바닥이 ±12점이라
1030/1032/1032 구간은 사실상 구분 불가 — (b) 하네스 재탐색은 2폴드 시절에만 돌았다.
`508`에서 2폴드 선별이 era 개입 2건을 통과시킬 뻔했음이 드러났으므로 3폴드로 다시 본다.

정직한 프로토콜 — leave-one-fold-out
-----------------------------------
설정을 2개 폴드에서 고르고 **남긴 폴드에서 평가**한다(3회 회전). 이는 단순히 "최적
설정"을 찾는 게 아니라 **설정 선택이 폴드를 넘어 전이되는가**를 측정한다. 오늘 하루가
보여준 대로 이 문제에서는 폴드 특유 효과가 지배적이므로, 전이되지 않는 최적화는
무의미하다.

  선택폴드에서만 좋고 홀드아웃에서 PROD 이하 → 설정 탐색은 전이 안 됨. PROD 유지.
  홀드아웃 3회 모두 PROD 초과            → 실재하는 개선.

scale/shift 는 기본적으로 고정한다. 평가 시즌 베이스레이트가 로컬(~.53)과 LB(2025,
r≈.456)에서 크게 다르고 SHIFT 는 LB 대수식으로 이미 유도된 값이라, 로컬에서 최적화하면
2025에서 반대로 갈 수 있다. 별도 스위치로만 켠다.
"""
import os, sys, itertools, argparse, warnings
import numpy as np

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict, skill, CACHE

FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]


def load():
    D = {}
    for y in FOLDS:
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        ps = {}
        for s in SEEDS:
            f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
            if os.path.exists(f):
                ps[s] = dict(np.load(f))
        D[y] = (yv, ps)
    return D


def score(cfg, D, years):
    """지정 폴드들에서 (폴드평균(시드평균)) skill."""
    per = []
    for y in years:
        yv, ps = D[y]
        per.append(np.mean([skill(predict(cfg, P), yv) for P in ps.values()]))
    return float(np.mean(per)), per


def grid_top(step=0.05):
    out = []
    for wg in np.arange(0.0, 1.0 + 1e-9, step):
        for wm in np.arange(0.0, 1.0 - wg + 1e-9, step):
            out.append((round(wg, 3), round(wm, 3), round(1 - wg - wm, 3)))
    return out


def grid_sub(step=0.1):
    out = []
    for wl in np.arange(0.0, 1.0 + 1e-9, step):
        for wc in np.arange(0.0, 1.0 - wl + 1e-9, step):
            out.append((round(wl, 3), round(wc, 3), round(1 - wl - wc, 3)))
    return out


def search(D, years, with_sub=False):
    best, bcfg = -1e18, None
    subs = grid_sub() if with_sub else [(PROD['w_lgb'], PROD['w_cb'], PROD['w_xgb'])]
    for wl, wc, wx in subs:
        for wg, wm, ws in grid_top():
            c = dict(PROD, w_lgb=wl, w_cb=wc, w_xgb=wx, w_gbdt=wg, w_mlp=wm, w_mse=ws)
            v, _ = score(c, D, years)
            if v > best:
                best, bcfg = v, c
    return bcfg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-sub', action='store_true', help='GBDT 하위 가중치도 탐색')
    a = ap.parse_args()

    D = load()
    for y in FOLDS:
        print(f'  {y}: 시드 {len(D[y][1])}개 로드')

    p_all, p_per = score(PROD, D, FOLDS)
    print(f'\n[기준] PROD 전체 {p_all:.1f}  폴드별 ' +
          ' '.join(f'{y}:{v:.1f}' for y, v in zip(FOLDS, p_per)))

    print(f'\n[LOFO] 2폴드로 선택 → 남긴 폴드에서 평가  (하위가중치 탐색={a.with_sub})')
    print(f'  {"홀드아웃":>8} {"선택된 top 가중치":>26} {"선택폴드":>10} {"홀드아웃":>10} '
          f'{"PROD":>9} {"차이":>9}')
    deltas = []
    for held in FOLDS:
        sel = [y for y in FOLDS if y != held]
        cfg, sv = search(D, sel, a.with_sub)
        hv, _ = score(cfg, D, [held])
        pv, _ = score(PROD, D, [held])
        deltas.append(hv - pv)
        w = f"{cfg['w_gbdt']:.2f}/{cfg['w_mlp']:.2f}/{cfg['w_mse']:.2f}"
        if a.with_sub:
            w += f" sub {cfg['w_lgb']:.1f}/{cfg['w_cb']:.1f}/{cfg['w_xgb']:.1f}"
        print(f'  {held:>8} {w:>26} {sv:10.1f} {hv:10.1f} {pv:9.1f} {hv-pv:+9.1f}')

    d = np.array(deltas)
    print(f'\n  홀드아웃 평균 차이 {d.mean():+.1f}  (양수 {(d>0).sum()}/3)')
    if (d > 0).all() and d.mean() > 12:
        print('  → 3회 모두 PROD 초과이고 LB 노이즈 바닥(12)도 넘음. 실재하는 개선 후보.')
    elif (d > 0).all():
        print('  → 3회 모두 양수지만 크기가 LB 노이즈 바닥(12) 미만. 개선으로 보지 않음.')
    else:
        print('  → 설정 선택이 폴드를 넘어 전이되지 않는다. PROD 유지가 맞다.')

    print('\n[참고] 전체 3폴드로 최적화했을 때의 설정 (제출용 아님, 상한 확인용)')
    cfg, sv = search(D, FOLDS, a.with_sub)
    print(f'  top {cfg["w_gbdt"]:.2f}/{cfg["w_mlp"]:.2f}/{cfg["w_mse"]:.2f}  '
          f'전체 {sv:.1f}  (PROD 대비 {sv-p_all:+.1f})')
    _, per = score(cfg, D, FOLDS)
    print('  폴드별 ' + ' '.join(f'{y}:{v:.1f}' for y, v in zip(FOLDS, per)))


if __name__ == '__main__':
    main()
