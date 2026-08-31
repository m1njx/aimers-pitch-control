"""
hunt_cells.py — 셀 단위 비선형 보정 대량 탐색 (규정 안전) + 위양성 바닥 동시 측정

`hunt_rowlocal.py` 는 **선형 방향**(잔차에 대한 1차 투영)을 1,899개 훑어 null 을 냈다.
이 스크립트는 **다른 함수 클래스**를 본다: 행을 셀로 나누고 **셀마다 상수 보정**을 적합한다.
선형으로는 못 잡는 비선형·비단조 구조가 여기서만 보인다.

  - 단일 컬럼 10분위 셀, 그리고 저카디널리티 컬럼 쌍의 교차 셀
  - 보정량은 **train 폴드의 잔차 평균**(EB 수축) — 다른 test 행을 쓰지 않으므로 규정 4 안전
  - 셀 정의는 **행 자신의 컬럼**으로만 결정된다

⚠️ 이 함수 클래스는 자유도가 크다(셀 100개 = 파라미터 100개). 폴드 내 이득은 반드시 크게
나온다. 그래서 **폴드 내 값은 인용하지 않고**, 이전 폴드에서 적합해 다음 폴드에서 실현된
값만 본다. 그리고 **무작위 셀 분할(라벨과 무관한 난수 그룹)** 을 대조군으로 섞어
자유도가 만드는 위양성 바닥을 같이 측정한다.

    python3 harness/hunt_cells.py --out outputs/530_hunt_cells.csv
"""
import argparse, glob, itertools, os, time
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
FOLDS = [2021, 2022, 2023, 2024]
K_SHRINK = 200.0          # EB 수축 (셀 표본이 작을수록 0 으로)


def prod(yr):
    ps = []
    for f in sorted(glob.glob(os.path.join(CACHE, f'pred_{yr}_*.npz'))):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def cell_transfer(fa, fb, ca, cb):
    """fa 의 셀별 잔차 평균(EB)을 fb 에 적용. 캘리브레이션 대비 증분."""
    ra = fa['p'] - fa['y']                     # 잔차(과대예측이 양수)
    n = np.bincount(ca); sm = np.bincount(ca, weights=ra)
    adj = sm / (n + K_SHRINK)                  # 0 방향으로 수축
    corr = adj[cb] if cb.max() < len(adj) else np.zeros(len(cb))
    base = skill(fb['p'], fb['y'])
    got = skill(np.clip(fb['p'] - corr, EPS, 1 - EPS), fb['y'])
    return got - base


def groupings(sub, rng, n_random):
    """셀 분할. 반환: {이름: 정수 코드 배열}"""
    d = sub
    out = {}
    lowcard, binned = [], []
    for c in d.columns:
        if c in ('row_id', 'season', 'control_success'):
            continue
        v = d[c]
        nu = v.nunique(dropna=False)
        if nu <= 12:
            out[f'g1:{c}'] = pd.factorize(v.astype(str))[0]
            lowcard.append(c)
        elif v.dtype.kind in 'if':
            q = pd.qcut(v.rank(method='first'), 10, labels=False, duplicates='drop')
            out[f'g1:{c}~q10'] = np.nan_to_num(q, nan=-1).astype(int)
            binned.append(c)
    # 쌍 교차 (저카디널리티 + 분위 구간)
    keys = [k for k in out]
    for k1, k2 in itertools.combinations(keys, 2):
        a, b = out[k1], out[k2]
        code = a * (b.max() + 1) + b
        if len(np.unique(code)) <= 400:
            out[f'g2:{k1[3:]}|{k2[3:]}'] = pd.factorize(code)[0]
    # 🎲 무작위 셀 분할 — 자유도가 만드는 위양성 바닥
    for i in range(n_random):
        k = int(rng.integers(10, 200))
        out[f'RANDOM:{i:03d}'] = rng.integers(0, k, len(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(LG, 'outputs/530_hunt_cells.csv'))
    ap.add_argument('--random', type=int, default=150)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    F, G = {}, {}
    for yr in FOLDS:
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(float)
        sub = df[df.season == yr].reset_index(drop=True)
        m = (sub.game_type.values == 'R') if yr == 2023 else np.ones(len(y), bool)
        F[yr] = dict(y=y[m], p=prod(yr)[m])
        G[yr] = {k: np.asarray(v)[m] for k, v in
                 groupings(sub, np.random.default_rng(2000 + yr), a.random).items()}
        print(f'fold {yr}: n={m.sum():,}  분할 {len(G[yr]):,}개', flush=True)

    keys = [k for k in G[FOLDS[0]] if all(k in G[y] for y in FOLDS)]
    print(f'\n공통 분할 {len(keys):,}개 (무작위 {a.random}개 포함) — 판정 시작', flush=True)

    rows, t0 = [], time.time()
    for i, k in enumerate(keys, 1):
        d = []
        for j in range(1, 4):
            ya, yb = FOLDS[j - 1], FOLDS[j]
            ca, cb = G[ya][k], G[yb][k]
            if ca.min() < 0 or cb.min() < 0:
                ca = ca - min(0, ca.min()); cb = cb - min(0, cb.min())
            M = max(ca.max(), cb.max()) + 1
            ca2 = np.clip(ca, 0, M - 1); cb2 = np.clip(cb, 0, M - 1)
            n = np.bincount(ca2, minlength=M)
            sm = np.bincount(ca2, weights=(F[ya]['p'] - F[ya]['y']), minlength=M)
            adj = sm / (n + K_SHRINK)
            got = skill(np.clip(F[yb]['p'] - adj[cb2], EPS, 1 - EPS), F[yb]['y'])
            d.append(got - skill(F[yb]['p'], F[yb]['y']))
        rows.append(dict(name=k, ncell=int(len(np.unique(G[FOLDS[0]][k]))),
                         t21_22=d[0], t22_23R=d[1], t23R_24=d[2],
                         mean=float(np.mean(d)), all_pos=bool(all(x > 0 for x in d)),
                         is_random=k.startswith('RANDOM')))
        if i % 300 == 0:
            print(f'  {i:,}/{len(keys):,}  ({time.time()-t0:.0f}s)', flush=True)

    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    real, rand = R[~R.is_random], R[R.is_random]
    print(f'\n=== 완료: 실분할 {len(real):,} / 무작위 {len(rand):,}  ({time.time()-t0:.0f}s) ===')
    if len(rand):
        print(f'\n[위양성 바닥] 무작위 셀 분할 {len(rand)}개')
        print(f'  3전이 전부 양수 비율 : {rand.all_pos.mean()*100:.1f}%')
        print(f'  2022→2023R 최대치    : {rand.t22_23R.max():+.2f}')
        rp = rand[rand.all_pos]
        print(f'  3전이 양수 중 2022→2023R 최대: '
              f'{rp.t22_23R.max() if len(rp) else float("nan"):+.2f}')
    thr = (rand[rand.all_pos].t22_23R.max()
           if len(rand) and rand.all_pos.any() else 0.0)
    P = real[real.all_pos].sort_values('t22_23R', ascending=False)
    print(f'\n[실분할] 3전이 전부 양수 = {len(P):,} / {len(real):,}')
    print(P.head(15).to_string(index=False))
    win = P[P.t22_23R > max(12.0, thr)]
    print(f'\n최종 게이트(3전이 양수 & 2022→2023R > max(12, 무작위최고={thr:+.2f})): {len(win)}개')
    print(win.to_string(index=False) if len(win) else '  없음 — 정직한 null.')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
