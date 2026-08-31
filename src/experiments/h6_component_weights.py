"""
h6_component_weights.py -- is there anything left in our own 5 component weights?

527's stability table found exactly one direction whose coefficient kept its
sign in all four folds AND transferred positively across all three fold
transitions: re-weighting our own components (+11.98 / +3.63 / +18.85).  Every
other axis flipped sign.  So this is the last live candidate, and it is cheap:
no retraining, the cache already holds every component.

Because the production affine calibration comes AFTER the blend and the weights
sum to 1, calibrating each component first and then blending is identical to
blending then calibrating.  So with SCALE/SHIFT pinned at production values the
problem is exactly a sum-to-one quadratic optimisation -- solvable in closed
form, no search.

Criteria are pre-registered in outputs/526 (08-27 21:40):
  G1  > +12 over production weights on 2023 R rows
  G2  > +12 over production weights on 2024
  G3  must not contradict the leaderboard: v42 (.40/.40/.20) scored 1032.1 and
      v50 (.25/.50/.25) scored 1032.0 -- a large weight change worth 0.1 real
      LB points.  A local claim of +12 inside that region indicts the local
      measurement, not the leaderboard.
"""
import os, glob
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
EPS = 1e-6
SCALE, SHIFT = 1.10, -0.0045192086
SHIFTS = dict(lgb_bin=-0.007, cb_bin=-0.008, xgb_bin=-0.006, lgb_mse=0.0, mlp=0.0)
COMP = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
# production effective weights over the 5 components
W_PROD = np.array([0.40 * 0.20, 0.40 * 0.72, 0.40 * 0.08, 0.20, 0.40])


def components(yr):
    """Seed-bagged, per-model shifted, production-calibrated component matrix."""
    fs = sorted(glob.glob(os.path.join(CACHE, f'pred_{yr}_*.npz')))
    acc = {k: [] for k in COMP}
    for f in fs:
        P = dict(np.load(f, allow_pickle=True))
        for k in COMP:
            acc[k].append(np.clip(np.asarray(P[k], np.float64) + SHIFTS[k], EPS, 1 - EPS))
    cols = [np.mean(acc[k], 0) for k in COMP]
    return np.stack([0.5 + SCALE * (c - 0.5) + SHIFT for c in cols], 1)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def gram(X, y):
    r = y.mean(); V = r * (1 - r)
    n = X.shape[1]
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 1e5 * (1 - ((X[:, i] - y) * (X[:, j] - y)).mean() / V)
    return M


def opt_w(M, nonneg=True):
    n = M.shape[0]
    A = np.block([[2 * M, -np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    w = np.linalg.solve(A, np.concatenate([np.zeros(n), [1.0]]))[:n]
    if not nonneg or (w >= -1e-9).all():
        return w
    from itertools import combinations
    best, bw = -1e18, None
    for k in range(1, n + 1):
        for S in combinations(range(n), k):
            S = list(S)
            Ms = M[np.ix_(S, S)]
            As = np.block([[2 * Ms, -np.ones((k, 1))],
                           [np.ones((1, k)), np.zeros((1, 1))]])
            try:
                ws = np.linalg.solve(As, np.concatenate([np.zeros(k), [1.0]]))[:k]
            except np.linalg.LinAlgError:
                continue
            if (ws < -1e-9).any():
                continue
            full = np.zeros(n); full[S] = ws
            v = full @ M @ full
            if v > best:
                best, bw = v, full
    return bw


def main():
    gt = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'game_type'])
    F = {}
    for yr in (2021, 2022, 2023, 2024):
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        m = (gt.loc[gt.season == yr, 'game_type'].values == 'R') if yr == 2023 \
            else np.ones(len(y), bool)
        X = components(yr)[m]
        F[yr] = dict(y=y[m], X=X, M=gram(X, y[m]))

    print(f'프로덕션 유효가중: ' + ' '.join(f'{c}={w:.3f}' for c, w in zip(COMP, W_PROD)))
    print(f'\n{"fold":>6} {"PROD":>9} {"폴드내 최적":>11} {"(상한)":>8}   최적 w')
    for yr, d in F.items():
        s_p = skill(d['X'] @ W_PROD, d['y'])
        w = opt_w(d['M'])
        print(f'{yr:>6} {s_p:9.2f} {float(w @ d["M"] @ w):11.2f} '
              f'{float(w @ d["M"] @ w) - s_p:+8.2f}   [' +
              ' '.join(f'{v:.3f}' for v in w) + ']')

    print('\n=== 배포형: 이전 폴드에서만 적합 → 대상 폴드 평가 (심플렉스 제약) ===')
    print(f'{"적합":>22} {"대상":>10} {"PROD":>9} {"적합가중":>9} {"Δ":>8}   적합된 w')
    order = [2021, 2022, 2023, 2024]
    for i in range(1, 4):
        cur = order[i]
        for label, fit_years in ((f'{order[i-1]} 단독', [order[i - 1]]),
                                 ('이전 전부 합산', order[:i])):
            M = sum(F[y]['M'] * len(F[y]['y']) for y in fit_years) / \
                sum(len(F[y]['y']) for y in fit_years)
            w = opt_w(M)
            d = F[cur]
            s_p, s_w = skill(d['X'] @ W_PROD, d['y']), skill(d['X'] @ w, d['y'])
            tag = f'{cur}' + (' R행' if cur == 2023 else '')
            print(f'{label:>22} {tag:>10} {s_p:9.2f} {s_w:9.2f} {s_w - s_p:+8.2f}   [' +
                  ' '.join(f'{v:.3f}' for v in w) + ']')

    # G3: where do v42 and v50 sit, and what does the real LB say about the gap?
    print('\n=== G3: LB 앵커 반증 ===')
    for name, top in (('v42 .40/.40/.20', (0.40, 0.40, 0.20)),
                      ('v50 .25/.50/.25', (0.25, 0.50, 0.25))):
        w = np.array([top[0] * 0.20, top[0] * 0.72, top[0] * 0.08, top[2], top[1]])
        row = ' '.join(f'{yr}:{skill(F[yr]["X"] @ w, F[yr]["y"]):8.2f}' for yr in order)
        print(f'  {name}  {row}')
    print('  실전 LB:  v42 = 1032.1,  v50 = 1032.0  (차이 0.1점)')


if __name__ == '__main__':
    main()
