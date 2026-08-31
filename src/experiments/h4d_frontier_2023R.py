"""
h4d_frontier_2023R.py -- re-judge the third-arm frontier on the LB-proxy fold.

H4/H4b/H4c judged candidate third arms on folds 2022 and 2024 only.  526 (19:20)
later established that those two folds carry the WRONG SIGN of A-B relative to
the leaderboard and a D_AB 1.4-2x too large, while 2023 R-rows match the LB on
all four structural indicators.  So the frontier verdict was never taken on the
fold that actually resembles 2025.

Criteria are pre-registered in outputs/526 (08-27 20:15) and are not edited here:
  G1  d_AC >= 0.020 AND d_BC >= 0.020
  G2  optimal-weight 3-arm minus optimal-weight 2-arm(A,B) > +12, with w_C > 0.05
  G3  weights fitted on fold 2022 must still beat the 2-arm blend on 2023R
All three must pass.
"""
import os, glob
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
PREDS = os.path.join(LG, 'teamB/out/preds')
EPS = 1e-6
PROD = dict(w_lgb=0.20, w_cb=0.72, w_xgb=0.08,
            s_lgb=-0.007, s_cb=-0.008, s_xgb=-0.006,
            w_gbdt=0.40, w_mlp=0.40, w_mse=0.20,
            scale=1.10, shift=-0.0045192086)


def prod_blend(P, c=PROD):
    lgb_ = np.clip(P['lgb_bin'] + c['s_lgb'], EPS, 1 - EPS)
    cb_ = np.clip(P['cb_bin'] + c['s_cb'], EPS, 1 - EPS)
    xgb_ = np.clip(P['xgb_bin'] + c['s_xgb'], EPS, 1 - EPS)
    gbdt = np.clip(c['w_lgb'] * lgb_ + c['w_cb'] * cb_ + c['w_xgb'] * xgb_, EPS, 1 - EPS)
    raw = (c['w_gbdt'] * gbdt + c['w_mlp'] * P['mlp']
           + c['w_mse'] * np.clip(P['lgb_mse'], EPS, 1 - EPS))
    return np.clip(0.5 + c['scale'] * (raw - 0.5) + c['shift'], EPS, 1 - EPS)


def ours(yr):
    return np.mean([prod_blend(dict(np.load(f, allow_pickle=True)))
                    for f in sorted(glob.glob(os.path.join(CACHE, f'pred_{yr}_*.npz')))], 0)


def teamB(yr, tag):
    fs = sorted(glob.glob(os.path.join(PREDS, f'{tag}_f{yr}_s*.npy')))
    return np.mean([np.load(f).astype(np.float64) for f in fs], 0) if fs else None


def skill(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def gram(ps, y):
    """M_ij of the exact quadratic form: skill of sum(w_i p_i) = w'Mw, sum(w)=1."""
    r = y.mean(); V = r * (1 - r)
    n = len(ps)
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = 1e5 * (1 - ((ps[i] - y) * (ps[j] - y)).mean() / V)
    return M


def opt_w(M, nonneg=True):
    n = M.shape[0]
    A = np.block([[2 * M, -np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    w = np.linalg.solve(A, np.concatenate([np.zeros(n), [1.0]]))[:n]
    if nonneg and (w < 0).any():           # project onto the simplex by brute force
        best, bw = -1e18, None
        for k in range(1, n + 1):
            from itertools import combinations
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
        return bw, best
    return w, float(w @ M @ w)


def main():
    gt = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'game_type'])
    out = []
    for yr, tagB in ((2023, 'l2384R'), (2022, 'l2384')):
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        m = (gt.loc[gt.season == yr, 'game_type'].values == 'R') if yr == 2023 \
            else np.ones(len(y), bool)
        A, B = ours(yr)[m], teamB(yr, tagB)[m]
        y = y[m]
        print(f'\n=== fold {yr} ({"R rows only" if yr == 2023 else "all rows"}, '
              f'n={len(y):,}, rbar={y.mean():.4f}) ===')
        print(f'  s_A={skill(A, y):8.2f}   s_B({tagB})={skill(B, y):8.2f}')
        M2 = gram([A, B], y)
        w2, u2 = opt_w(M2)
        d_AB = float(np.sqrt(((A - B) ** 2).mean()))
        print(f'  2-arm optimum {u2:8.2f}  w=[{w2[0]:.3f}, {w2[1]:.3f}]  d_AB={d_AB:.4f}')
        for tag in ('lgbm_isf', 'base_xgb', 'lgbmT'):
            C = teamB(yr, tag)
            if C is None:
                continue
            C = C[m]
            d_AC = float(np.sqrt(((A - C) ** 2).mean()))
            d_BC = float(np.sqrt(((B - C) ** 2).mean()))
            M3 = gram([A, B, C], y)
            w3, u3 = opt_w(M3)
            print(f'  {tag:10s} s_C={skill(C, y):8.2f}  d_AC={d_AC:.4f} d_BC={d_BC:.4f}  '
                  f'3-arm={u3:8.2f} (Δ{u3 - u2:+7.2f})  w=[{w3[0]:.3f},{w3[1]:.3f},{w3[2]:.3f}]')
            out.append(dict(fold=yr, arm=tag, s_C=skill(C, y), d_AC=d_AC, d_BC=d_BC,
                            u2=u2, u3=u3, delta=u3 - u2, w=w3.tolist()))

    # ---- G3: weights fitted on 2022, applied to 2023R -------------------
    print('\n=== G3 transfer: weights fitted on 2022 -> applied to 2023 R rows ===')
    y23 = np.load(os.path.join(CACHE, 'y_2023.npy')).astype(np.float64)
    m23 = (gt.loc[gt.season == 2023, 'game_type'].values == 'R')
    A23, B23, y23 = ours(2023)[m23], teamB(2023, 'l2384R')[m23], y23[m23]
    y22 = np.load(os.path.join(CACHE, 'y_2022.npy')).astype(np.float64)
    A22, B22 = ours(2022), teamB(2022, 'l2384')
    u2_23 = opt_w(gram([A23, B23], y23))[1]
    for tag in ('lgbm_isf', 'base_xgb'):
        C22, C23 = teamB(2022, tag), teamB(2023, tag)
        if C22 is None or C23 is None:
            continue
        C23 = C23[m23]
        w, _ = opt_w(gram([A22, B22, C22], y22))
        p = np.clip(w[0] * A23 + w[1] * B23 + w[2] * C23, EPS, 1 - EPS)
        s = skill(p, y23)
        print(f'  {tag:10s} w_2022=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}] -> 2023R {s:8.2f} '
              f'(vs 2-arm optimum {u2_23:.2f}, Δ{s - u2_23:+7.2f})')

    pd.DataFrame(out).to_csv(os.path.join(LG, 'outputs/527_h4d_frontier.csv'), index=False)


if __name__ == '__main__':
    main()
