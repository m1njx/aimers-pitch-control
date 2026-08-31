"""
screen_stack.py -- the ceiling of recombining EVERYTHING.

Individually no cached direction clears the noise floor on 2023-R
(screen_stability.py).  This asks the union question instead: if we are allowed
to re-weight every arm this project has ever trained, with unconstrained real
coefficients, on top of a refitted calibration -- how much is there?

Two numbers per fold:
  ORACLE   coefficients fitted in-fold          -> hard upper bound on the axis
  HONEST   coefficients fitted on the PREVIOUS fold, applied forward
           (+ ridge shrinkage, since with ~20 directions the era transfer, not
            the sample size, is what breaks)

If ORACLE on 2023-R is small, the axis is closed and no stacking of existing
material reaches 1150.  If ORACLE is large but HONEST is not, the information
is era-specific -- the same verdict this project has reached on every other
time-varying axis.
"""
import os, glob
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
H = os.path.join(LG, 'harness')
CACHE = os.path.join(H, 'cache')
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


def skill(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def load_arm(cdir, year, component=None):
    fs = sorted(glob.glob(os.path.join(cdir, f'pred_{year}_*.npz')))
    if not fs:
        return None
    ps = []
    for f in fs:
        P = dict(np.load(f, allow_pickle=True))
        if component is not None:
            if component not in P:
                return None
            ps.append(np.clip(np.asarray(P[component], np.float64), EPS, 1 - EPS))
        else:
            ps.append(prod_blend(P))
    return np.mean(ps, axis=0)


def load_teamB(year, tag):
    fs = sorted(glob.glob(os.path.join(LG, 'teamB/out/preds', f'{tag}_f{year}_s*.npy')))
    return np.mean([np.load(f).astype(np.float64) for f in fs], axis=0) if fs else None


def design(yr, m, base, names):
    cols = [np.ones_like(base), base - 0.5]
    used = ['const', 'scale']
    for name, fn in names:
        p = fn(yr)
        if p is None or len(p) != len(m):
            continue
        cols.append(p[m] - base)
        used.append(name)
    return np.stack(cols, 1), used


def main():
    gt = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'game_type'])
    names = [(os.path.basename(c), lambda yr, c=c: load_arm(c, yr))
             for c in sorted(glob.glob(os.path.join(H, 'cache*')))
             if os.path.basename(c) != 'cache']
    names += [(f'own:{k}', lambda yr, k=k: load_arm(CACHE, yr, k))
              for k in ('lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp')]
    names += [(f'teamB:{t}', lambda yr, t=t: load_teamB(yr, t))
              for t in ('l2384',)]

    F = {}
    for yr in (2021, 2022, 2023, 2024):
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        mm = (gt.loc[gt.season == yr, 'game_type'].values == 'R') if yr == 2023 \
            else np.ones(len(y), bool)
        b = load_arm(CACHE, yr)[mm]
        X, used = design(yr, mm, b, names)
        F[yr] = dict(y=y[mm], base=b, X=X, used=used)
        print(f'{yr}: n={mm.sum():,}  directions={X.shape[1] - 2}  s_A={skill(b, y[mm]):.2f}')

    def fit(X, y, base, lam=0.0):
        A = X.T @ X + lam * np.eye(X.shape[1]) * len(y)
        return np.linalg.solve(A, X.T @ (y - base))

    print('\n=== stacking every cached direction, unconstrained coefficients ===')
    print(f'{"fold":>6} {"s_A":>9} {"calib":>9} {"ORACLE":>9} {"oracle-calib":>13} '
          f'{"HONEST(prev)":>13} {"honest ridge":>13}')
    order = [2021, 2022, 2023, 2024]
    for i, yr in enumerate(order):
        d = F[yr]
        bc = fit(d['X'][:, :2], d['y'], d['base'])
        s_cal = skill(np.clip(d['base'] + d['X'][:, :2] @ bc, EPS, 1 - EPS), d['y'])
        bo = fit(d['X'], d['y'], d['base'])
        s_or = skill(np.clip(d['base'] + d['X'] @ bo, EPS, 1 - EPS), d['y'])
        hon = hon_r = float('nan')
        if i > 0:
            pv = F[order[i - 1]]
            shared = [c for c in pv['used'] if c in d['used']]
            ip = [pv['used'].index(c) for c in shared]
            ic = [d['used'].index(c) for c in shared]
            bp = fit(pv['X'][:, ip], pv['y'], pv['base'])
            hon = skill(np.clip(d['base'] + d['X'][:, ic] @ bp, EPS, 1 - EPS), d['y']) - s_cal
            bpr = fit(pv['X'][:, ip], pv['y'], pv['base'], lam=1e-4)
            hon_r = skill(np.clip(d['base'] + d['X'][:, ic] @ bpr, EPS, 1 - EPS), d['y']) - s_cal
        print(f'{yr:>6} {skill(d["base"], d["y"]):9.2f} {s_cal:9.2f} {s_or:9.2f} '
              f'{s_or - s_cal:13.2f} {hon:13.2f} {hon_r:13.2f}')

    # what rho would 1150 need?  (LB anchor: team v29 = 1082)
    V = 0.25
    for target, cur in ((1150, 1082), (1100, 1082)):
        need = target - cur
        BS = V * (1 - cur / 1e5)
        rho = np.sqrt(need * BS * V / 1e5) / np.sqrt(BS * V) if False else \
            np.sqrt(need / (1e5 * BS / V))
        print(f'\nLB {cur} -> {target} needs a direction with '
              f'corr(residual, u) = {rho:.4f}  ({rho * 100:.2f}%)')


if __name__ == '__main__':
    main()
