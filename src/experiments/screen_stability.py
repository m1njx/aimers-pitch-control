"""
screen_stability.py -- per-fold stability of the orthogonalized residual
projection.

screen_resid2.py showed one arm (`mtl10`) clearing the noise floor on the
2022->2023R transfer and then reversing on 2023->2024.  A two-point transfer
cannot distinguish "real but noisy" from "era-specific artifact", so this
script fits the coefficient IN-FOLD on every available fold and prints the
whole row.  A direction that carries transferable information has a coefficient
with a stable sign and a comparable magnitude across folds; one that does not
is a per-era fit and is worthless for 2025 no matter how large any single fold
looks.

Model per fold (exact, Brier is quadratic in p):
    y - p_A  ~  a*1 + b*(p_A-0.5) + t*u        u = p_cand - p_A
`a,b` absorb the production SCALE/SHIFT knobs so `t` measures NEW information
only.  `incr` is the in-fold gain of adding u on top of the refitted
calibration -- an upper bound, printed to show the size of the prize, never as
a result.

2023 is scored on R rows only, and is the primary judgment fold (526, 19:20).
"""
import os, glob
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
H = os.path.join(LG, 'harness')
CACHE = os.path.join(H, 'cache')
EPS = 1e-6
FOLDS = [2021, 2022, 2023, 2024]
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


def main():
    gt = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'game_type'])
    m, y, base = {}, {}, {}
    for yr in FOLDS:
        yy = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        mm = (gt.loc[gt.season == yr, 'game_type'].values == 'R') if yr == 2023 \
            else np.ones(len(yy), bool)
        m[yr], y[yr] = mm, yy[mm]
        base[yr] = load_arm(CACHE, yr)[mm]

    print('fold  n        rbar    s_A(prod)  +calib(in-fold)')
    for yr in FOLDS:
        X = np.stack([np.ones_like(base[yr]), base[yr] - 0.5], 1)
        b, *_ = np.linalg.lstsq(X, y[yr] - base[yr], rcond=None)
        pc = np.clip(base[yr] + X @ b, EPS, 1 - EPS)
        print(f'{yr}  {len(y[yr]):>7,}  {y[yr].mean():.4f}  {skill(base[yr], y[yr]):9.2f}  '
              f'{skill(pc, y[yr]):9.2f}   shift={b[0]:+.5f} scale_adj={b[1]:+.4f}')

    cands = [(os.path.basename(c), lambda yr, c=c: load_arm(c, yr))
             for c in sorted(glob.glob(os.path.join(H, 'cache*')))
             if os.path.basename(c) != 'cache']
    cands += [(f'own:{k}', lambda yr, k=k: load_arm(CACHE, yr, k))
              for k in ('lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp')]
    cands += [(f'teamB:{t}', lambda yr, t=t: load_teamB(yr, t))
              for t in ('l2384', 'lgbm_isf', 'base_xgb')]

    rows = []
    for name, fn in cands:
        rec = dict(arm=name)
        ts, gs = [], []
        for yr in FOLDS:
            p = fn(yr)
            if p is None or len(p) != len(m[yr]):
                rec[f't{yr}'] = np.nan; rec[f'g{yr}'] = np.nan
                continue
            u = p[m[yr]] - base[yr]
            X = np.stack([np.ones_like(base[yr]), base[yr] - 0.5, u], 1)
            b, *_ = np.linalg.lstsq(X, y[yr] - base[yr], rcond=None)
            b2, *_ = np.linalg.lstsq(X[:, :2], y[yr] - base[yr], rcond=None)
            g = (skill(np.clip(base[yr] + X @ b, EPS, 1 - EPS), y[yr])
                 - skill(np.clip(base[yr] + X[:, :2] @ b2, EPS, 1 - EPS), y[yr]))
            rec[f't{yr}'] = b[2]; rec[f'g{yr}'] = g
            ts.append(b[2]); gs.append(g)
        if len(ts) < 3:
            continue
        rec['n_folds'] = len(ts)
        rec['sign_consistent'] = bool(all(np.sign(t) == np.sign(ts[0]) for t in ts))
        rec['t_mean'] = float(np.mean(ts))
        rec['t_cv'] = float(np.std(ts) / (abs(np.mean(ts)) + 1e-12))
        rec['g_min'] = float(np.min(gs))
        rows.append(rec)

    df = pd.DataFrame(rows)
    df = df.sort_values(['sign_consistent', 'g_min'], ascending=[False, False])
    out = os.path.join(LG, 'outputs/527_screen_stability.csv')
    df.to_csv(out, index=False)
    cols = ['arm', 't2021', 't2022', 't2023', 't2024',
            'g2021', 'g2022', 'g2023', 'g2024', 'sign_consistent', 't_cv']
    pd.set_option('display.width', 200)
    print('\n=== in-fold orthogonalized coefficient per fold (2023 = R rows) ===')
    print(df[cols].to_string(index=False, float_format=lambda v: f'{v:8.3f}'))
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
