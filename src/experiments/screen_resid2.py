"""
screen_resid2.py -- the same residual-projection screen, but ORTHOGONALIZED
against the calibration knobs the production pipeline already owns.

screen_resid.py found +30 transferred gain for the `era` family at a NEGATIVE
coefficient.  Before believing that, it has to survive the obvious null: the
production affine calibration (SCALE, SHIFT) is a 2-parameter family spanning
the directions  1  and  (p_A - 0.5).  Any candidate direction u that is mostly
a linear combination of those two is not new information -- it is the scale
knob wearing a costume, and 526 (08-27 06:30) already logged that trap.

So here the model is

    p(t) = p_A + a*1 + b*(p_A - 0.5) + t*u

fitted by ordinary least squares on the FIT fold (exact: Brier is quadratic),
then evaluated on the TEST fold.  We report

    base        skill of p_A as cached
    calib       skill after fitting (a,b) on FIT, applied to TEST   <- the null
    full        skill after fitting (a,b,t) on FIT, applied to TEST
    incr        full - calib   <- the ONLY number that means "new information"

Judgment fold is 2023 R-rows (closest analogue of the 2025 LB).
"""
import os, glob, argparse
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
    mse = np.clip(P['lgb_mse'], EPS, 1 - EPS)
    raw = c['w_gbdt'] * gbdt + c['w_mlp'] * P['mlp'] + c['w_mse'] * mse
    return np.clip(0.5 + c['scale'] * (raw - 0.5) + c['shift'], EPS, 1 - EPS)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def row_mask(year):
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'), usecols=['season', 'game_type'])
    return (df.loc[df.season == year, 'game_type'].values == 'R')


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


def load_teamB(year, tag='l2384'):
    fs = sorted(glob.glob(os.path.join(LG, 'teamB/out/preds', f'{tag}_f{year}_s*.npy')))
    return np.mean([np.load(f).astype(np.float64) for f in fs], axis=0) if fs else None


def fit_ls(pA, y, extra=None):
    """OLS of (y - pA) on [1, pA-0.5] (+ extra). Returns coefficient vector."""
    cols = [np.ones_like(pA), pA - 0.5]
    if extra is not None:
        cols.append(extra)
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, y - pA, rcond=None)
    return beta


def apply_ls(pA, beta, extra=None):
    p = pA + beta[0] + beta[1] * (pA - 0.5)
    if extra is not None:
        p = p + beta[2] * extra
    return np.clip(p, EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', type=int, default=2022)
    ap.add_argument('--test', type=int, default=2023)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(LG, f'outputs/527_screen_orth_{a.fit}to{a.test}.csv')

    m, y, base = {}, {}, {}
    for yr in (a.fit, a.test):
        yy = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        mm = row_mask(yr) if yr == 2023 else np.ones(len(yy), bool)
        assert len(mm) == len(yy)
        m[yr], y[yr] = mm, yy[mm]
        base[yr] = load_arm(CACHE, yr)[mm]

    # the null model: calibration only, fitted on FIT, applied to TEST
    b_cal = fit_ls(base[a.fit], y[a.fit])
    s_base = skill(base[a.test], y[a.test])
    s_cal = skill(apply_ls(base[a.test], b_cal), y[a.test])
    print(f'fit={a.fit} test={a.test}  n_test={len(y[a.test]):,}')
    print(f'  base (as cached)        {s_base:9.2f}')
    print(f'  + calib fitted on {a.fit}  {s_cal:9.2f}   ({s_cal - s_base:+.2f})   '
          f'beta=[{b_cal[0]:+.5f}, {b_cal[1]:+.4f}]')
    print(f'  {"arm":22s} {"t_fit":>9s} {"full":>9s} {"INCR":>9s} {"incr_half":>9s}')

    cands = [(os.path.basename(c), lambda yr, c=c: load_arm(c, yr))
             for c in sorted(glob.glob(os.path.join(H, 'cache*')))
             if os.path.basename(c) != 'cache']
    cands += [(f'own:{k}', lambda yr, k=k: load_arm(CACHE, yr, k))
              for k in ('lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp')]
    cands += [(f'teamB:{t}', lambda yr, t=t: load_teamB(yr, t))
              for t in ('l2384', 'lgbm_isf', 'base_xgb', 'l2384R')]

    rows = []
    for name, fn in cands:
        pf, pt = fn(a.fit), fn(a.test)
        if pf is None or pt is None:
            continue
        pf, pt = pf[m[a.fit]], pt[m[a.test]]
        if len(pf) != len(y[a.fit]) or len(pt) != len(y[a.test]):
            continue
        uf, ut = pf - base[a.fit], pt - base[a.test]
        b = fit_ls(base[a.fit], y[a.fit], uf)
        s_full = skill(apply_ls(base[a.test], b, ut), y[a.test])
        bh = b.copy(); bh[2] *= 0.5
        s_half = skill(apply_ls(base[a.test], bh, ut), y[a.test])
        rows.append(dict(arm=name, t_fit=b[2], full=s_full,
                         incr=s_full - s_cal, incr_half=s_half - s_cal,
                         # in-fold ceiling on TEST (upper bound, not a result)
                         ceil=skill(apply_ls(base[a.test],
                                             fit_ls(base[a.test], y[a.test], ut), ut),
                                    y[a.test]) - s_cal))
        print(f"  {name:22s} {b[2]:+9.3f} {s_full:9.2f} {rows[-1]['incr']:+9.2f} "
              f"{rows[-1]['incr_half']:+9.2f}")

    df = pd.DataFrame(rows).sort_values('incr', ascending=False)
    df.to_csv(out, index=False)
    print(f'\n-> {out}')
    print(df.head(12).to_string(index=False))


if __name__ == '__main__':
    main()
