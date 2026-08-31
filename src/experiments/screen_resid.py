"""
screen_resid.py -- residual-projection screen over every cached arm.

WHY THIS IS NEW
---------------
Brier score is *exactly quadratic* in the prediction vector, so for any second
prediction vector p_C the skill of  p = p_A + t*(p_C - p_A)  is an exact
parabola in t, for EVERY real t -- including t<0 and t>1.  Every blend search
in this project so far constrained the weights to the simplex (t in [0,1]).
That constraint is not implied by the math and it threw away half the search
space: an arm that is *worse* than A can still carry skill if you subtract it.

The gain of the optimal 1-D line search along u = p_C - p_A is, in closed form,

    dSkill = 1e5 * mean(r*u)^2 / ( mean(u^2) * V ),     r = p_A - y,  V = rbar(1-rbar)

i.e. it depends only on the CORRELATION between our residual and the
disagreement direction.  Solving dSkill = +68 (what 1082 -> 1150 needs) gives
rho = corr(r, u) ~= 0.026.  So the whole 1150 question reduces to one number
per candidate direction, computable in milliseconds from cached predictions.

Because t is a SINGLE parameter fitted on ~250k rows, in-fold overfitting is
O(1/n) and negligible.  The only real risk is era transfer, so this script
never reports an in-fold number as a result: it fits t on one fold and reports
the gain realized on a later fold.

DISCIPLINE
----------
* fold 2023 is scored on R rows only (F rows changed regime in 2023; see
  outputs/526 08-27 18:45).
* 2023-R is the closest analogue of the 2025 LB (matches corr, d_AB, D_AB and
  the sign of A-B; see 526 08-27 19:20), so it is the primary judgment fold.
* predictions are seed-bagged before scoring, per the search protocol.
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
H = os.path.join(LG, 'harness')
CACHE = os.path.join(H, 'cache')

PROD = dict(w_lgb=0.20, w_cb=0.72, w_xgb=0.08,
            s_lgb=-0.007, s_cb=-0.008, s_xgb=-0.006,
            w_gbdt=0.40, w_mlp=0.40, w_mse=0.20,
            scale=1.10, shift=-0.0045192086)
EPS = 1e-6


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
    """R-row mask for the fold, in build_cache.py's row order (raw CSV order)."""
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     usecols=['season', 'game_type'])
    gt = df.loc[df.season == year, 'game_type'].values
    return (gt == 'R')


def load_arm(cdir, year, component=None):
    """Seed-bagged prediction for one cached arm on one fold, or None."""
    fs = sorted(glob.glob(os.path.join(cdir, f'pred_{year}_*.npz')))
    if not fs:
        return None
    ps = []
    for f in fs:
        P = dict(np.load(f, allow_pickle=True))
        if component is not None:
            if component not in P:
                return None
            ps.append(np.clip(np.asarray(P[component], dtype=np.float64), EPS, 1 - EPS))
        else:
            ps.append(prod_blend(P))
    return np.mean(ps, axis=0)


def load_teamB(year, tag='l2384'):
    d = os.path.join(LG, 'teamB/out/preds')
    fs = sorted(glob.glob(os.path.join(d, f'{tag}_f{year}_s*.npy')))
    if not fs:
        return None
    return np.mean([np.load(f).astype(np.float64) for f in fs], axis=0)


def line_stats(pA, pC, y):
    """Exact optimum of p_A + t*(p_C - p_A). Returns dict."""
    u = pC - pA
    r = pA - y
    uu = float((u * u).mean())
    if uu <= 0:
        return None
    ru = float((r * u).mean())
    V = float(y.mean() * (1 - y.mean()))
    t = -ru / uu
    gain = 1e5 * ru * ru / (uu * V)
    rho = ru / np.sqrt(uu * float((r * r).mean()))
    return dict(t=t, gain=gain, rho=float(rho), d=float(np.sqrt(uu)),
                s_C=float(skill(pC, y)))


def apply_t(pA, pC, t, y):
    p = np.clip(pA + t * (pC - pA), EPS, 1 - EPS)
    return float(skill(p, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', type=int, default=2022)
    ap.add_argument('--test', type=int, default=2023)
    ap.add_argument('--out', default=os.path.join(LG, 'outputs/527_screen_resid.csv'))
    a = ap.parse_args()

    masks, ys, bases = {}, {}, {}
    for yr in (a.fit, a.test):
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        m = row_mask(yr) if yr == 2023 else np.ones(len(y), bool)
        assert len(m) == len(y), f'mask/label length mismatch {yr}: {len(m)} vs {len(y)}'
        masks[yr], ys[yr] = m, y[m]
        b = load_arm(CACHE, yr)
        assert b is not None, f'no base cache for {yr}'
        bases[yr] = b[m]
        print(f'fold {yr}: n={m.sum():,}/{len(m):,}  rbar={ys[yr].mean():.4f}  '
              f's_A={skill(bases[yr], ys[yr]):.2f}', flush=True)

    # ---- candidate directions -------------------------------------------
    cands = []
    for cdir in sorted(glob.glob(os.path.join(H, 'cache*'))):
        name = os.path.basename(cdir)
        if name == 'cache':
            continue
        cands.append((name, lambda yr, c=cdir: load_arm(c, yr)))
    for comp in ('lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp'):
        cands.append((f'own:{comp}', lambda yr, k=comp: load_arm(CACHE, yr, k)))
    for tag in ('l2384', 'lgbm_isf', 'base_xgb', 'l2384R'):
        cands.append((f'teamB:{tag}', lambda yr, t=tag: load_teamB(yr, t)))

    rows = []
    for name, fn in cands:
        pf, pt = fn(a.fit), fn(a.test)
        if pf is None or pt is None:
            continue
        pf, pt = pf[masks[a.fit]], pt[masks[a.test]]
        if len(pf) != len(ys[a.fit]) or len(pt) != len(ys[a.test]):
            print(f'  skip {name}: length mismatch')
            continue
        sf = line_stats(bases[a.fit], pf, ys[a.fit])
        st = line_stats(bases[a.test], pt, ys[a.test])
        if sf is None or st is None:
            continue
        base_t = skill(bases[a.test], ys[a.test])
        rows.append(dict(
            arm=name,
            t_fit=sf['t'], gain_infold_fit=sf['gain'], rho_fit=sf['rho'],
            t_test=st['t'], gain_ceil_test=st['gain'], rho_test=st['rho'],
            d_test=st['d'], s_C_test=st['s_C'],
            # THE number: coefficient learned on `fit`, gain realized on `test`
            gain_transfer=apply_t(bases[a.test], pt, sf['t'], ys[a.test]) - base_t,
            # shrunk half-step, the conservative deployment
            gain_transfer_half=apply_t(bases[a.test], pt, sf['t'] * 0.5, ys[a.test]) - base_t,
        ))
        print(f"{name:22s} t_fit={sf['t']:+7.3f} rho_fit={sf['rho']:+.4f} "
              f"rho_test={st['rho']:+.4f} ceil_test={st['gain']:8.2f} "
              f"TRANSFER={rows[-1]['gain_transfer']:+8.2f}", flush=True)

    df = pd.DataFrame(rows).sort_values('gain_transfer', ascending=False)
    df.to_csv(a.out, index=False)
    print(f'\n-> {a.out}  ({len(df)} arms)')
    print(df[['arm', 't_fit', 'rho_test', 'gain_ceil_test',
              'gain_transfer', 'gain_transfer_half']].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
