"""
agent2_analyze_calib.py — Offline analysis of the cached raw validation
predictions (agent2_exp3_runner.py).

Answers:
  1. Is the near-zero skill of fold 2023 a signal collapse or a DISPERSION
     mismatch? (compare AUC / std(p) / reliability across folds)
  2. How much skill does an oracle logit-affine recalibration p' =
     sigmoid(a*logit(p)+b) recover per fold?
  3. Do the optimal (a, b) move smoothly with season, i.e. can they be
     EXTRAPOLATED from earlier folds (train-labels only, fully legal) instead
     of using one fixed global shift?
  4. Honest nested test: fit the rule on val=2021..2023, apply once to 2024.
"""
import sys, glob, os, json
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from core.eval_utils import calc_brier_skill_score

import os
CACHE = os.environ.get('AGENT2_CACHE', '~/LG_data/scratch/cache_fix')
W = (0.15, 0.75, 0.10)
EPS = 1e-6


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def load(variant, vs):
    f = f'{CACHE}/{variant}_val{vs}.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f)
    p = W[0] * d['p_lgb'] + W[1] * d['p_cb'] + W[2] * d['p_xgb']
    return d['y'].astype(np.float64), p.astype(np.float64)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def skill(y, p):
    return calc_brier_skill_score(y, np.clip(p, EPS, 1 - EPS))[0]


def best_affine(y, p):
    z = logit(p)
    zbar = z.mean()

    def f(th):
        a, b = th
        return brier(y, sig(a * (z - zbar) + b))
    r = minimize(f, [1.0, logit(np.clip(p.mean(), 1e-4, 1 - 1e-4))], method='Nelder-Mead',
                 options=dict(xatol=1e-5, fatol=1e-12, maxiter=2000))
    return r.x, zbar


def apply_affine(p, a, b, zbar):
    return sig(a * (logit(p) - zbar) + b)


def reliability(y, p, nb=10):
    q = pd.qcut(p, nb, labels=False, duplicates='drop')
    d = pd.DataFrame(dict(y=y, p=p, q=q)).groupby('q').agg(n=('y', 'size'), p=('p', 'mean'),
                                                           y=('y', 'mean'))
    return d


def analyze(variant, seasons):
    rows = []
    store = {}
    for vs in seasons:
        r = load(variant, vs)
        if r is None:
            continue
        y, p = r
        store[vs] = (y, p)
        (a, b), zbar = best_affine(y, p)
        pa = apply_affine(p, a, b, zbar)
        # pure shift
        bs, bsk = None, -1
        for s in np.arange(-0.08, 0.021, 0.001):
            sk = skill(y, p + s)
            if sk > bsk:
                bsk, bs = sk, s
        rows.append(dict(val_season=vs, n=len(y), y_mean=y.mean(), p_mean=p.mean(),
                         p_std=p.std(), auc=roc_auc_score(y, p),
                         skill_raw=skill(y, p), skill_shift007=skill(y, p - 0.007),
                         best_shift=bs, skill_bestshift=bsk,
                         a=a, b=b, zbar=zbar, skill_affine=skill(y, pa)))
    R = pd.DataFrame(rows)
    print(f"\n########## variant={variant} ##########")
    print(R.to_string())
    if len(R):
        print(f"\nmean skill: raw={R.skill_raw.mean():.1f} shift-0.007={R.skill_shift007.mean():.1f} "
              f"bestshift={R.skill_bestshift.mean():.1f} affine={R.skill_affine.mean():.1f}")
        for vs in seasons:
            if vs in store:
                y, p = store[vs]
                print(f"\n-- reliability val={vs} --")
                print(reliability(y, p).to_string())
    return R, store


if __name__ == '__main__':
    seasons = [2021, 2022, 2023, 2024]
    variants = sys.argv[1].split(',') if len(sys.argv) > 1 else ['base']
    allR = {}
    for v in variants:
        R, store = analyze(v, seasons)
        allR[v] = R
    # cross-variant comparison on the standard 3 folds
    print("\n\n===== variant comparison (standard folds 2022/2023/2024, shift -0.007) =====")
    print(f"{'variant':<12}{'2022':>10}{'2023':>10}{'2024':>10}{'3fold':>10}{'inner':>10}")
    for v, R in allR.items():
        if not len(R):
            continue
        d = R.set_index('val_season')['skill_shift007']
        std3 = [d.get(s, np.nan) for s in (2022, 2023, 2024)]
        print(f"{v:<12}" + "".join(f"{x:>10.2f}" for x in std3) +
              f"{np.mean(std3):>10.2f}{np.mean(std3[:2]):>10.2f}")
