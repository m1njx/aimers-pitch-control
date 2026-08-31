"""hunt_budget.py — how much headroom is left in ESTIMATION (as opposed to information)?

skill = 1e5*[2Cov(p,y) - Var(p) - (mean p - r)^2]/(r(1-r)).  Three separable terms:
  (a) level     : the (mean p - r)^2 penalty        -> removable by a global shift
  (b) slope     : Var(p) != Cov(p,y)                -> removable by an affine rescale
  (c) variance  : finite-seed estimation noise      -> removable by more seeds
Anything that merely regularises or averages better is capped by (a)+(b)+(c).
Whatever is left over is INFORMATION, which needs a new channel, not a better fit.
"""
import os, sys, numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD, predict, skill
CACHE=os.path.join(LG,'harness/cache'); SEEDS=[7,123,2025,31415,8675309]
print(f'{"fold":>6} {"5-seed bag":>11} {"1-seed avg":>11} {"inf-seed":>10} {"(c) seeds":>10} '
      f'{"(a) level":>10} {"(b) slope":>10} {"a+b+c":>8}')
tot=[]
for y in [2021,2022,2023,2024]:
    yv=np.load(os.path.join(CACHE,f'y_{y}.npy')); r=yv.mean(); V=r*(1-r)
    ps=[predict(dict(PROD),dict(np.load(os.path.join(CACHE,f'pred_{y}_{s}.npz')))) for s in SEEDS]
    P=np.stack(ps); bag=P.mean(0)
    k=len(SEEDS)
    s1=float(np.mean([skill(p,yv) for p in ps])); sk=skill(bag,yv)
    # Brier(bag_k) = Brier(inf) + (1/k)*E[Var_seed(p)] ; estimate seed variance unbiasedly
    vs=float(P.var(0,ddof=1).mean())
    sinf=sk - 1e5*(vs/k)/V*0 + 1e5*(vs/k)/V   # removing the residual 1/k noise term
    lvl=1e5*(bag.mean()-r)**2/V
    c=np.cov(bag,yv)[0,1]; vp=bag.var()
    slope=1e5*((c-vp)**2/max(vp,1e-12))/V     # gain from the oracle affine rescale
    print(f'{y:>6} {sk:11.1f} {s1:11.1f} {sinf:10.1f} {sinf-sk:10.1f} {lvl:10.1f} {slope:10.1f} '
          f'{sinf-sk+lvl+slope:8.1f}')
    tot.append((sinf-sk, lvl, slope))
t=np.array(tot).mean(0)
print(f'\n  4-fold mean removable headroom: seeds {t[0]:+.1f}, level {t[1]:+.1f}, '
      f'slope {t[2]:+.1f}   TOTAL {t.sum():+.1f} skill points')
print('  (LB noise floor is 12; the gap to close is 118)')
