"""hunt_tails.py — how much of our Brier is spent on a handful of extreme rows?

Motivation: v56 (md5-identical artifacts to v50, only blend weights differ) scored
915 vs v50's 1032.  A Brier quadratic in blend weights cannot produce -117:
moving 0.1 of weight changes Var(p) by ~1e-5 -> ~4 skill points.  The only
mechanism that CAN is a heavy tail: with N=253k and r(1-r)=.248, ONE row at
p=0, y=1 costs 1e5*1/(.248*253507) = 1.59 skill points.  ~74 such rows = -117.
lgb_mse is an unconstrained LightGBM *regression*, so it can emit values far
outside [0,1]; raising its weight to .35 amplifies that tail.
"""
import os, sys, numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD, predict
CACHE=os.path.join(LG,'harness/cache'); SEEDS=[7,123,2025,31415,8675309]
e=1e-6
CFG={'v42':dict(w_gbdt=.40,w_mlp=.40,w_mse=.20,scale=1.10,shift=-0.0045192086),
     'v50':dict(w_gbdt=.25,w_mlp=.50,w_mse=.25,scale=1.10,shift=-0.00350),
     'v56':dict(w_gbdt=.14,w_mlp=.51,w_mse=.35,scale=1.10,shift=-0.00350)}
def skill(p,y):
    r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
for yr in [2021,2022,2023,2024]:
    y=np.load(os.path.join(CACHE,f'y_{yr}.npy')); N=len(y); r=y.mean()
    Ps=[dict(np.load(os.path.join(CACHE,f'pred_{yr}_{s}.npz'))) for s in SEEDS]
    print(f'\n===== {yr}  N={N:,}  r={r:.4f}  1 fully-wrong row = {1e5/(r*(1-r)*N):.2f} skill pts =====')
    lm=np.mean([P['lgb_mse'] for P in Ps],0)
    print(f'  lgb_mse raw: min {lm.min():.4f} max {lm.max():.4f}  <0: {(lm<0).sum()}  >1: {(lm>1).sum()}'
          f'  <0.05: {(lm<0.05).sum()}  >0.95: {(lm>0.95).sum()}')
    mlp=np.mean([P['mlp'] for P in Ps],0)
    print(f'  mlp      : min {mlp.min():.4f} max {mlp.max():.4f}')
    for name,cfg in CFG.items():
        p=np.mean([predict({**PROD,**cfg},P) for P in Ps],0)
        L=(p-y)**2
        o=np.argsort(-L)
        tot=L.mean()
        fr=[(k,L[o[:k]].sum()/(N*tot)) for k in (10,100,1000)]
        # skill if we clip p into [lo,1-lo]
        row=f'  {name}: skill {skill(p,y):8.1f}  p range [{p.min():.4f},{p.max():.4f}]'
        row+=f'  top10 rows={fr[0][1]*100:.3f}% of Brier, top1000={fr[2][1]*100:.2f}%'
        print(row)
        s=[]
        for lo in [0.0,0.02,0.05,0.10,0.15,0.20,0.25]:
            pc=np.clip(p,lo,1-lo); s.append((lo,skill(pc,y)-skill(p,y)))
        print('       clip gain: '+'  '.join(f'{lo:.2f}:{d:+.1f}' for lo,d in s))
