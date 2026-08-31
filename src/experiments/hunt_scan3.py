#!/usr/bin/env python3
"""hunt_scan3.py — third residual scan: partitions that are DERIVED COMBINATIONS
of a row's own columns, which axis-aligned trees cannot easily construct.

Scans 1 and 2 (`residual_scan.py`, `residual_scan2.py`, 36 partitions, max +3.03)
covered raw columns and simple buckets.  They never covered the one derived key
that is physically meaningful and structurally hidden from the model:

  VENUE.  top_bottom tells us which side is batting, so
      home team = pitcher_team_id if top of the inning else batter_team_id
  and the home team's park is the venue.  Park identity plausibly affects a
  command outcome (mound, backdrop, and above all the local umpire crew's zone).
  The model has pitcher_team_id, batter_team_id and top_bottom as three separate
  categoricals; recovering `venue` from them requires a conditional swap that
  axis-aligned splits express only with deep, sample-starved interactions.

Also scanned: pitcher-is-home, venue x count, venue x month, team matchup,
inning x outs (a within-game fatigue proxy that uses only the row's own columns),
and prev1-minus-season-to-date form deviation.

Method identical to residual_scan.py: additive per-cell residual fitted on two
folds with the global level component removed, evaluated on the held-out fold
(LOFO), production-identical 5-seed bagged predictions, zero training.
Rule 4: every key is a function of the row's own columns only.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD, predict, skill, CACHE
FOLDS=[2021,2022,2023]; SEEDS=[7,123,2025,31415,8675309]
def bucket(v,e): return np.digitize(np.nan_to_num(v,nan=0.0),e).astype(str)
def J(*a):
    out=pd.Series(a[0]).astype(str)
    for x in a[1:]: out=out+('|' if not isinstance(x,str) else '')+ (x if isinstance(x,str) else pd.Series(x).astype(str))
    return out.values

def main():
    df=pd.read_csv(os.path.join(LG,'open/data/train.csv'))
    df.columns=[c.replace('﻿','') for c in df.columns]
    print('top_bottom values:',df.top_bottom.value_counts().to_dict())
    D={}
    for y in FOLDS:
        va=df[df.season==y].reset_index(drop=True)
        yv=np.load(os.path.join(CACHE,f'y_{y}.npy')); assert len(yv)==len(va)
        p=np.mean([predict(dict(PROD),dict(np.load(os.path.join(CACHE,f'pred_{y}_{s}.npz'))))
                   for s in SEEDS],0)
        top=va.top_bottom.astype(str).str.upper().str.startswith('T').values
        pt=va.pitcher_team_id.astype(str).values; bt=va.batter_team_id.astype(str).values
        venue=np.where(top,pt,bt)                      # home team's park
        b=va.balls_before.fillna(0).astype(int).astype(str).values
        st=va.strikes_before.fillna(0).astype(int).astype(str).values
        cc=J(b,'_',st)
        mon=va.game_month.fillna(0).astype(int).astype(str).values
        inn=va.inning.fillna(0).astype(int).clip(1,10).astype(str).values
        out=va.outs_before.fillna(0).astype(int).astype(str).values
        s2d=va.asof_pitcher_success_rate.fillna(0.5).values
        dev1=va.asof_pitcher_prev1_game_success_rate.fillna(0.5).values-s2d
        dev5=va.asof_pitcher_prev5_game_success_rate.fillna(0.5).values-s2d
        trend=(va.asof_pitcher_prev1_game_success_rate.fillna(0.5).values
               -va.asof_pitcher_prev5_game_success_rate.fillna(0.5).values)
        D[y]=(yv,p,{
          'venue (홈팀 구장)':venue,
          'pitcher_is_home':top.astype(int).astype(str),
          'venue × count':J(venue,'#',cc),
          'venue × month':J(venue,'#',mon),
          'venue × pitcher_hand':J(venue,'#',va.pitcher_hand.astype(str).values),
          'team matchup (pit×bat)':J(pt,'v',bt),
          'inning × outs':J(inn,'_',out),
          'inning × count':J(inn,'#',cc),
          'prev1 − season 구간':bucket(dev1,[-0.08,-0.03,0.0,0.03,0.08]),
          'prev5 − season 구간':bucket(dev5,[-0.05,-0.02,0.0,0.02,0.05]),
          'prev1 − prev5 구간':bucket(trend,[-0.08,-0.03,0.0,0.03,0.08]),
          'dayofweek × venue':J(va.game_dayofweek.astype(str).values,'#',venue),
        })
    names=list(D[FOLDS[0]][2])
    print(f'\n  {"분할":>26} {"셀수":>6} {"LOFO 평균":>10} {"양수":>6} {"폴드별 델타":>26}')
    res=[]
    for nm in names:
        keys=sorted(set(k for y in FOLDS for k in np.unique(D[y][2][nm])))
        ds=[]
        for held in FOLDS:
            num={k:0.0 for k in keys}; den={k:0.0 for k in keys}
            for y in [f for f in FOLDS if f!=held]:
                yv,p,segs=D[y]; r=yv-p; g=segs[nm]
                idx=pd.Series(r).groupby(g).agg(['sum','size'])
                for k,(s_,n_) in idx.iterrows(): num[k]+=s_; den[k]+=n_
            tw=sum(den.values()); sh={k:(num[k]/den[k] if den[k] else 0.0) for k in keys}
            mu=sum(sh[k]*den[k] for k in keys)/tw if tw else 0.0
            sh={k:sh[k]-mu for k in keys}
            yv,p,segs=D[held]
            add=pd.Series(segs[nm]).map(sh).fillna(0.0).values
            ds.append(skill(np.clip(p+add,1e-6,1-1e-6),yv)-skill(p,yv))
        d=np.array(ds); res.append((nm,len(keys),d))
        print(f'  {nm:>26} {len(keys):>6} {d.mean():+10.2f} {(d>0).sum():>4}/3 '
              f'{" ".join(f"{v:+.2f}" for v in d):>26}')
    best=max(res,key=lambda r:r[2].mean()); ap=[r for r in res if (r[2]>0).all()]
    print(f'\n  최고 {best[0]} {best[2].mean():+.2f}점 | 3폴드 모두 양수: '
          f'{len(ap)}개'+(f' → {", ".join(r[0] for r in ap)}' if ap else ''))
    print('  ※ LB 노이즈 바닥 ±12점')

if __name__=='__main__': main()
