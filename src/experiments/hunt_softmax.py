#!/usr/bin/env python3
"""hunt_softmax.py — Hypothesis V: outcome-granularity reformulation.

The recovered per-pitch flags partition every train pitch into FIVE mutually
exclusive and exhaustive outcome classes (counts over 1,472,040 recovered rows):

    0 succ            770,759     (succ=1 always implies rev=0 and mid=0)
    1 reverse only    287,063
    2 middle only     170,000
    3 reverse+middle   50,208
    4 neither         194,010

Today's MLP estimates P(succ | x) directly, so the four negative subclasses are
collapsed into one bucket even though they behave very differently.  Hypothesis:
estimating the 5-class distribution and reading off class 0 is a better-conditioned
estimator of the same quantity, because each subclass has a simpler dependence on x
than their sum does.

CONFOUND CONTROL: the loss family is held fixed.  Baseline is squared error on a
sigmoid; this is squared error on a softmax against the one-hot class (multiclass
Brier).  The ONLY manipulated variable is the granularity of the target.  Trunk,
width, dropout, optimiser, lr, weight decay, epochs, batch size and seed are all
byte-identical to build_cache.SimpleMLP_MSE.

Rows whose class could not be recovered (group boundary, 0.23%) keep a SOFT target:
class 0 one-hot when control_success=1, else the train-frequency distribution over
classes 1-4.  P(succ) is therefore exactly right for those rows too.

RULE 4: labels come from train rows only and are used only as training targets.
Inference reads head 0 from the row's own inputs. Nothing from test is used.

EFFECTIVE WEIGHT: mlp = 0.50, so MLP-alone must reach +24 to clear the LB floor of 12.

PRE-REGISTERED CRITERION (fixed before running): inner folds 2021/2022/2023 x
seeds 7,123,2025,31415,8675309, production-identical bagging, paired 15 cells,
ALL THREE fold means positive AND t > 2.5.
"""
import os, sys, time, argparse, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore')
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
import build_cache as bc, exp_template as ET
BASE=os.path.join(LG,"harness/cache"); FOLDS=[int(x) for x in os.environ.get("SMX_FOLDS","2021,2022,2023").split(",")]
SEEDS=[7,123,2025,31415,8675309]

def fine_class(df):
    g=df.groupby(['pitcher_id','season'],sort=False)
    n=df['asof_pitcher_n'].astype(np.float64)
    nn_=g['asof_pitcher_n'].shift(-1).astype(np.float64)
    lab={}
    for col,nm in [('asof_pitcher_reverse_rate','rev'),('asof_pitcher_middle_rate','mid')]:
        lab[nm]=nn_*g[col].shift(-1).astype(np.float64)-n*df[col].astype(np.float64)
    ok=(~lab['rev'].isna()).values & (~lab['mid'].isna()).values
    rev=np.clip(np.round(np.nan_to_num(lab['rev'].values)),0,1)
    mid=np.clip(np.round(np.nan_to_num(lab['mid'].values)),0,1)
    y=df['control_success'].values.astype(int)
    cls=np.where(y==1,0,np.where((rev>0)&(mid>0),3,np.where(rev>0,1,np.where(mid>0,2,4))))
    T=np.zeros((len(df),5),np.float32)
    T[np.arange(len(df)),cls]=1.0
    # soft target for unrecovered negatives
    bad=(~ok)&(y==0)
    if bad.any():
        good=ok&(y==0)
        fr=np.bincount(cls[good],minlength=5)[1:].astype(np.float32); fr/=fr.sum()
        T[bad]=0.0; T[bad,1:]=fr
    print(f'  fine-class counts {np.bincount(cls[ok],minlength=5).tolist()}  '
          f'unrecovered negatives given soft targets: {int(bad.sum()):,}')
    return T

class SoftmaxMLP(nn.Module):
    def __init__(self,num_dim,cards,K=5,hidden=(128,64),dropout=0.12):
        super().__init__()
        self.cat_embedder=bc.CatEmbedder(cards)
        layers,prev=[],num_dim+self.cat_embedder.out_dim
        for h in hidden:
            layers+=[nn.Linear(prev,h),nn.ReLU(),nn.Dropout(dropout)]; prev=h
        layers+=[nn.Linear(prev,K),nn.Softmax(dim=1)]
        self.net=nn.Sequential(*layers)
    def forward(self,xn,xc): return self.net(torch.cat([xn,self.cat_embedder(xc)],1))

def train_one(seed,nz,ca,T,nzv,cav,cards):
    torch.manual_seed(seed)
    net=SoftmaxMLP(nz.shape[1],cards)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(nz),torch.tensor(ca),torch.tensor(T)),batch_size=2048,shuffle=True)
    net.train()
    for _ in range(5):
        for bn,bcat,bt in dl:
            opt.zero_grad(); ((net(bn,bcat)-bt)**2).mean().backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(nzv),torch.tensor(cav))[:,0].numpy().astype(np.float64)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='smx')
    a=ap.parse_args(); print(__doc__,flush=True)
    t0=time.time(); torch.set_num_threads(2)
    df=pd.read_csv(os.path.join(LG,'open/data/train.csv'))
    df.columns=[c.replace('﻿','') for c in df.columns]
    T=fine_class(df)
    cdir=os.path.join(LG,f'harness/cache_{a.tag}'); os.makedirs(cdir,exist_ok=True)
    from evaluate import skill
    for y in FOLDS:
        need=[s for s in SEEDS if not os.path.exists(os.path.join(cdir,f'pred_{y}_{s}.npz'))]
        if not need: print(f'=== {y}: cached, skip ==='); continue
        past,va,prep,dec,cat_map=ET.fold_data(df,y)
        _,Xp=bc.build_features(past,prep,dec,cat_map)
        _,Xv=bc.build_features(va,prep,dec,cat_map)
        nz,ca,art=bc.mlp_arrays(Xp); nzv,cav,_=bc.mlp_arrays(Xv,art); del Xp,Xv
        Tp=T[past.index.values]
        yv=np.load(os.path.join(BASE,f'y_{y}.npy'))
        print(f'\n=== eval {y}: past {len(past):,} ({time.time()-t0:.0f}s) ===',flush=True)
        for s in need:
            t1=time.time(); src=dict(np.load(os.path.join(BASE,f'pred_{y}_{s}.npz')))
            out=dict(src); out['mlp']=train_one(s,nz,ca,Tp,nzv,cav,art['cards'])
            np.savez_compressed(os.path.join(cdir,f'pred_{y}_{s}.npz'),**out)
            print(f'  seed {s}: ({time.time()-t1:.0f}s) MLP-alone '
                  f'{skill(src["mlp"],yv):.1f} -> {skill(out["mlp"],yv):.1f}',flush=True)
        del nz,ca,nzv,cav,past,va
    ET.score(cdir); print(f'\n총 {(time.time()-t0)/60:.1f}분')

if __name__=='__main__': main()
