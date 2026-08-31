"""Prospective test of train-season-relative features in the production-style MLP.

Reports both the standalone neural component and the complete production blend,
so component improvements cannot be mistaken for a better submission model.
"""
import json, os, sys, time

ROOT = os.path.expanduser("~/LG_data")
sys.path[:0] = [os.path.join(ROOT, "scratch"), ROOT]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import dl_common as dlc
from core.eval_utils import calc_brier_skill_score
from cv_utils import get_cv_folds
from screen_multitask_aux_mlp import frames
from screen_era_relative_features import add_rel

SEEDS = [7, 123, 2025]
OUT = os.path.join(ROOT, "scratch", "era_relative_mlp_results.json")
PRED_DIR = os.path.join(ROOT, "scratch", "era_relative_mlp_preds")

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)

class MLP(nn.Module):
    def __init__(self, nnum, cards):
        super().__init__()
        self.emb = dlc.CatEmbedder(cards)
        self.net = nn.Sequential(nn.Linear(nnum+self.emb.out_dim,128),nn.ReLU(),nn.Dropout(.15),
                                 nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,xn,xc): return self.net(torch.cat([xn,self.emb(xc)],1)).squeeze(1)

def fit_predict(t, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(t['num_tr'].shape[1],t['cat_cardinalities']); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5)
    lossfn=nn.BCEWithLogitsLoss(); rng=np.random.RandomState(seed+1); order=rng.permutation(len(y)); dev=order[:int(.05*len(y))]; train=order[int(.05*len(y)):]
    yt=torch.tensor(y,dtype=torch.float32); best=1e9; state=None; bad=0
    for ep in range(8):
        m.train(); perm=train[np.random.permutation(len(train))]
        for st in range(0,len(perm),8192):
            ix=perm[st:st+8192]; loss=lossfn(m(t['num_tr'][ix],t['cat_tr'][ix]),yt[ix]); opt.zero_grad();loss.backward();opt.step()
        m.eval()
        with torch.no_grad(): dl=lossfn(m(t['num_tr'][dev],t['cat_tr'][dev]),yt[dev]).item()
        if dl<best-1e-5: best=dl;state={k:v.clone() for k,v in m.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=2: break
    m.load_state_dict(state);m.eval()
    with torch.no_grad(): pdv=torch.sigmoid(m(t['num_tr'][dev],t['cat_tr'][dev])).numpy();p=torch.sigmoid(m(t['num_val'],t['cat_val'])).numpy()
    return np.clip(p+dlc.search_best_shift(y[dev],pdv),1e-6,1-1e-6)

def score(y,p): return float(calc_brier_skill_score(y,p)[0])

def main():
    os.makedirs(PRED_DIR,exist_ok=True);df=pd.read_csv(config.TRAIN_PATH);res={}
    for fold in get_cv_folds(df):
        vs=fold.val_season;tr,va,xt,xv=frames(df,fold);rt,rv=add_rel(tr,va,xt,xv)
        ytr=tr[config.TARGET_COL].to_numpy(np.float32);yva=va[config.TARGET_COL].to_numpy(np.float32)
        preds={}
        for name,a,b in [('base',xt,xv),('era',rt,rv)]:
            tens=dlc.to_tensors(a,b); ps=[fit_predict(tens,ytr,s) for s in SEEDS];preds[name]=np.mean(ps,0)
            log(f"{vs} {name} standalone={score(yva,preds[name]):.3f}")
        cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'))
        gbdt=.15*(cache['p_lgb']-.007)+.75*(cache['p_cb']-.008)+.10*(cache['p_xgb']-.006)
        old=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p']
        # Same final formula on both sides; only the neural prediction changes.
        full_old=np.clip((1-.32)*gbdt+.32*old,1e-6,1-1e-6)
        full_new=np.clip((1-.32)*gbdt+.32*preds['era'],1e-6,1-1e-6)
        res[str(vs)]={'base_mlp':score(yva,preds['base']),'era_mlp':score(yva,preds['era']),
                      'full_old':score(yva,full_old),'full_new':score(yva,full_new),
                      'full_delta':score(yva,full_new)-score(yva,full_old)}
        np.savez_compressed(os.path.join(PRED_DIR,f'val{vs}.npz'),y=yva,p_base=preds['base'],p_era=preds['era'],full_old=full_old,full_new=full_new)
        json.dump(res,open(OUT,'w'),indent=2);log(f"{vs} FULL old={res[str(vs)]['full_old']:.3f} new={res[str(vs)]['full_new']:.3f} delta={res[str(vs)]['full_delta']:+.3f}")
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
