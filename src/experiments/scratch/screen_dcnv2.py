"""Deep & Cross Network v2 screen on the legal production 116-feature frame."""
import json, os, sys, time
ROOT=os.path.expanduser("~/LG_data"); sys.path[:0]=[os.path.join(ROOT,"scratch"),ROOT]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config, dl_common as dlc
from agent2_asof_decomp2 import AsofDecomposer2
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
DEVICE=torch.device('cpu'); SEEDS=[7,123,2025]
OUT=os.path.join(ROOT,'scratch','dcnv2_outer_results.json'); PDIR=os.path.join(ROOT,'scratch','dcnv2_outer_preds')
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def frames(df,fold):
 tr=df.iloc[fold.train_idx].copy(); va=df.iloc[fold.val_idx].copy()
 prep=PitchPreprocessor().fit(tr,as_of_season=fold.fold_max_season,is_final=False)
 xt,xv=prep.transform(tr),prep.transform(va); dlc.add_count_x_base(tr,xt); dlc.add_count_x_base(va,xv)
 mp={v:i for i,v in enumerate(xt.count_x_base.unique())}; xt['count_x_base']=xt.count_x_base.map(mp).fillna(-1).astype(int); xv['count_x_base']=xv.count_x_base.map(mp).fillna(-1).astype(int)
 dec=AsofDecomposer2().fit(tr,fold.val_season); at,av=dec.transform(tr),dec.transform(va); at.index,av.index=xt.index,xv.index
 return tr,va,pd.concat([xt,at],axis=1),pd.concat([xv,av],axis=1)
class CrossLayer(nn.Module):
 def __init__(self,d,rank=32):
  super().__init__(); self.u=nn.Linear(d,rank,bias=False); self.v=nn.Linear(rank,d,bias=True)
 def forward(self,x0,x): return x+x0*self.v(self.u(x))
class DCNv2(nn.Module):
 def __init__(self,num_dim,cards):
  super().__init__(); self.emb=dlc.CatEmbedder(cards); d=num_dim+self.emb.out_dim
  self.cross=nn.ModuleList([CrossLayer(d,32) for _ in range(3)])
  self.deep=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.1))
  self.head=nn.Linear(d+64,1)
 def forward(self,xn,xc):
  x0=torch.cat([xn,self.emb(xc)],1); x=x0
  for layer in self.cross: x=layer(x0,x)
  return self.head(torch.cat([x,self.deep(x0)],1)).squeeze(1)
def train(tens,y,seed):
 torch.manual_seed(seed); np.random.seed(seed); m=DCNv2(tens['num_tr'].shape[1],tens['cat_cardinalities'])
 opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-5); lossfn=nn.BCEWithLogitsLoss()
 rng=np.random.RandomState(seed+1); order=rng.permutation(len(y)); dev=order[:int(.05*len(y))]; tri=order[int(.05*len(y)):]; yt=torch.tensor(y,dtype=torch.float32)
 best=1e9; state=None; bad=0
 for ep in range(10):
  m.train(); perm=tri[np.random.permutation(len(tri))]
  for st in range(0,len(perm),8192):
   ix=perm[st:st+8192]; pred=m(tens['num_tr'][ix],tens['cat_tr'][ix]); loss=lossfn(pred,yt[ix]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
  m.eval()
  with torch.no_grad(): dl=lossfn(m(tens['num_tr'][dev],tens['cat_tr'][dev]),yt[dev]).item()
  log(f"seed={seed} ep={ep+1} dev={dl:.6f}")
  if dl<best-1e-5: best=dl; state={k:v.clone() for k,v in m.state_dict().items()}; bad=0
  else:
   bad+=1
   if bad>=2: break
 m.load_state_dict(state); m.eval()
 with torch.no_grad():
  pdv=torch.sigmoid(m(tens['num_tr'][dev],tens['cat_tr'][dev])).numpy(); pva=torch.sigmoid(m(tens['num_val'],tens['cat_val'])).numpy()
 shift=dlc.search_best_shift(y[dev],pdv); return np.clip(pva+shift,1e-6,1-1e-6),float(shift)
def main():
 os.makedirs(PDIR,exist_ok=True); df=pd.read_csv(config.TRAIN_PATH); res={}
 for fold in get_cv_folds(df)[2:]:
  tr,va,xt,xv=frames(df,fold); tens=dlc.to_tensors(xt,xv); preds=[]; shifts=[]
  for s in SEEDS:
   p,sh=train(tens,tr.control_success.to_numpy(np.float32),s); preds.append(p); shifts.append(sh)
  p=np.mean(preds,0); y=va.control_success.to_numpy(); sk,br,_,_=calc_brier_skill_score(y,p)
  res[str(fold.val_season)]={'skill':sk,'brier':br,'shifts':shifts,'seed_skills':[calc_brier_skill_score(y,q)[0] for q in preds]}
  np.savez_compressed(os.path.join(PDIR,f'val{fold.val_season}.npz'),y=y,p=p); json.dump(res,open(OUT,'w'),indent=2); log(f"RESULT {fold.val_season}: {res[str(fold.val_season)]}")
 print(json.dumps(res,indent=2))
if __name__=='__main__': main()
