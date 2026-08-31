"""Low-capacity DeepFM on the legal production 116-feature frame."""
import os,sys,json,time
ROOT=os.path.expanduser("~/LG_data"); sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,torch
import torch.nn as nn
import config,dl_common as dlc
from agent2_asof_decomp2 import AsofDecomposer2
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123,2025]; OUT=os.path.join(ROOT,'scratch','deepfm_results.json'); PDIR=os.path.join(ROOT,'scratch','deepfm_preds')
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def frames(df,f):
 tr=df.iloc[f.train_idx].copy();va=df.iloc[f.val_idx].copy();pr=PitchPreprocessor().fit(tr,as_of_season=f.fold_max_season,is_final=False);xt,xv=pr.transform(tr),pr.transform(va)
 dlc.add_count_x_base(tr,xt);dlc.add_count_x_base(va,xv);mp={v:i for i,v in enumerate(xt.count_x_base.unique())};xt['count_x_base']=xt.count_x_base.map(mp).fillna(-1).astype(int);xv['count_x_base']=xv.count_x_base.map(mp).fillna(-1).astype(int)
 de=AsofDecomposer2().fit(tr,f.val_season);a,b=de.transform(tr),de.transform(va);a.index,b.index=xt.index,xv.index
 return tr,va,pd.concat([xt,a],axis=1),pd.concat([xv,b],axis=1)
class DeepFM(nn.Module):
 def __init__(self,nnum,cards,k=8):
  super().__init__();self.nnum=nnum;self.num_v=nn.Parameter(torch.randn(nnum,k)*.02);self.cat_v=nn.ModuleList([nn.Embedding(c,k) for c in cards]);self.cat_w=nn.ModuleList([nn.Embedding(c,1) for c in cards]);self.num_w=nn.Parameter(torch.zeros(nnum));self.bias=nn.Parameter(torch.zeros(1));nf=nnum+len(cards)
  self.deep=nn.Sequential(nn.Linear(nf*k,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1));self.fm_head=nn.Linear(k,1,bias=False)
 def forward(self,xn,xc):
  fields=[xn.unsqueeze(2)*self.num_v.unsqueeze(0)]+[e(xc[:,i]).unsqueeze(1) for i,e in enumerate(self.cat_v)];v=torch.cat(fields,1);sv=v.sum(1);fm=.5*(sv*sv-(v*v).sum(1));lin=(xn*self.num_w).sum(1)+self.bias
  for i,e in enumerate(self.cat_w):lin=lin+e(xc[:,i]).squeeze(1)
  return lin+self.fm_head(fm).squeeze(1)+self.deep(v.flatten(1)).squeeze(1)
def train(t,y,seed):
 torch.manual_seed(seed);np.random.seed(seed);m=DeepFM(t['num_tr'].shape[1],t['cat_cardinalities']);o=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-5);lf=nn.BCEWithLogitsLoss();ix=np.random.RandomState(seed+1).permutation(len(y));dev=ix[:int(.05*len(y))];tri=ix[int(.05*len(y)):];yt=torch.tensor(y,dtype=torch.float32);best=9;state=None;bad=0
 for ep in range(10):
  m.train();pm=tri[np.random.permutation(len(tri))]
  for st in range(0,len(pm),8192):
   q=pm[st:st+8192];loss=lf(m(t['num_tr'][q],t['cat_tr'][q]),yt[q]);o.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step()
  m.eval()
  with torch.no_grad():dl=lf(m(t['num_tr'][dev],t['cat_tr'][dev]),yt[dev]).item()
  log(f"seed={seed} ep={ep+1} dev={dl:.6f}")
  if dl<best-1e-5:best=dl;state={k:v.clone() for k,v in m.state_dict().items()};bad=0
  else:
   bad+=1
   if bad>=2:break
 m.load_state_dict(state);m.eval()
 with torch.no_grad():pdv=torch.sigmoid(m(t['num_tr'][dev],t['cat_tr'][dev])).numpy();pv=torch.sigmoid(m(t['num_val'],t['cat_val'])).numpy()
 sh=dlc.search_best_shift(y[dev],pdv);return np.clip(pv+sh,1e-6,1-1e-6),float(sh)
def main():
 os.makedirs(PDIR,exist_ok=True);df=pd.read_csv(config.TRAIN_PATH);res={}
 for f in get_cv_folds(df)[:2]:
  tr,va,xt,xv=frames(df,f);t=dlc.to_tensors(xt,xv);ps=[];sh=[]
  for s in SEEDS:q,h=train(t,tr.control_success.to_numpy(np.float32),s);ps.append(q);sh.append(h)
  p=np.mean(ps,0);y=va.control_success.to_numpy();sk,br,_,_=calc_brier_skill_score(y,p);res[str(f.val_season)]={'skill':sk,'brier':br,'shifts':sh};np.savez_compressed(os.path.join(PDIR,f'val{f.val_season}.npz'),y=y,p=p);json.dump(res,open(OUT,'w'),indent=2);log(f"RESULT {f.val_season} {res[str(f.val_season)]}")
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
