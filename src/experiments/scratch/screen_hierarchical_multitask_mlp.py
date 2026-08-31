"""Direct-success + five-outcome hierarchical multitask MLP, inner only."""
import json, sys, time
sys.path[:0]=["~/LG_data/scratch",os.path.expanduser("~/LG_data")]
import numpy as np, pandas as pd, torch, torch.nn as nn
import config, dl_common as dlc
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score
from agent2_recover_labels import recover
from screen_multitask_aux_mlp import frames, AUX_COLS

SEEDS=[7,123]; DEVICE=torch.device('cpu')
OUT='~/LG_data/scratch/hierarchical_multitask_mlp_results.json'
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)

class Net(nn.Module):
 def __init__(self,n,cards):
  super().__init__(); self.emb=dlc.CatEmbedder(cards)
  self.trunk=nn.Sequential(nn.Linear(n+self.emb.out_dim,128),nn.ReLU(),nn.Dropout(.15),
                           nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15))
  self.main=nn.Linear(64,1); self.outcome=nn.Linear(64,5); self.aux=nn.Linear(64,len(AUX_COLS))
 def forward(self,xn,xc):
  h=self.trunk(torch.cat([xn,self.emb(xc)],1))
  return self.main(h).squeeze(1),self.outcome(h),self.aux(h)

def labels5(y,r,m):
 o=np.full(len(y),-1,np.int64); k=np.isfinite(r)&np.isfinite(m)
 o[k&(y==1)]=0;o[k&(y==0)&(r==1)&(m==0)]=1;o[k&(y==0)&(r==0)&(m==1)]=2
 o[k&(y==0)&(r==1)&(m==1)]=3;o[k&(y==0)&(r==0)&(m==0)]=4
 return o

def fit(t,y,a,o,seed):
 torch.manual_seed(seed);np.random.seed(seed); net=Net(t['num_tr'].shape[1],t['cat_cardinalities'])
 opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5); bce=nn.BCEWithLogitsLoss()
 yt=torch.tensor(y,dtype=torch.float32); at=torch.tensor(np.nan_to_num(a,nan=0),dtype=torch.float32)
 am=torch.tensor(np.isfinite(a),dtype=torch.float32); ot=torch.tensor(o,dtype=torch.long)
 rng=np.random.RandomState(seed+1); z=rng.permutation(len(y)); dev=z[:int(.05*len(y))];tr=z[int(.05*len(y)):]
 best=1e9;state=None;bad=0
 for ep in range(8):
  net.train(); ixall=tr[np.random.permutation(len(tr))]
  for st in range(0,len(tr),8192):
   ix=ixall[st:st+8192]; lm,lo,la=net(t['num_tr'][ix],t['cat_tr'][ix])
   raw=nn.functional.binary_cross_entropy_with_logits(la,at[ix],reduction='none')
   loss=bce(lm,yt[ix])+nn.functional.cross_entropy(lo,ot[ix],ignore_index=-1)+ (raw*am[ix]).sum()/am[ix].sum()
   opt.zero_grad();loss.backward();opt.step()
  net.eval()
  with torch.no_grad(): lm,lo,_=net(t['num_tr'][dev],t['cat_tr'][dev]); dl=bce(lm,yt[dev]).item()
  log(f"seed={seed} ep={ep+1} dev={dl:.6f}")
  if dl<best-1e-5:best=dl;state={k:v.clone() for k,v in net.state_dict().items()};bad=0
  else:
   bad+=1
   if bad>=2:break
 net.load_state_dict(state);net.eval()
 with torch.no_grad():
  dm,do,_=net(t['num_tr'][dev],t['cat_tr'][dev]); vm,vo,_=net(t['num_val'],t['cat_val'])
  pd=torch.sigmoid(dm).numpy(); pm=torch.sigmoid(vm).numpy(); po=torch.softmax(vo,1)[:,0].numpy()
 sh=dlc.search_best_shift(y[dev],pd)
 return np.clip(pm+sh,1e-6,1-1e-6),po

def main():
 df=pd.read_csv(config.TRAIN_PATH);L=recover(df);res={}
 for fold in [f for f in get_cv_folds(df) if f.val_season in (2022,2023)]:
  tr,va,xt,xv=frames(df,fold);t=dlc.to_tensors(xt,xv);idx=fold.train_idx
  y=tr.control_success.to_numpy(np.float32);yv=va.control_success.to_numpy(np.float32)
  a=L.iloc[idx][AUX_COLS].to_numpy(np.float32);o=labels5(y,a[:,0],a[:,1])
  P=[fit(t,y,a,o,s) for s in SEEDS];p_direct=np.mean([x[0] for x in P],0);p_outcome=np.mean([x[1] for x in P],0)
  grid={}
  for w in np.linspace(0,1,21):
   sk,br,_,_=calc_brier_skill_score(yv,(1-w)*p_direct+w*p_outcome);grid[str(round(float(w),2))]={'skill':sk,'brier':br}
  best=max(grid.items(),key=lambda x:x[1]['skill']);res[str(fold.val_season)]={'best':best,'grid':grid}
  json.dump(res,open(OUT,'w'),indent=2);log(f"RESULT {fold.val_season} {best}")
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
