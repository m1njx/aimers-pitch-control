"""Stable player-matchup embedding extension of the production-style MLP."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,torch,torch.nn as nn
import config,dl_common as dlc
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score
from screen_multitask_aux_mlp import frames
SEEDS=[7];OUT=os.path.join(ROOT,'scratch','matchup_embedding_mlp_results.json')
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def ids(tr,va,col):
 mp={v:i+1 for i,v in enumerate(pd.unique(tr[col]))};return np.array([mp.get(v,0) for v in tr[col]],np.int64),np.array([mp.get(v,0) for v in va[col]],np.int64),len(mp)+1
class Net(nn.Module):
 def __init__(self,n,cards,npit,nbat,use_id):
  super().__init__();self.use_id=use_id;self.e=dlc.CatEmbedder(cards);self.pe=nn.Embedding(npit,12);self.be=nn.Embedding(nbat,12);self.pbias=nn.Embedding(npit,1);self.bbias=nn.Embedding(nbat,1);extra=27 if use_id else 0;self.net=nn.Sequential(nn.Linear(n+self.e.out_dim+extra,128),nn.ReLU(),nn.Dropout(.18),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.18),nn.Linear(64,1))
 def forward(self,xn,xc,p,b):
  z=[xn,self.e(xc)]
  if self.use_id:z += [self.pe(p),self.be(b),(self.pe(p)*self.be(b)).sum(1,keepdim=True),self.pbias(p),self.bbias(b)]
  return self.net(torch.cat(z,1)).squeeze(1)
def fit(t,y,pit,bat,npit,nbat,use,seed):
 torch.manual_seed(seed);np.random.seed(seed);m=Net(t['num_tr'].shape[1],t['cat_cardinalities'],npit,nbat,use);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=2e-4);loss=nn.BCEWithLogitsLoss();rng=np.random.RandomState(seed+1);o=rng.permutation(len(y));dev=o[:int(.05*len(y))];train=o[int(.05*len(y)):];yt=torch.tensor(y,dtype=torch.float32);pt=torch.tensor(pit[0]);pv=torch.tensor(pit[1]);bt=torch.tensor(bat[0]);bv=torch.tensor(bat[1]);best=1e9;state=None;bad=0
 for ep in range(10):
  m.train();perm=train[np.random.permutation(len(train))]
  for st in range(0,len(perm),8192):
   ix=perm[st:st+8192];v=loss(m(t['num_tr'][ix],t['cat_tr'][ix],pt[ix],bt[ix]),yt[ix]);opt.zero_grad();v.backward();opt.step()
  m.eval()
  with torch.no_grad():v=loss(m(t['num_tr'][dev],t['cat_tr'][dev],pt[dev],bt[dev]),yt[dev]).item()
  if v<best-1e-5:best=v;state={k:q.clone() for k,q in m.state_dict().items()};bad=0
  else:
   bad+=1
   if bad>=2:break
 m.load_state_dict(state);m.eval()
 with torch.no_grad():pdv=torch.sigmoid(m(t['num_tr'][dev],t['cat_tr'][dev],pt[dev],bt[dev])).numpy();p=torch.sigmoid(m(t['num_val'],t['cat_val'],pv,bv)).numpy()
 return np.clip(p+dlc.search_best_shift(y[dev],pdv),1e-6,1-1e-6)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);res={}
 for f in get_cv_folds(df)[:2]:
  vs=f.val_season;tr,va,xt,xv=frames(df,f);t=dlc.to_tensors(xt,xv);y=tr.control_success.to_numpy(np.float32);yv=va.control_success.to_numpy(np.float32);pi=ids(tr,va,'pitcher_id');ba=ids(tr,va,'batter_id');pred={}
  for n,u in [('base',False),('matchup',True)]:pred[n]=np.mean([fit(t,y,(pi[0],pi[1]),(ba[0],ba[1]),pi[2],ba[2],u,s) for s in SEEDS],0);log(f'{vs} {n} standalone={sk(yv,pred[n]):.3f}')
  c=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));g=.15*(c['p_lgb']-.007)+.75*(c['p_cb']-.008)+.10*(c['p_xgb']-.006)
  def full(p):return np.clip(.5+1.1*((.68*g+.32*p)-.5)-.0045192086,1e-6,1-1e-6)
  res[str(vs)]={'base':sk(yv,pred['base']),'matchup':sk(yv,pred['matchup']),'full_base':sk(yv,full(pred['base'])),'full_matchup':sk(yv,full(pred['matchup'])),'delta':sk(yv,full(pred['matchup']))-sk(yv,full(pred['base']))};json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
