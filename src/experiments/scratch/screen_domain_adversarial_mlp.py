"""Domain-adversarial MLP: learn control signal while suppressing season identity."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,torch,torch.nn as nn
import config,dl_common as dlc
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score
from screen_multitask_aux_mlp import frames
SEEDS=[7];OUT=os.path.join(ROOT,'scratch','domain_adversarial_mlp_results.json')
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
class GR(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,a):ctx.a=a;return x.view_as(x)
 @staticmethod
 def backward(ctx,g):return -ctx.a*g,None
class Net(nn.Module):
 def __init__(self,n,cards,ndom):
  super().__init__();self.emb=dlc.CatEmbedder(cards);self.tr=nn.Sequential(nn.Linear(n+self.emb.out_dim,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15));self.y=nn.Linear(64,1);self.d=nn.Linear(64,ndom)
 def forward(self,xn,xc,a=0):
  h=self.tr(torch.cat([xn,self.emb(xc)],1));return self.y(h).squeeze(1),self.d(GR.apply(h,a))
def fit(t,y,dom,alpha,seed):
 torch.manual_seed(seed);np.random.seed(seed);nd=int(dom.max()+1);m=Net(t['num_tr'].shape[1],t['cat_cardinalities'],nd);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5);bce=nn.BCEWithLogitsLoss();ce=nn.CrossEntropyLoss()
 rng=np.random.RandomState(seed+1);o=rng.permutation(len(y));dev=o[:int(.05*len(y))];train=o[int(.05*len(y)):];yt=torch.tensor(y,dtype=torch.float32);dt=torch.tensor(dom,dtype=torch.long);best=1e9;state=None;bad=0
 for ep in range(8):
  m.train();perm=train[np.random.permutation(len(train))]
  for st in range(0,len(perm),8192):
   ix=perm[st:st+8192];py,pd=m(t['num_tr'][ix],t['cat_tr'][ix],alpha);loss=bce(py,yt[ix])+ce(pd,dt[ix]);opt.zero_grad();loss.backward();opt.step()
  m.eval()
  with torch.no_grad():z,_=m(t['num_tr'][dev],t['cat_tr'][dev]);v=bce(z,yt[dev]).item()
  if v<best-1e-5:best=v;state={k:q.clone() for k,q in m.state_dict().items()};bad=0
  else:
   bad+=1
   if bad>=2:break
 m.load_state_dict(state);m.eval()
 with torch.no_grad():pdv=torch.sigmoid(m(t['num_tr'][dev],t['cat_tr'][dev])[0]).numpy();p=torch.sigmoid(m(t['num_val'],t['cat_val'])[0]).numpy()
 return np.clip(p+dlc.search_best_shift(y[dev],pdv),1e-6,1-1e-6)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);res={}
 for fold in get_cv_folds(df):
  vs=fold.val_season;tr,va,xt,xv=frames(df,fold);t=dlc.to_tensors(xt,xv);y=tr.control_success.to_numpy(np.float32);yv=va.control_success.to_numpy(np.float32);ss=sorted(tr.season.unique());mp={s:i for i,s in enumerate(ss)};dom=tr.season.map(mp).to_numpy()
  pred={}
  for name,a in [('base',0.0),('dann05',.05),('dann15',.15)]:pred[name]=np.mean([fit(t,y,dom,a,s) for s in SEEDS],0);log(f'{vs} {name} standalone={sk(yv,pred[name]):.3f}')
  cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));gbdt=.15*(cache['p_lgb']-.007)+.75*(cache['p_cb']-.008)+.10*(cache['p_xgb']-.006)
  def full(p):
   raw=.68*gbdt+.32*p;return np.clip(.5+1.1*(raw-.5)-.0045192086,1e-6,1-1e-6)
  f0=full(pred['base']);res[str(vs)]={'full_base':sk(yv,f0)}
  for n in ('dann05','dann15'):res[str(vs)]['full_'+n]=sk(yv,full(pred[n]));res[str(vs)]['delta_'+n]=res[str(vs)]['full_'+n]-res[str(vs)]['full_base']
  json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
