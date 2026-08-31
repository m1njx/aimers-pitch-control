"""Soft MoE using only pre-pitch expected pitch-type probabilities as gates."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from agent3_tkm_sit import build_situational,attach
from core.eval_utils import calc_brier_skill_score
SEED=7;OUT=os.path.join(ROOT,'scratch','pitchtype_soft_moe_results.json')
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def prep(x,y):
 a,b=x.copy(),y.copy();cats=base_cat_cols(a)
 for c in cats:a[c]=a[c].fillna(-1).astype(str);b[c]=b[c].fillna(-1).astype(str)
 for c in a:
  if c not in cats:a[c]=a[c].astype(np.float32);b[c]=b[c].astype(np.float32)
 return a,b,cats
def fit(a,y,b,cats,w=None):
 m=CatBoostClassifier(iterations=200,depth=6,learning_rate=.06,l2_leaf_reg=12,random_seed=SEED,verbose=0,cat_features=cats,thread_count=-1,allow_writing_files=False);m.fit(a,y,sample_weight=w);return m.predict_proba(b)[:,1]-.008
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def gates(x, fallback):
 q=x[['sit_is_fb','sit_is_br','sit_is_os']].to_numpy(float);bad=~np.isfinite(q).all(1);q[bad]=fallback;q=np.clip(q,.01,None);return q/q.sum(1,keepdims=True)
def main():
 df=pd.read_csv(config.TRAIN_PATH);yy=df.control_success.to_numpy();res={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);de=AsofDecomposer2().fit(tr,vs);at,av=de.transform(tr),de.transform(va);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);F=build_situational(vs-1);st=attach(F,tr,xt.copy());sv=attach(F,va,xv.copy());a,b,cats=prep(st,sv)
  fb=F[['sit_is_fb','sit_is_br','sit_is_os']].mean().to_numpy();qt,qv=gates(st,fb),gates(sv,fb)
  log(f'{vs} pooled');p0=fit(a,yy[mt],b,cats)
  experts=[]
  for j,n in enumerate(('fb','br','os')):
   # A small floor prevents an expert from discarding most training rows.
   w=.10+.90*qt[:,j];log(f'{vs} expert={n} effective_weight={w.mean():.3f}');experts.append(fit(a,yy[mt],b,cats,w))
  pm=np.sum(qv*np.stack(experts,axis=1),axis=1)
  cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));oldmlp=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p'];other=.15*(cache['p_lgb']-.007)+.10*(cache['p_xgb']-.006)
  def full(pc):
   raw=.68*(other+.75*pc)+.32*oldmlp;return np.clip(.5+1.1*(raw-.5)-.0045192086,1e-6,1-1e-6)
  f0,fm=full(p0),full(pm);res[str(vs)]={'cb_pooled_sit':sk(yy[mv],p0),'cb_soft_moe':sk(yy[mv],pm),'full_pooled_sit':sk(yy[mv],f0),'full_soft_moe':sk(yy[mv],fm),'full_delta':sk(yy[mv],fm)-sk(yy[mv],f0)};json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
