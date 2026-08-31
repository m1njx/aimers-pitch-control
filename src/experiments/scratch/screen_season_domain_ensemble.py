"""Season-domain CatBoost ensemble, evaluated only as a complete v23-style model."""
import json, os, sys, time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score

SEEDS=[7,123]; OUT=os.path.join(ROOT,'scratch','season_domain_ensemble_results.json')
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def prepared(a,b):
 a,b=a.copy(),b.copy();cats=base_cat_cols(a)
 for c in cats:a[c]=a[c].fillna(-1).astype(str);b[c]=b[c].fillna(-1).astype(str)
 for c in a:
  if c not in cats:a[c]=a[c].astype(np.float32);b[c]=b[c].astype(np.float32)
 return a,b,cats
def fit(a,y,b,cats,seed):
 m=CatBoostClassifier(iterations=250,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=seed,
  verbose=0,cat_features=cats,thread_count=-1,allow_writing_files=False)
 m.fit(a,y);return m.predict_proba(b)[:,1]-.008
def skill(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);res={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();y=tr.control_success.to_numpy();yv=va.control_success.to_numpy()
  bt,bv=build_base_features(tr,va,vs-1,fix_index=True);dec=AsofDecomposer2().fit(tr,vs);at,av=dec.transform(tr),dec.transform(va)
  xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);a,b,cats=prepared(xt,xv)
  pooled=[];equal=[];recent=[]
  seasons=sorted(tr.season.unique())
  # Fixed prospective weights: equal domains and linearly recent domains.
  rw=np.arange(1,len(seasons)+1,dtype=float);rw/=rw.sum()
  for seed in SEEDS:
   log(f'{vs} seed={seed} pooled');pooled.append(fit(a,y,b,cats,seed))
   domain=[]
   for s in seasons:
    ix=tr.season.to_numpy()==s;log(f'{vs} seed={seed} domain={s} n={ix.sum()}');domain.append(fit(a.loc[ix],y[ix],b,cats,seed))
   equal.append(np.mean(domain,axis=0));recent.append(np.average(domain,axis=0,weights=rw))
  pp=np.mean(pooled,axis=0);pe=np.mean(equal,axis=0);pr=np.mean(recent,axis=0)
  cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'))
  oldmlp=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p']
  other=.15*(cache['p_lgb']-.007)+.10*(cache['p_xgb']-.006)
  def full(pc):
   raw=.68*(other+.75*pc)+.32*oldmlp
   return np.clip(.5+1.1*(raw-.5)-.0045192086,1e-6,1-1e-6)
  f0,fe,fr=full(pp),full(pe),full(pr)
  res[str(vs)]={'cb_pooled':skill(yv,pp),'cb_equal_domain':skill(yv,pe),'cb_recent_domain':skill(yv,pr),
   'full_pooled':skill(yv,f0),'full_equal_domain':skill(yv,fe),'full_recent_domain':skill(yv,fr),
   'equal_delta':skill(yv,fe)-skill(yv,f0),'recent_delta':skill(yv,fr)-skill(yv,f0)}
  json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
