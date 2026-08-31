"""Actual CatBoost Bayesian-bootstrap diversity screen (not ignored MVS temp)."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','catboost_bayesian_bootstrap_results.json')
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def prep(x,y):
 a,b=x.copy(),y.copy();c=base_cat_cols(a)
 for z in (a,b):
  for k in c:z[k]=z[k].fillna(-1).astype(str)
  for k in z:
   if k not in c:z[k]=z[k].astype(np.float32)
 return a,b,c
def fit(x,y,v,temp):
 a,b,c=prep(x,v);kw=dict(iterations=250,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=7,verbose=0,cat_features=c,thread_count=6,allow_writing_files=False)
 if temp is not None:kw.update(bootstrap_type='Bayesian',bagging_temperature=temp)
 m=CatBoostClassifier(**kw);m.fit(a,y);return m.predict_proba(b)[:,1]-.008
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();o={}
 for vs in (2022,2023):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);x,v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);p0=fit(x,y[mt],v,None);o[str(vs)]={'base':sk(y[mv],p0),'temps':{}}
  for t in (.25,1.,3.):
   q=fit(x,y[mt],v,t);grid={}
   for w in (.1,.2,.3,.4,.5):grid[str(w)]=sk(y[mv],(1-w)*p0+w*q)-sk(y[mv],p0)
   o[str(vs)]['temps'][str(t)]={'alone':sk(y[mv],q),'best_gain':max(grid.values()),'grid':grid}
  json.dump(o,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {o[str(vs)]}')
 print(json.dumps(o,indent=2))
if __name__=='__main__':main()
