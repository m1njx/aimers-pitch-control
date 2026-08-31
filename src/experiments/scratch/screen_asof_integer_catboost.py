"""CatBoost/full-blend confirmation of integer-reconstructed asof decomposition."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from screen_asof_integer_reconstruction import rounded
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123];OUT=os.path.join(ROOT,'scratch','asof_integer_catboost_results.json')
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def prep(x,y):
 a,b=x.copy(),y.copy();cats=base_cat_cols(a)
 for c in cats:a[c]=a[c].fillna(-1).astype(str);b[c]=b[c].fillna(-1).astype(str)
 for c in a:
  if c not in cats:a[c]=a[c].astype(np.float32);b[c]=b[c].astype(np.float32)
 return a,b,cats
def fit(x,y,v,s):
 a,b,c=prep(x,v);m=CatBoostClassifier(iterations=250,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=s,verbose=0,cat_features=c,thread_count=-1,allow_writing_files=False);m.fit(a,y);return m.predict_proba(b)[:,1]-.008
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();o={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);rt,rv=rounded(tr,at),rounded(va,av);x,v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);u,w=pd.concat([bt,rt],axis=1),pd.concat([bv,rv],axis=1);a=[];b=[]
  for s in SEEDS:log(f'{vs} seed={s} base');a.append(fit(x,y[mt],v,s));log(f'{vs} seed={s} rounded');b.append(fit(u,y[mt],w,s))
  p,q=np.mean(a,0),np.mean(b,0);c=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));mlp=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p'];other=.15*(c['p_lgb']-.007)+.10*(c['p_xgb']-.006)
  def full(cb):return np.clip(.5+1.1*((.68*(other+.75*cb)+.32*mlp)-.5)-.0045192086,1e-6,1-1e-6)
  f,g=full(p),full(q);o[str(vs)]={'cb_base':sk(y[mv],p),'cb_rounded':sk(y[mv],q),'full_base':sk(y[mv],f),'full_rounded':sk(y[mv],g),'delta':sk(y[mv],g)-sk(y[mv],f)};json.dump(o,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {o[str(vs)]}')
 print(json.dumps(o,indent=2))
if __name__=='__main__':main()
