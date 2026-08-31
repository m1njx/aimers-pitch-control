"""Honest retest of pitcher×count×batter-hand Trackman priors on current CatBoost."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from agent3_tkm_sit import build_situational,attach
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123];OUT=os.path.join(ROOT,'scratch','situational_trackman_catboost_results.json');PDIR=os.path.join(ROOT,'scratch','situational_trackman_catboost_preds')
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def prep(x,y):
 a,b=x.copy(),y.copy();cats=base_cat_cols(a)
 for c in cats:a[c]=a[c].fillna(-1).astype(str);b[c]=b[c].fillna(-1).astype(str)
 for c in a:
  if c not in cats:a[c]=a[c].astype(np.float32);b[c]=b[c].astype(np.float32)
 return a,b,cats
def pred(x,y,t,seed):
 a,b,c=prep(x,y);m=CatBoostClassifier(iterations=250,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=seed,verbose=0,cat_features=c,thread_count=-1,allow_writing_files=False);m.fit(a,t);return m.predict_proba(b)[:,1]-.008
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def main():
 os.makedirs(PDIR,exist_ok=True);df=pd.read_csv(config.TRAIN_PATH);yy=df.control_success.to_numpy();res={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);de=AsofDecomposer2().fit(tr,vs);at,av=de.transform(tr),de.transform(va);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1)
  F=build_situational(vs-1);st=attach(F,tr,xt.copy());sv=attach(F,va,xv.copy());log(f'{vs} sit coverage={sv.sit_n.notna().mean():.4f} features={sv.shape[1]-xv.shape[1]}')
  pb=[];ps=[]
  for seed in SEEDS:log(f'{vs} seed={seed} base');pb.append(pred(xt,xv,yy[mt],seed));log(f'{vs} seed={seed} sit');ps.append(pred(st,sv,yy[mt],seed))
  p0,p1=np.mean(pb,0),np.mean(ps,0);cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));oldmlp=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p'];other=.15*(cache['p_lgb']-.007)+.10*(cache['p_xgb']-.006)
  def full(pc):
   raw=.68*(other+.75*pc)+.32*oldmlp;return np.clip(.5+1.1*(raw-.5)-.0045192086,1e-6,1-1e-6)
  f0,f1=full(p0),full(p1);res[str(vs)]={'cb_base':sk(yy[mv],p0),'cb_sit':sk(yy[mv],p1),'full_base':sk(yy[mv],f0),'full_sit':sk(yy[mv],f1),'full_delta':sk(yy[mv],f1)-sk(yy[mv],f0)}
  np.savez_compressed(os.path.join(PDIR,f'val{vs}.npz'),y=yy[mv],base=p0,sit=p1,full_base=f0,full_sit=f1);json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
