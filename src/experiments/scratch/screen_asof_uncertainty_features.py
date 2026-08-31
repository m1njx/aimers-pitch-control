"""Explicit posterior uncertainty for legal organizer-provided asof rates."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123,2025]
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def add_u(x):
 z=x.copy();pairs=[('p_succ','cs_pit_cur_n'),('p_rev','cs_pit_cur_n'),('p_mid','cs_pit_cur_n'),('p_ball','cs_pit_cur_n'),('p_str','cs_pit_cur_n'),('p_fb','cs_pit_cur_n'),('p_br','cs_pit_cur_n'),('p_os','cs_pit_cur_n'),('b_succ','cs_bat_cur_n'),('b_mid','cs_bat_cur_n')]
 for p,nc in pairs:
  r=z[f'cs_{p}_rate'].to_numpy(float);h=z[f'cs_{p}_hist'].to_numpy(float);n=np.maximum(z[nc].to_numpy(float),0);v=np.clip(np.nan_to_num(h,nan=.5),.02,.98);se=np.sqrt(v*(1-v)/np.maximum(n,1));d=r-h
  z[f'u_{p}_z']=(d/np.maximum(se,.01)).astype(np.float32);z[f'u_{p}_se']=se.astype(np.float32);z[f'u_{p}_rel']=(np.abs(d)/np.maximum(se,.01)).astype(np.float32)
 return z
def mod(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();o={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);x,v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);u,w=add_u(x),add_u(v);ci=[x.columns.get_loc(c) for c in base_cat_cols(x)];a=[];b=[]
  for s in SEEDS:a.append(mod(s).fit(x,y[mt],categorical_feature=ci).predict_proba(v)[:,1]-.007);b.append(mod(s).fit(u,y[mt],categorical_feature=ci).predict_proba(w)[:,1]-.007)
  p,q=np.mean(a,0),np.mean(b,0);o[str(vs)]={'base':sk(y[mv],p),'uncertainty':sk(y[mv],q),'gain':sk(y[mv],q)-sk(y[mv],p)};json.dump(o,open(os.path.join(ROOT,'scratch','asof_uncertainty_features_results.json'),'w'),indent=2);log(f'RESULT {vs}: {o[str(vs)]}')
 print(json.dumps(o,indent=2))
if __name__=='__main__':main()
