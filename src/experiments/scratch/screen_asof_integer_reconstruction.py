"""Recover integer cumulative counts from organizer asof rates before decomposition."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2,PITCHER_RATES,BATTER_RATES
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123,2025]
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def rounded(raw,dec):
 x=dec.copy()
 for specs,prefix,ncur,nhist in [(PITCHER_RATES,'p','cs_pit_cur_n','cs_pit_hist_n'),(BATTER_RATES,'b','cs_bat_cur_n','cs_bat_hist_n')]:
  for rc,nc,pre in specs:
   hn=x[f'cs_{pre}_hist_n'].to_numpy(float)
   cn=x[f'cs_{pre}_cur_n'].to_numpy(float)
   total=np.rint(raw[nc].fillna(0).to_numpy(float)*raw[rc].fillna(0).to_numpy(float));hist=np.rint(hn*x[f'cs_{pre}_hist'].fillna(0).to_numpy(float));cur=np.clip(total-hist,0,None);rate=np.where(cn>=3,cur/np.maximum(cn,1),np.nan);hr=np.where(hn>0,hist/np.maximum(hn,1),np.nan);fb=np.nanmean(hr);x[f'cs_{pre}_rate']=rate.astype(np.float32);x[f'cs_{pre}_hist']=hr.astype(np.float32);x[f'cs_{pre}_minus_hist']=(rate-hr).astype(np.float32);x[f'cs_{pre}_eb']=((np.nan_to_num(cur*rate,nan=0)+150*np.nan_to_num(hr,nan=fb))/(cn+150)).astype(np.float32)
 x['cs_pb_succ_sum']=x.cs_p_succ_eb+x.cs_b_succ_eb;x['cs_pb_succ_diff']=x.cs_p_succ_eb-x.cs_b_succ_eb;return x
def mod(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();o={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);rt,rv=rounded(tr,at),rounded(va,av);x,v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);u,w=pd.concat([bt,rt],axis=1),pd.concat([bv,rv],axis=1);ci=[x.columns.get_loc(c) for c in base_cat_cols(x)];a=[];b=[]
  for s in SEEDS:a.append(mod(s).fit(x,y[mt],categorical_feature=ci).predict_proba(v)[:,1]-.007);b.append(mod(s).fit(u,y[mt],categorical_feature=ci).predict_proba(w)[:,1]-.007)
  p,q=np.mean(a,0),np.mean(b,0);o[str(vs)]={'base':sk(y[mv],p),'rounded':sk(y[mv],q),'gain':sk(y[mv],q)-sk(y[mv],p)};json.dump(o,open(os.path.join(ROOT,'scratch','asof_integer_reconstruction_results.json'),'w'),indent=2);log(f'RESULT {vs}: {o[str(vs)]}')
 print(json.dumps(o,indent=2))
if __name__=='__main__':main()
