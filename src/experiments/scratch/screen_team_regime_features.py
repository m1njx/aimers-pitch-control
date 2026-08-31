"""Train-only next-season team-relative target priors."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123,2025]
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def add(tr,va,x,v):
 g=tr.groupby('season').control_success.mean();last=int(tr.season.max())
 for col,tag in [('pitcher_team_id','pt'),('batter_team_id','bt')]:
  z=tr.groupby(['season',col]).control_success.agg(['mean','size']).reset_index();z['rel']=z['mean']-z.season.map(g);p=z.pivot(index=col,columns='season',values='rel');n=z.pivot(index=col,columns='season',values='size');ss=sorted(tr.season.unique());trend=p[ss].diff(axis=1).mean(axis=1).fillna(0);lastrel=p[last].fillna(0);sh=n[last].fillna(0)/(n[last].fillna(0)+500);nextrel=sh*(lastrel+.5*trend)+(1-sh)*0
  for d,q in [(tr,x),(va,v)]:q[f'team_{tag}_lastrel']=d[col].map(lastrel).fillna(0).astype(np.float32);q[f'team_{tag}_nextrel']=d[col].map(nextrel).fillna(0).astype(np.float32)
 return x,v
def mod(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();o={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);x,v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);u,w=add(tr,va,x.copy(),v.copy());ci=[x.columns.get_loc(c) for c in base_cat_cols(x)];a=[];b=[]
  for s in SEEDS:a.append(mod(s).fit(x,y[mt],categorical_feature=ci).predict_proba(v)[:,1]-.007);b.append(mod(s).fit(u,y[mt],categorical_feature=ci).predict_proba(w)[:,1]-.007)
  p,q=np.mean(a,0),np.mean(b,0);o[str(vs)]={'base':sk(y[mv],p),'team':sk(y[mv],q),'gain':sk(y[mv],q)-sk(y[mv],p)};json.dump(o,open(os.path.join(ROOT,'scratch','team_regime_features_results.json'),'w'),indent=2);log(f'RESULT {vs}: {o[str(vs)]}')
 print(json.dumps(o,indent=2))
if __name__=='__main__':main()
