"""Strict temporal ExtraTrees diversity screen against exact v33 OOF arrays."""
import gc,json,os,sys
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
ROOT=os.path.expanduser('~/LG_data');sys.path.insert(0,os.path.join(ROOT,'scratch'))
from audit_v16_exact_cv import add_features
CATS=['top_bottom','base_state','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id','count_code','platoon_matchup','tkm_match','count_x_base']
def skill(y,p):
 b=float(np.mean((y-p)**2));return 100000*(1-b/(y.mean()*(1-y.mean()))),b
def main():
 raw=pd.read_csv(os.path.join(ROOT,'open/data/train.csv'));out={}
 for s in [2022,2023]:
  tr=raw[raw.season<s].copy();va=raw[raw.season==s].copy();xt,xv=add_features(tr,va,s)
  # Trees need a stable numerical representation only; categories are already train-only integer codes.
  xt=xt.apply(pd.to_numeric,errors='coerce').fillna(-1).astype(np.float32);xv=xv.apply(pd.to_numeric,errors='coerce').fillna(-1).astype(np.float32)
  pp=np.zeros(len(va))
  for seed in [7,2025]:
   m=ExtraTreesClassifier(n_estimators=350,max_features=.70,min_samples_leaf=30,max_depth=None,class_weight=None,n_jobs=-1,random_state=seed,criterion='log_loss')
   m.fit(xt,tr.control_success.to_numpy());pp+=m.predict_proba(xv)[:,1]/2
  b=np.load(os.path.join(ROOT,'scratch/audit_v16_exact',f'val{s}.npz'));g=.15*b['p_lgb']+.75*b['p_cb']+.10*b['p_xgb'];rows=[]
  for w in [0,.025,.05,.1,.15,.2,.3]:
   p=np.clip(.5+1.10*((1-w)*(.65*g+.35*b['p_mlp'])+w*pp-.5)-.0045192086,1e-6,1-1e-6);sk,br=skill(va.control_success.to_numpy(),p);rows.append({'weight':w,'skill':sk,'brier':br})
  out[str(s)]={'best':max(rows,key=lambda r:r['skill']),'grid':rows};print(s,out[str(s)],flush=True);del tr,va,xt,xv;gc.collect()
 json.dump(out,open(os.path.join(ROOT,'scratch/extratrees_exact_results.json'),'w'),indent=2)
if __name__=='__main__':main()
