"""Exact-v23-frame era-relative LGB candidate; stores predictions and results."""
import os,sys,json
ROOT=os.path.expanduser('~/LG_data');PKG=os.path.join(ROOT,'work','submit_v16');sys.path.insert(0,PKG)
import numpy as np,pandas as pd,lightgbm as lgb
from audit_v16_exact_cv import add_features
from core.eval_utils import calc_brier_skill_score
SEEDS=(7,123)
def add_rel(tr,va,xt,xv):
 cols=[c for c in xt if c.startswith('asof_') and 'rate' in c or c.startswith('cs_') and (c.endswith('_rate') or c.endswith('_hist') or c.endswith('_eb'))]
 a=pd.DataFrame(index=xt.index);b=pd.DataFrame(index=xv.index)
 for c in cols:
  st=pd.DataFrame({'x':xt[c].values,'s':tr.season.values}).groupby('s').x.agg(['mean','std']);mu=tr.season.map(st['mean']).values;sd=tr.season.map(st['std']).values
  a['era_'+c]=((xt[c].values-mu)/np.maximum(sd,1e-6)).astype('float32');b['era_'+c]=((xv[c].values-st.iloc[-1,0])/max(st.iloc[-1,1],1e-6)).astype('float32')
 return pd.concat([xt,a],axis=1),pd.concat([xv,b],axis=1)
def main():
 df=pd.read_csv(os.path.join(ROOT,'open','data','train.csv'));res={};out=os.path.join(ROOT,'scratch','era_relative_exact');os.makedirs(out,exist_ok=True)
 for vs in (2022,2023):
  tr=df[df.season<vs].copy();va=df[df.season==vs].copy();xt,xv=add_features(tr,va,vs);xt,xv=add_rel(tr,va,xt,xv)
  cats=[c for c in xt if c in ('top_bottom','base_state','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id','count_code','platoon_matchup','tkm_match','count_x_base')];ci=[xt.columns.get_loc(c) for c in cats];ps=[]
  for s in SEEDS:
   print(vs,s,xt.shape,flush=True);m=lgb.LGBMClassifier(n_estimators=300,num_leaves=31,learning_rate=.05,min_child_samples=50,subsample=.8,colsample_bytree=.8,random_state=s,verbosity=-1,n_jobs=-1);m.fit(xt,tr.control_success,categorical_feature=ci);ps.append(m.predict_proba(xv)[:,1])
  q=np.mean(ps,0);z=np.load(os.path.join(ROOT,'scratch','audit_v16_exact',f'val{vs}.npz'));base=.15*np.clip(z['p_lgb']-.007,1e-6,1-1e-6)+.75*np.clip(z['p_cb']-.008,1e-6,1-1e-6)+.10*np.clip(z['p_xgb']-.006,1e-6,1-1e-6);r={}
  for w in (0,.15,.3,.5,.75,1):
   g=.15*((1-w)*np.clip(z['p_lgb']-.007,1e-6,1-1e-6)+w*np.clip(q-.007,1e-6,1-1e-6))+.75*np.clip(z['p_cb']-.008,1e-6,1-1e-6)+.10*np.clip(z['p_xgb']-.006,1e-6,1-1e-6);r[str(w)]=float(calc_brier_skill_score(z['y'],.68*g+.32*z['p_mlp'])[0])
  res[str(vs)]={'features':int(xt.shape[1]),'scores':r};np.save(os.path.join(out,f'val{vs}_lgb.npy'),q);print(vs,res[str(vs)],flush=True)
 json.dump(res,open(os.path.join(ROOT,'scratch','era_relative_exact_results.json'),'w'),indent=2)
if __name__=='__main__':main()
