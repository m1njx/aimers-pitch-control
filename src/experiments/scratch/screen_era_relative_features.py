"""Train-only era-relative standardization features on the 116-feature frame."""
import json,os,sys,time
ROOT=os.path.expanduser("~/LG_data");sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import lightgbm as lgb,numpy as np,pandas as pd
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','era_relative_features_results.json');SEEDS=[7,123,2025]
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def add_rel(raw_tr,raw_va,xt,xv):
 cols=[c for c in xt if ((c.startswith('asof_') and 'rate' in c) or (c.startswith('cs_') and (c.endswith('_rate') or c.endswith('_hist') or c.endswith('_eb'))))]
 seasons=sorted(raw_tr.season.unique());outt=pd.DataFrame(index=xt.index);outv=pd.DataFrame(index=xv.index)
 for c in cols:
  vals=xt[c].astype(float);stats=pd.DataFrame({'v':vals,'s':raw_tr.season.to_numpy()}).groupby('s').v.agg(['mean','std']);mu=raw_tr.season.map(stats['mean']).to_numpy();sd=raw_tr.season.map(stats['std']).to_numpy();outt['era_'+c]=((vals.to_numpy()-mu)/np.maximum(sd,1e-6)).astype(np.float32);lm,ls=stats.iloc[-1];outv['era_'+c]=((xv[c].astype(float)-lm)/max(ls,1e-6)).astype(np.float32)
 return pd.concat([xt,outt],axis=1),pd.concat([xv,outv],axis=1)
def model(seed):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=seed,verbosity=-1,n_jobs=-1)
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();res={}
 for vs in (2022,2023,2024):
  tr=(df.season<vs).to_numpy();va=(df.season==vs).to_numpy();a=df[tr].copy();b=df[va].copy();bt,bv=build_base_features(a,b,vs-1,fix_index=True);de=AsofDecomposer2().fit(a,vs);at,av=de.transform(a),de.transform(b);base_t,base_v=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);rel_t,rel_v=add_rel(a,b,base_t,base_v);res[str(vs)]={}
  for name,xt,xv in [('base',base_t,base_v),('era_relative',rel_t,rel_v)]:
   ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)];ps=[model(s).fit(xt,y[tr],categorical_feature=ci).predict_proba(xv)[:,1]-.007 for s in SEEDS];p=np.mean(ps,0);sk,br,_,_=calc_brier_skill_score(y[va],p);res[str(vs)][name]={'skill':sk,'brier':br,'features':xt.shape[1]};log(f"{vs} {name}: {sk:.3f}")
  json.dump(res,open(OUT,'w'),indent=2)
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
