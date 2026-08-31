"""Era-neutral target regression on the current legal 116-feature frame."""
import json,os,sys,time
ROOT=os.path.expanduser("~/LG_data");sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import lightgbm as lgb,numpy as np,pandas as pd
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','era_target_asof_results.json');SEEDS=[7,123,2025]
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def model(seed):return lgb.LGBMRegressor(objective='regression',n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=seed,verbosity=-1,n_jobs=-1)
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy(np.float32);res={}
 for vs in (2022,2023,2024):
  tr=(df.season<vs).to_numpy();va=(df.season==vs).to_numpy();a=df[tr].copy();b=df[va].copy();xt,xv=build_base_features(a,b,vs-1,fix_index=True);de=AsofDecomposer2().fit(a,vs);at,av=de.transform(a),de.transform(b);xt=pd.concat([xt,at],axis=1);xv=pd.concat([xv,av],axis=1);ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)]
  means=a.groupby('season').control_success.mean();last=float(means.iloc[-1]);yt_era=y[tr]-a.season.map(means).to_numpy(np.float32)
  res[str(vs)]={}
  for name,target,offset in [('raw_l2',y[tr],0),('era_last',yt_era,last)]:
   ps=[model(s).fit(xt,target,categorical_feature=ci).predict(xv)+offset for s in SEEDS];p=np.clip(np.mean(ps,0),1e-6,1-1e-6);sk,br,_,_=calc_brier_skill_score(y[va],p);res[str(vs)][name]={'skill':sk,'brier':br,'mean':float(p.mean()),'offset':offset};log(f"{vs} {name}: {sk:.3f}")
  json.dump(res,open(OUT,'w'),indent=2)
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
