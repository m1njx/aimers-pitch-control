"""CatBoost confirmation of stable era-relative features, with saved deltas."""
import json,os,sys,time
ROOT=os.path.expanduser("~/LG_data");sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from screen_era_relative_features import add_rel
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','era_relative_catboost_results.json');PDIR=os.path.join(ROOT,'scratch','era_relative_catboost_preds');SEEDS=[7,123]
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def prep(x,y):
 a,b=x.copy(),y.copy();cats=base_cat_cols(a)
 for c in cats:a[c]=a[c].fillna(-1).astype(str);b[c]=b[c].fillna(-1).astype(str)
 for c in a:
  if c not in cats:a[c]=a[c].astype(np.float32);b[c]=b[c].astype(np.float32)
 return a,b,cats
def pred(x,y,t,seed):
 a,b,c=prep(x,y);m=CatBoostClassifier(iterations=250,depth=6,learning_rate=.06,l2_leaf_reg=10,random_seed=seed,verbose=0,cat_features=c,thread_count=-1,allow_writing_files=False);m.fit(a,t);return m.predict_proba(b)[:,1]-.008
def main():
 os.makedirs(PDIR,exist_ok=True);df=pd.read_csv(config.TRAIN_PATH);yy=df.control_success.to_numpy();res={}
 for vs in (2022,2023,2024):
  tr=(df.season<vs).to_numpy();va=(df.season==vs).to_numpy();a=df[tr].copy();b=df[va].copy();bt,bv=build_base_features(a,b,vs-1,fix_index=True);de=AsofDecomposer2().fit(a,vs);at,av=de.transform(a),de.transform(b);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);rt,rv=add_rel(a,b,xt,xv);pb=[];pr=[]
  for s in SEEDS:log(f"{vs} seed={s} base");pb.append(pred(xt,xv,yy[tr],s));log(f"{vs} seed={s} era");pr.append(pred(rt,rv,yy[tr],s))
  p0,p1=np.mean(pb,0),np.mean(pr,0);s0=calc_brier_skill_score(yy[va],p0)[0];s1=calc_brier_skill_score(yy[va],p1)[0];res[str(vs)]={'base':s0,'era':s1,'gain':s1-s0};np.savez_compressed(os.path.join(PDIR,f'val{vs}.npz'),y=yy[va],base=p0,era=p1);json.dump(res,open(OUT,'w'),indent=2);log(f"RESULT {vs}: {res[str(vs)]}")
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
