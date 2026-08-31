"""Situation-conditional historical Trackman variability, strictly prior-only."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from agent3_tkm_sit import _tm,build_situational,attach
from core.eval_utils import calc_brier_skill_score
COLS=['rel_speed','induced_vert_break','horz_break','rel_height','rel_side','extension','spin_rate']
SEEDS=[7,123,2025]
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def vartable(asof,F,k=50.):
 t=_tm();t=t[t.season<=asof];g=['pid','b','s','bh'];std=t.groupby(g)[COLS].std();n=t.groupby(g).size();pstd=t.groupby('pid')[COLS].std();glob=t[COLS].std();idx=F.index.union(std.index);pid=idx.get_level_values('pid');base=pstd.reindex(pid).set_axis(idx).fillna(glob);raw=std.reindex(idx);nv=n.reindex(idx).fillna(0);lam=(nv/(nv+k)).to_numpy()[:,None];est=lam*raw.fillna(base).to_numpy()+(1-lam)*base.to_numpy();R=pd.DataFrame(index=idx)
 for j,c in enumerate(COLS):R['sitstd_'+c]=est[:,j].astype(np.float32)
 R['sitstd_n']=np.log1p(nv).astype(np.float32);return R
def mod(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);y=df.control_success.to_numpy();out={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);d=AsofDecomposer2().fit(tr,vs);at,av=d.transform(tr),d.transform(va);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);F=build_situational(vs-1);st,sv=attach(F,tr,xt.copy()),attach(F,va,xv.copy());V=vartable(vs-1,F);vt,vv=attach(V,tr,st.copy()),attach(V,va,sv.copy());ci=[st.columns.get_loc(c) for c in base_cat_cols(st)];a=[];b=[]
  for s in SEEDS:a.append(mod(s).fit(st,y[mt],categorical_feature=ci).predict_proba(sv)[:,1]-.007);b.append(mod(s).fit(vt,y[mt],categorical_feature=ci).predict_proba(vv)[:,1]-.007)
  p,q=np.mean(a,0),np.mean(b,0);out[str(vs)]={'sit':sk(y[mv],p),'sit_variability':sk(y[mv],q),'gain':sk(y[mv],q)-sk(y[mv],p)};json.dump(out,open(os.path.join(ROOT,'scratch','situational_trackman_variability_results.json'),'w'),indent=2);log(f'RESULT {vs}: {out[str(vs)]}')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
