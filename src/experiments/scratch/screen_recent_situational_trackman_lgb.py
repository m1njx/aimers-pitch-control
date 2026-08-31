"""Does last-season situational repertoire drift add signal beyond all-history priors?"""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd,lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from agent3_tkm_sit import build_situational,attach,_tm,TARGETS
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123,2025];OUT=os.path.join(ROOT,'scratch','recent_situational_trackman_lgb_results.json')
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def recent_table(asof,F,k=80.):
 t=_tm();t=t[t.season==asof];grp=['pid','b','s','bh'];m=t.groupby(grp)[TARGETS].mean();n=t.groupby(grp).size();base=F[[f'sit_{c}' for c in TARGETS]].copy();base.columns=TARGETS;idx=base.index.union(m.index);bv=base.reindex(idx);mv=m.reindex(idx);nv=n.reindex(idx).fillna(0);lam=(nv/(nv+k)).to_numpy()[:,None];est=lam*mv.fillna(bv).to_numpy()+(1-lam)*bv.to_numpy();R=pd.DataFrame(index=idx)
 for j,c in enumerate(TARGETS):R[f'recent_{c}']=est[:,j].astype(np.float32);R[f'recent_delta_{c}']=(est[:,j]-bv[c].to_numpy()).astype(np.float32)
 R['recent_n']=np.log1p(nv).astype(np.float32);return R
def model(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);yy=df.control_success.to_numpy();res={}
 for vs in (2022,2023,2024):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();bt,bv=build_base_features(tr,va,vs-1,fix_index=True);de=AsofDecomposer2().fit(tr,vs);at,av=de.transform(tr),de.transform(va);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1);F=build_situational(vs-1);st,sv=attach(F,tr,xt.copy()),attach(F,va,xv.copy());R=recent_table(vs-1,F);rt,rv=attach(R,tr,st.copy()),attach(R,va,sv.copy());ci=[st.columns.get_loc(c) for c in base_cat_cols(st)];p0=[];p1=[]
  for s in SEEDS:p0.append(model(s).fit(st,yy[mt],categorical_feature=ci).predict_proba(sv)[:,1]-.007);p1.append(model(s).fit(rt,yy[mt],categorical_feature=ci).predict_proba(rv)[:,1]-.007)
  a,b=np.mean(p0,0),np.mean(p1,0);res[str(vs)]={'sit':sk(yy[mv],a),'sit_recent':sk(yy[mv],b),'gain':sk(yy[mv],b)-sk(yy[mv],a)};json.dump(res,open(OUT,'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
