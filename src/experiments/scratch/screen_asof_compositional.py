"""Explicit compositional/log-ratio features from legal row-local asof decomposition."""
import json, os, sys, time
ROOT=os.path.expanduser("~/LG_data"); sys.path[:0]=[os.path.join(ROOT,"scratch"),ROOT]
import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,"scratch","asof_compositional_results.json"); SEEDS=[7,123,2025]
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def lg(x):
 x=np.clip(x.astype(float),1e-4,1-1e-4); return np.log(x/(1-x)).astype(np.float32)
def ratio(a,b): return (a.astype(float)/np.maximum(b.astype(float),.02)).clip(0,5).astype(np.float32)
def compose(a):
 z=pd.DataFrame(index=a.index)
 for suf in ('rate','hist','eb'):
  ps=a[f'cs_p_succ_{suf}']; pf=1-ps
  z[f'cmp_p_succ_logit_{suf}']=lg(ps)
  z[f'cmp_p_rev_given_fail_{suf}']=ratio(a[f'cs_p_rev_{suf}'],pf)
  z[f'cmp_p_mid_given_fail_{suf}']=ratio(a[f'cs_p_mid_{suf}'],pf)
  z[f'cmp_p_ball_minus_str_{suf}']=(a[f'cs_p_ball_{suf}']-a[f'cs_p_str_{suf}']).astype(np.float32)
  z[f'cmp_b_succ_logit_{suf}']=lg(a[f'cs_b_succ_{suf}'])
  z[f'cmp_b_mid_given_fail_{suf}']=ratio(a[f'cs_b_mid_{suf}'],1-a[f'cs_b_succ_{suf}'])
  rates=np.column_stack([a[f'cs_p_fb_{suf}'],a[f'cs_p_br_{suf}'],a[f'cs_p_os_{suf}']]).astype(float)
  rates=np.clip(rates,1e-5,None); rates=rates/np.maximum(rates.sum(1,keepdims=True),1e-5)
  z[f'cmp_mix_entropy_{suf}']=(-np.sum(rates*np.log(rates),axis=1)).astype(np.float32)
  z[f'cmp_fb_br_logratio_{suf}']=np.log(rates[:,0]/rates[:,1]).clip(-8,8).astype(np.float32)
  z[f'cmp_os_fb_logratio_{suf}']=np.log(rates[:,2]/rates[:,0]).clip(-8,8).astype(np.float32)
 z['cmp_pb_logit_diff_eb']=lg(a.cs_p_succ_eb)-lg(a.cs_b_succ_eb)
 z['cmp_p_cur_log_n']=np.log1p(a.cs_pit_cur_n).astype(np.float32)
 z['cmp_b_cur_log_n']=np.log1p(a.cs_bat_cur_n).astype(np.float32)
 return z.replace([np.inf,-np.inf],np.nan)
def model(seed):
 return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,
  colsample_bytree=.7,subsample=.7,random_state=seed,verbosity=-1,n_jobs=-1)
def main():
 df=pd.read_csv(config.TRAIN_PATH); y=df.control_success.to_numpy(); res={}
 for vs in (2022,2023,2024):
  tr=(df.season<vs).to_numpy(); va=(df.season==vs).to_numpy(); dtr=df[tr].copy(); dva=df[va].copy()
  bt,bv=build_base_features(dtr,dva,vs-1,fix_index=True); dec=AsofDecomposer2().fit(dtr,vs)
  at,av=dec.transform(dtr),dec.transform(dva)
  variants={'base':(at,av),'compositional':(pd.concat([at,compose(at)],axis=1),pd.concat([av,compose(av)],axis=1))}
  res[str(vs)]={}
  for name,(aa,bb) in variants.items():
   xt=pd.concat([bt,aa],axis=1); xv=pd.concat([bv,bb],axis=1); ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)]
   ps=[model(s).fit(xt,y[tr],categorical_feature=ci).predict_proba(xv)[:,1]-.007 for s in SEEDS]
   p=np.mean(ps,axis=0); sk,br,_,_=calc_brier_skill_score(y[va],p)
   res[str(vs)][name]={'skill':float(sk),'brier':float(br)}; log(f"{vs} {name}: {sk:.3f}")
  json.dump(res,open(OUT,'w'),indent=2)
 print(json.dumps(res,indent=2))
if __name__=='__main__': main()
