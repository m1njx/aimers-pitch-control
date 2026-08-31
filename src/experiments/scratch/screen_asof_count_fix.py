"""Screen the AsofDecomposer2 pitcher-count overwrite bug fix."""
import json, os, sys, time
ROOT=os.path.expanduser("~/LG_data"); sys.path[:0]=[os.path.join(ROOT,"scratch"),ROOT]
import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from agent2_common import build_base_features, base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score

OUT=os.path.join(ROOT,"scratch","asof_count_fix_results.json"); SEEDS=[7,123,2025]
def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)

def corrected(dec,df,add_pitchmix=False):
 x=dec.transform(df); is_val=df.season.to_numpy()==dec.val_season_
 key=pd.MultiIndex.from_arrays([df.pitcher_id.to_numpy(),df.season.to_numpy()])
 st,tot,_=dec.exact_['p_succ']; e=st.reindex(key); e.index=df.index
 et=tot.reindex(df.pitcher_id.to_numpy()); et.index=df.index
 hn=e.cum_n.to_numpy().copy(); hn[is_val]=et.cum_n.to_numpy()[is_val]; hn=np.nan_to_num(hn)
 x['cs_pit_hist_n']=hn.astype(np.float32)
 x['cs_pit_cur_n']=np.maximum(df.asof_pitcher_n.fillna(0).to_numpy()-hn,0).astype(np.float32)
 if add_pitchmix:
  b=dec.pb_.reindex(key); b.index=df.index
  bl=dec.pb_val_.reindex(df.pitcher_id.to_numpy()); bl.index=df.index
  ph=b['__den_p_fb'].to_numpy().copy(); ph[is_val]=bl['__den_p_fb'].to_numpy()[is_val]
  ph=np.nan_to_num(ph)
  x['cs_pitchmix_hist_n']=ph.astype(np.float32)
  x['cs_pitchmix_cur_n']=np.maximum(df.asof_pitcher_pitchmix_n.fillna(0).to_numpy()-ph,0).astype(np.float32)
 return x

def model(seed):
 return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,
  colsample_bytree=.7,subsample=.7,random_state=seed,verbosity=-1,n_jobs=-1)

def main():
 df=pd.read_csv(config.TRAIN_PATH); y=df.control_success.to_numpy(); res={}
 for vs in (2022,2023,2024):
  tr=(df.season<vs).to_numpy(); va=(df.season==vs).to_numpy(); dtr=df[tr].copy(); dva=df[va].copy()
  bt,bv=build_base_features(dtr,dva,vs-1,fix_index=True); dec=AsofDecomposer2().fit(dtr,vs)
  variants={'buggy':(dec.transform(dtr),dec.transform(dva)),
            'fixed':(corrected(dec,dtr),corrected(dec,dva)),
            'fixed_pitchmix':(corrected(dec,dtr,True),corrected(dec,dva,True))}
  res[str(vs)]={}
  for name,(at,av) in variants.items():
   xt=pd.concat([bt,at],axis=1); xv=pd.concat([bv,av],axis=1); ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)]
   ps=[]
   for seed in SEEDS:
    ps.append(model(seed).fit(xt,y[tr],categorical_feature=ci).predict_proba(xv)[:,1]-.007)
   p=np.mean(ps,axis=0); sk,br,_,_=calc_brier_skill_score(y[va],p)
   res[str(vs)][name]={'skill':float(sk),'brier':float(br)}; log(f"{vs} {name} {sk:.3f}")
  json.dump(res,open(OUT,'w'),indent=2)
 print(json.dumps(res,indent=2))

if __name__=='__main__': main()
