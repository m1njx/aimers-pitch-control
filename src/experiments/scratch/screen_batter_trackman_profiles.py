"""Official-data, train-only historical Trackman exposure profiles per batter."""
import gc,json,os,sys,time
ROOT=os.path.expanduser("~/LG_data");sys.path[:0]=[os.path.join(ROOT,'scratch'),os.path.join(ROOT,'track_claude_z'),ROOT]
import numpy as np,pandas as pd
import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score
from screen_id_trackman_profiles import build_fold,cat_cols,predict
SEEDS=[7,123];OUT=os.path.join(ROOT,'scratch','batter_trackman_profile_results.json')
PHYS=['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side','zone_speed']
def log(x):print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def attach(X,raw,asof):
 mp=pd.read_csv(os.path.join(ROOT,'scratch','agent3_cache','batter_map_raw.csv'));mp=mp[mp.margin>=.3];idmap=dict(zip(mp.tm_id,mp.batter_id))
 cols=['season','batter_trackman_id','pitcher_hand','pitch_type_group','auto_pitch_type']+PHYS
 tm=pd.read_csv(config.TRACKMAN_PATH,usecols=cols);tm=tm[tm.season<=asof].copy();tm['bid']=tm.batter_trackman_id.map(idmap);tm=tm.dropna(subset=['bid']);tm.bid=tm.bid.astype(np.int32)
 def block(d,p):
  a=d.groupby('bid')[PHYS].agg(['mean','std']);a.columns=[f'{p}{c}_{s}' for c,s in a.columns];a[f'{p}n']=d.groupby('bid').size();return a
 prof=block(tm,'btk_all_').join(block(tm[tm.pitcher_hand=='Left'],'btk_lhp_')).join(block(tm[tm.pitcher_hand=='Right'],'btk_rhp_')).join(block(tm[tm.season==asof],'btk_recent_'))
 mix=pd.crosstab(tm.bid,tm.pitch_type_group,normalize='index');mix.columns=[f'btk_mix_{c}' for c in mix.columns];prof=prof.join(mix);prof['btk_pitch_types_seen']=tm.groupby('bid').auto_pitch_type.nunique()
 prof['btk_lr_velo_gap']=prof.get('btk_lhp_rel_speed_mean')-prof.get('btk_rhp_rel_speed_mean');prof['btk_move_mag']=np.hypot(prof.btk_all_induced_vert_break_mean,prof.btk_all_horz_break_mean)
 v=prof.reindex(raw.batter_id.to_numpy());v.index=X.index;X=pd.concat([X,v.astype(np.float32)],axis=1);X['btk_mapped']=v.notna().any(axis=1).astype(np.int8);return X
def main():
 df=pd.read_csv(config.TRAIN_PATH);res={}
 for f in get_cv_folds(df)[:2]:
  tr,va,xt,xv=build_fold(df,f);xt=attach(xt,tr,f.fold_max_season);xv=attach(xv,va,f.fold_max_season);cats=cat_cols(xt);ps=[]
  for s in SEEDS:log(f"fold={f.val_season} seed={s} features={xt.shape[1]}");ps.append(predict(xt,xv,tr.control_success.to_numpy(),cats,s))
  p=np.mean(ps,0);sk,br,_,_=calc_brier_skill_score(va.control_success.to_numpy(),p);res[str(f.val_season)]={'skill':sk,'brier':br,'coverage':float(xv.btk_mapped.mean()),'features':xt.shape[1]};json.dump(res,open(OUT,'w'),indent=2);log(f"RESULT {f.val_season}: {res[str(f.val_season)]}");del tr,va,xt,xv;gc.collect()
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
