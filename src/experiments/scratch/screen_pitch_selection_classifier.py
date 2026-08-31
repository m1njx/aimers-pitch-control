"""Learn legal pre-pitch pitch-family probabilities from historical Trackman rows."""
import json,os,sys,time
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import config
from agent2_common import build_base_features,base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from core.eval_utils import calc_brier_skill_score
MAP=pd.read_csv(os.path.join(ROOT,'scratch','agent3_cache','pitcher_map_raw.csv'));PID=dict(zip(MAP[MAP.margin>=.3].tm_id,MAP[MAP.margin>=.3].pitcher_id));TEAM=json.load(open(os.path.join(ROOT,'scratch','agent3_cache','team_map.json')))
PCOLS=['pid','game_month','game_dayofweek','inning','top_bottom','balls_before','strikes_before','outs_before','pitcher_hand','batter_hand','pitcher_team','batter_team']
def log(x):print(f'[{time.strftime("%H:%M:%S")}] {x}',flush=True)
def tm_frame(asof):
 cols=['season','game_month','game_dayofweek','inning','top_bottom','balls_before','strikes_before','outs_before','pitcher_trackman_id','pitcher_hand','batter_hand','pitcher_team','batter_team','pitch_type_group']
 t=pd.read_csv(os.path.join(ROOT,'open','data','trackman_history.csv'),usecols=cols);t=t[t.season<=asof].copy();t=t[t.pitch_type_group.isin(['fastball','breaking','offspeed'])];t['pid']=t.pitcher_trackman_id.map(PID);t=t.dropna(subset=['pid']);t['pid']=t.pid.astype(int).astype(str);t['y']=t.pitch_type_group.map({'fastball':0,'breaking':1,'offspeed':2}).astype(int);return t
def main_frame(d):
 z=pd.DataFrame(index=d.index);z['pid']=d.pitcher_id.astype(int).astype(str);z['game_month']=d.game_month;z['game_dayofweek']=d.game_dayofweek;z['inning']=d.inning;z['top_bottom']=d.top_bottom;z['balls_before']=d.balls_before;z['strikes_before']=d.strikes_before;z['outs_before']=d.outs_before;z['pitcher_hand']=d.pitcher_hand;z['batter_hand']=d.batter_hand;z['pitcher_team']=d.pitcher_team_id.astype(str).map(TEAM).fillna('UNK');z['batter_team']=d.batter_team_id.astype(str).map(TEAM).fillna('UNK');return z
def psel(asof,tr,va):
 t=tm_frame(asof);x=t[PCOLS].copy();a=main_frame(tr);b=main_frame(va);cats=list(PCOLS)
 for c in cats:x[c]=x[c].astype(str);a[c]=a[c].astype(str);b[c]=b[c].astype(str)
 m=CatBoostClassifier(iterations=180,depth=7,learning_rate=.08,l2_leaf_reg=15,loss_function='MultiClass',random_seed=7,verbose=0,cat_features=cats,thread_count=-1,allow_writing_files=False);m.fit(x,t.y);return m.predict_proba(a),m.predict_proba(b)
def model(s):return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=s,verbosity=-1,n_jobs=-1)
def sk(y,p):return float(calc_brier_skill_score(y,p)[0])
def main():
 df=pd.read_csv(config.TRAIN_PATH);yy=df.control_success.to_numpy();res={}
 # Inner screen only; unlike the target model this prior model sees no validation-season Trackman.
 for vs in (2022,2023):
  mt=df.season<vs;mv=df.season==vs;tr=df[mt].copy();va=df[mv].copy();qtr,qv=psel(vs-1,tr,va);bt,bv=build_base_features(tr,va,vs-1,fix_index=True);de=AsofDecomposer2().fit(tr,vs);at,av=de.transform(tr),de.transform(va);xt,xv=pd.concat([bt,at],axis=1),pd.concat([bv,av],axis=1)
  for j,n in enumerate(('psel_fb','psel_br','psel_os')):xt[n]=qtr[:,j];xv[n]=qv[:,j]
  xt['psel_entropy']=-(qtr*np.log(np.clip(qtr,1e-6,1))).sum(1);xv['psel_entropy']=-(qv*np.log(np.clip(qv,1e-6,1))).sum(1);ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)];p=[]
  for s in (7,123,2025):p.append(model(s).fit(xt,yy[mt],categorical_feature=ci).predict_proba(xv)[:,1]-.007)
  # Baseline LGB cache allows a precise component comparison.
  base=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'))['p_lgb']-.007;pp=np.mean(p,0);res[str(vs)]={'base_lgb':sk(yy[mv],base),'psel_lgb':sk(yy[mv],pp),'gain':sk(yy[mv],pp)-sk(yy[mv],base),'mapping_rows':len(tm_frame(vs-1))};json.dump(res,open(os.path.join(ROOT,'scratch','pitch_selection_classifier_results.json'),'w'),indent=2);log(f'RESULT {vs}: {res[str(vs)]}')
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()
