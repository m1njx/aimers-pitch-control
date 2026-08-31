"""Legal fixed count-regime correction derived only from prior train seasons."""
import json,os,sys
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
import config
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','count_regime_lookup_results.json')
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def key(d):return (d.balls_before.clip(0,3).astype(int)*3+d.strikes_before.clip(0,2).astype(int)).to_numpy()
def correction(tr):
 d=tr[['season','balls_before','strikes_before','control_success']].copy();d['g']=key(d)
 tab=d.groupby(['season','g']).control_success.agg(['mean','size']).reset_index();glob=d.groupby('season').control_success.mean()
 tab['rel']=tab['mean']-tab.season.map(glob);piv=tab.pivot(index='g',columns='season',values='rel');cnt=tab.pivot(index='g',columns='season',values='size');ss=sorted(d.season.unique());last=ss[-1]
 pooled=(piv[ss]*cnt[ss]).sum(1)/cnt[ss].sum(1);trend=piv[ss].diff(axis=1).mean(axis=1);forecast=piv[last]+.5*trend
 # Center with the last training season's fixed count distribution: no val/test rows used.
 corr=forecast-pooled;corr-=np.average(corr.fillna(0),weights=cnt[last].fillna(0));return corr.fillna(0)
def main():
 df=pd.read_csv(config.TRAIN_PATH);res={};alphas=[0,.25,.5,.75,1,1.25]
 for vs in (2022,2023,2024):
  tr=df[df.season<vs];va=df[df.season==vs];y=va.control_success.to_numpy();cache=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));mlp=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p'];gbdt=.15*(cache['p_lgb']-.007)+.75*(cache['p_cb']-.008)+.10*(cache['p_xgb']-.006);raw=.68*gbdt+.32*mlp;base=np.clip(.5+1.1*(raw-.5)-.0045192086,1e-6,1-1e-6);c=correction(tr);delta=pd.Series(key(va)).map(c).fillna(0).to_numpy();res[str(vs)]={'base':sk(y,base),'corr_std':float(delta.std()),'alphas':{}}
  for a in alphas:res[str(vs)]['alphas'][str(a)]={'skill':sk(y,base+a*delta),'gain':sk(y,base+a*delta)-sk(y,base)}
 json.dump(res,open(OUT,'w'),indent=2);print(json.dumps(res,indent=2))
if __name__=='__main__':main()
