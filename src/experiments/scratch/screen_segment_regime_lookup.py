"""Inner-select one legal fixed segment drift lookup; outer-check selected only."""
import os,sys,json
ROOT=os.path.expanduser('~/LG_data');sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np,pandas as pd
import config
from core.eval_utils import calc_brier_skill_score
OUT=os.path.join(ROOT,'scratch','segment_regime_lookup_results.json')
def sk(y,p):return float(calc_brier_skill_score(y,np.clip(p,1e-6,1-1e-6))[0])
def keys(d,name):
 c=d.balls_before.clip(0,3).astype(int)*3+d.strikes_before.clip(0,2).astype(int)
 platoon=(d.pitcher_hand.astype(int)-1)*2+(d.batter_hand.astype(int)-1)
 base=((d.runner_on_1b.fillna(0)>0).astype(int)+2*(d.runner_on_2b.fillna(0)>0).astype(int)+4*(d.runner_on_3b.fillna(0)>0).astype(int));outs=d.outs_before.clip(0,2).astype(int);inn=np.minimum((d.inning.fillna(1).astype(int)-1)//3,2)
 score=np.sign(d.score_diff_pitcher_team.fillna(0)).astype(int)+1
 return {'count':c,'count_platoon':c*4+platoon,'count_base':c*8+base,'count_outs':c*3+outs,'count_inning':c*3+inn,'count_score':c*3+score}[name].to_numpy()
def table(tr,name,k=300):
 d=tr.copy();d['g']=keys(d,name);z=d.groupby(['season','g']).control_success.agg(['mean','size']).reset_index();glob=d.groupby('season').control_success.mean();z['rel']=z['mean']-z.season.map(glob);p=z.pivot(index='g',columns='season',values='rel');n=z.pivot(index='g',columns='season',values='size');ss=sorted(d.season.unique());last=ss[-1];pool=(p[ss]*n[ss]).sum(1)/n[ss].sum(1);fc=p[last]+.5*p[ss].diff(axis=1).mean(axis=1);lam=n[last]/(n[last]+k);fc=lam*fc+(1-lam)*p[last];corr=(fc-pool).fillna(0);corr-=np.average(corr,weights=n[last].fillna(0));return corr
def basepred(vs):
 c=np.load(os.path.join(ROOT,'scratch','cache_final',f'final_val{vs}.npz'));m=np.load(os.path.join(ROOT,'scratch','multitask_aux_preds',f'val{vs}.npz'))['p'];g=.15*(c['p_lgb']-.007)+.75*(c['p_cb']-.008)+.10*(c['p_xgb']-.006);r=.68*g+.32*m;return np.clip(.5+1.1*(r-.5)-.0045192086,1e-6,1-1e-6)
def evaluate(df,vs,name,a):
 tr=df[df.season<vs];va=df[df.season==vs];p=basepred(vs);co=table(tr,name);delta=pd.Series(keys(va,name)).map(co).fillna(0).to_numpy();return sk(va.control_success,p+a*delta)-sk(va.control_success,p),sk(va.control_success,p+a*delta)
def main():
 df=pd.read_csv(config.TRAIN_PATH);names=['count','count_platoon','count_base','count_outs','count_inning','count_score'];alphas=[.15,.25,.4,.6];inner={}
 for n in names:
  for a in alphas:
   gs=[evaluate(df,v,n,a)[0] for v in (2022,2023)];inner[f'{n}|{a}']={'gains':gs,'mean':float(np.mean(gs)),'min':float(np.min(gs))}
 # Robust selection: maximize mean only among candidates positive in both folds.
 valid={k:v for k,v in inner.items() if v['min']>0};sel=max(valid,key=lambda k:valid[k]['mean']) if valid else max(inner,key=lambda k:inner[k]['mean']);n,a=sel.split('|');outer_gain,outer_skill=evaluate(df,2024,n,float(a));out={'selected':sel,'inner':inner[sel],'outer_gain':outer_gain,'outer_skill':outer_skill,'all_inner':inner};json.dump(out,open(OUT,'w'),indent=2);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
