"""Nested screen for stable train-only group residual corrections."""
import itertools, json, os, sys, time
ROOT=os.path.expanduser("~/LG_data"); sys.path[:0]=[ROOT,os.path.join(ROOT,"core")]
import numpy as np
import pandas as pd
import config
from eval_utils import calc_brier_skill_score

OUT=os.path.join(ROOT,"scratch","stable_group_residuals_prevseason_results.json")

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def pred(year):
 z=np.load(os.path.join(ROOT,"scratch","cache_final",f"final_val{year}.npz"))
 p=.15*(z['p_lgb']-.007)+.75*(z['p_cb']-.008)+.10*(z['p_xgb']-.006)
 return np.clip(p,1e-5,1-1e-5),z['y']

def features(d):
 x=pd.DataFrame(index=d.index)
 x['count']=d.balls_before.astype(str)+'_'+d.strikes_before.astype(str)
 x['base']=d.base_state.fillna(-1).astype(str)
 x['outs']=d.outs_before.fillna(-1).astype(str)
 x['inning3']=np.minimum(d.inning.fillna(0).astype(int)//3,4).astype(str)
 x['month']=d.game_month.astype(str)
 x['tb']=d.top_bottom.astype(str); x['gt']=d.game_type.astype(str)
 x['hands']=d.pitcher_hand.astype(str)+'_'+d.batter_hand.astype(str)
 x['runners']=d.num_runners_on.fillna(0).astype(str)
 x['score5']=np.clip(d.score_diff_pitcher_team.fillna(0),-4,4).astype(int).astype(str)
 x['li5']=np.clip(np.floor(d.li.fillna(0)*2),0,5).astype(int).astype(str)
 x['pnlog']=np.floor(np.log2(d.asof_pitcher_n.fillna(0)+1)).astype(int).astype(str)
 x['bnlog']=np.floor(np.log2(d.asof_batter_n.fillna(0)+1)).astype(int).astype(str)
 x['psr20']=np.clip(np.floor(d.asof_pitcher_success_rate.fillna(.5)*20),0,19).astype(int).astype(str)
 x['bsr20']=np.clip(np.floor(d.asof_batter_success_rate.fillna(.5)*20),0,19).astype(int).astype(str)
 return x

def apply(hist,val,cols,m,alpha):
 g=hist.groupby(cols,dropna=False).resid.agg(sum='sum',n='count').reset_index()
 g['corr']=alpha*g['sum']/(g['n']+m)
 q=val.merge(g[cols+['corr']],on=cols,how='left',sort=False)
 assert len(q)==len(val)
 return np.clip(val.p.to_numpy()+q['corr'].fillna(0).to_numpy(),1e-6,1-1e-6)

def main():
 df=pd.read_csv(config.TRAIN_PATH); frames={}
 for yr in (2021,2022,2023,2024):
  d=df[df.season==yr].reset_index(drop=True); p,y=pred(yr)
  assert np.array_equal(d.control_success.to_numpy(),y)
  f=features(d); f['p']=p; f['y']=y; f['resid']=y-p; frames[yr]=f
 singles=['count','base','outs','inning3','month','tb','gt','hands','runners','score5','li5','pnlog','bnlog','psr20','bsr20']
 pairs=[('count',z) for z in singles if z!='count']+[
  ('base','outs'),('base','runners'),('hands','psr20'),('pnlog','psr20'),
  ('bnlog','bsr20'),('inning3','score5'),('inning3','li5')]
 candidates=[(x,) for x in singles]+pairs
 rows=[]
 for cols in candidates:
  for m in (100,300,1000,3000,10000):
   for alpha in (.25,.5,1.0):
    gains={}
    for yr in (2022,2023):
     hist=frames[yr-1]
     va=frames[yr]; pp=apply(hist,va,list(cols),m,alpha)
     gains[yr]=calc_brier_skill_score(va.y,pp)[0]-calc_brier_skill_score(va.y,va.p)[0]
    rows.append({'cols':list(cols),'m':m,'alpha':alpha,'g2022':gains[2022],
                 'g2023':gains[2023],'inner_mean':np.mean(list(gains.values())),
                 'inner_min':min(gains.values())})
 rows.sort(key=lambda r:(r['inner_min'],r['inner_mean']),reverse=True)
 # Strictly one outer touch: highest worst-inner candidate.
 best=rows[0]; hist=frames[2023]; va=frames[2024]
 po=apply(hist,va,best['cols'],best['m'],best['alpha'])
 best['g2024']=calc_brier_skill_score(va.y,po)[0]-calc_brier_skill_score(va.y,va.p)[0]
 json.dump({'best':best,'top20':rows[:20]},open(OUT,'w'),indent=2)
 log({'best':best,'top5':rows[:5]})

if __name__=='__main__': main()
