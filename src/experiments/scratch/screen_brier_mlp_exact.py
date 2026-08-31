"""Brier-loss MLP screen on the exact v33 feature construction; no test.csv."""
import json, os, random, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(ROOT,'scratch'))
from audit_v16_exact_cv import add_features, SimpleMLP

cats=['top_bottom','base_state','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id','count_code','platoon_matchup','tkm_match','count_x_base']
def skill(y,p):
 b=float(np.mean((y-p)**2)); return 100000*(1-b/(y.mean()*(1-y.mean()))),b
def one(season):
 d=pd.read_csv(os.path.join(ROOT,'open/data/train.csv'));tr=d[d.season<season].copy();va=d[d.season==season].copy();xt,xv=add_features(tr,va,season)
 cc=[c for c in xt if c in cats]; nnc=[c for c in xt if c not in cc]; mean=xt[nnc].to_numpy(np.float32).mean(0);std=xt[nnc].to_numpy(np.float32).std(0);std[std<1e-6]=1
 nt=torch.tensor(np.nan_to_num((xt[nnc].to_numpy(np.float32)-mean)/std));nv=torch.tensor(np.nan_to_num((xv[nnc].to_numpy(np.float32)-mean)/std))
 voc=[{v:i for i,v in enumerate(sorted(xt[c].astype(str).unique()))} for c in cc];cards=[len(v)+1 for v in voc]
 def enc(x):return torch.tensor(np.stack([x[c].astype(str).map(v).fillna(len(v)).to_numpy(np.int64) for c,v in zip(cc,voc)],1))
 ct,cv=enc(xt),enc(xv);yt=torch.tensor(tr.control_success.to_numpy(np.float32));ld=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(nt,ct,yt),batch_size=4096,shuffle=True)
 p=np.zeros(len(va))
 for seed in [7,2025]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);m=SimpleMLP(len(nnc),cards);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-5)
  for ep in range(8):
   for a,b,y in ld:
    opt.zero_grad();q=torch.sigmoid(m(a,b));loss=((q-y)**2).mean();loss.backward();opt.step()
   print(f'{season} seed{seed} ep{ep+1}',flush=True)
  m.eval()
  with torch.no_grad():p+=torch.sigmoid(m(nv,cv)).numpy()/2
 base=np.load(os.path.join(ROOT,'scratch/audit_v16_exact',f'val{season}.npz'));g=.15*base['p_lgb']+.75*base['p_cb']+.10*base['p_xgb']; rows=[]
 for r in [0,.25,.5,.75,1]:
  pm=(1-r)*base['p_mlp']+r*p;z=np.clip(.5+1.1*((.65*g+.35*pm)-.5)-.0045192086,1e-6,1-1e-6);s,b=skill(va.control_success.to_numpy(),z);rows.append({'replace':r,'skill':s,'brier':b})
 return rows
out={str(s):one(s) for s in [2022,2023]};json.dump(out,open(os.path.join(ROOT,'scratch/brier_mlp_exact_results.json'),'w'),indent=2);print(out,flush=True)
