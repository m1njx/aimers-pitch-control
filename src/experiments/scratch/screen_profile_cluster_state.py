"""Profile-cluster hierarchical state correction, legal and row-independent."""
import os, sys, json
ROOT=os.path.expanduser('~/LG_data'); sys.path[:0]=[os.path.join(ROOT,'scratch'),ROOT]
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from core.eval_utils import calc_brier_skill_score
from agent2_tkm_profile import PitcherTrackmanProfile, load_pitcher_map, load_trackman_upto

def baseline(v):
 z=np.load(os.path.join(ROOT,'scratch','audit_v16_exact',f'val{v}.npz'))
 g=.15*np.clip(z['p_lgb']-.007,1e-6,1-1e-6)+.75*np.clip(z['p_cb']-.008,1e-6,1-1e-6)+.10*np.clip(z['p_xgb']-.006,1e-6,1-1e-6)
 return z['y'],.68*g+.32*z['p_mlp']
def make(tr,va,asof,k=12,m=300):
 p=PitcherTrackmanProfile(load_pitcher_map()).fit(load_trackman_upto(asof))
 raw=p.prof_.copy(); cols=['p_rel_speed_mean','p_spin_rate_mean','p_induced_vert_break_mean','p_horz_break_mean','p_extension_mean','p_rel_height_mean','p_rel_side_mean','p_rel_speed_std','p_rel_height_std','p_rel_side_std','p_mix_entropy']
 raw=raw.reindex(columns=cols).replace([np.inf,-np.inf],np.nan); good=raw.notna().mean(1)>.8; fill=raw.median(); A=raw.fillna(fill)
 km=KMeans(n_clusters=k,random_state=17,n_init=20).fit(StandardScaler().fit_transform(A.loc[good]))
 cl=pd.Series(-1,index=A.index);cl.loc[good]=km.labels_; inv={tm:pid for pid,tm in p.pmap.items()}; pidcl={inv[x]:int(c) for x,c in cl.items() if c>=0 and x in inv}
 def key(d): return pd.DataFrame({'cl':d.pitcher_id.map(pidcl).fillna(-1).astype(int),'count':(d.balls_before.fillna(0).astype(int)*3+d.strikes_before.fillna(0).astype(int)),'bh':d.batter_hand.fillna(0).astype(int)})
 a=key(tr); a['y']=tr.control_success.values; glob=a.y.mean(); l2=a.groupby('cl').y.agg(['sum','size']);l2['r']=(l2['sum']+m*glob)/(l2['size']+m)
 l1=a.groupby(['cl','count']).y.agg(['sum','size']).join(l2.r.rename('prior'),on='cl');l1['r']=(l1['sum']+100*l1.prior)/(l1['size']+100)
 l0=a.groupby(['cl','count','bh']).y.agg(['sum','size']).join(l1.r.rename('prior'),on=['cl','count']);l0['r']=(l0['sum']+40*l0.prior)/(l0['size']+40)
 q=key(va);r=l0.r.reindex(pd.MultiIndex.from_frame(q)).to_numpy();r=np.where(np.isfinite(r),r,l1.r.reindex(pd.MultiIndex.from_frame(q[['cl','count']])).to_numpy());r=np.where(np.isfinite(r),r,l2.r.reindex(q.cl).to_numpy());return np.nan_to_num(r,nan=glob)-glob,float((q.cl>=0).mean())
def main():
 df=pd.read_csv(os.path.join(ROOT,'open','data','train.csv'));out={}
 for v in (2022,2023,2024):
  tr=df[df.season<v];va=df[df.season==v];y,p=baseline(v);c,cov=make(tr,va,v-1);d={}
  for a in (0,.1,.2,.3,.5,.75,1):d[str(a)]=float(calc_brier_skill_score(y,np.clip(p+a*c,1e-6,1-1e-6))[0])
  out[str(v)]={'coverage':cov,'scores':d};print(v,out[str(v)],flush=True)
 json.dump(out,open(os.path.join(ROOT,'scratch','profile_cluster_state_results.json'),'w'),indent=2)
if __name__=='__main__':main()
