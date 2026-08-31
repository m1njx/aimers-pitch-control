"""Strict temporal ablation of v40's 133-feature LightGBM-MSE branch.

Uses only train.csv and fold-legal historical Trackman artifacts.  The new
branch is blended against the saved exact-v33 temporal predictions.
"""
import json, os, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/LG_data")
sys.path[:0] = [os.path.join(ROOT, "scratch"), ROOT]
from audit_v16_exact_cv import add_features

SEEDS = [7, 123]
OUT = os.path.join(ROOT, "scratch", "v40_lgb_mse_exact_results.json")

def skill(y, p):
    b = float(np.mean((p-y)**2)); r = float(np.mean(y))
    return float(100000*(1-b/(r*(1-r)))), b

def augment(x, raw):
    z = x.copy()
    v = z['tkm_rel_speed_mean'].clip(lower=60.0)
    ext = z['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
    rh = z['tkm_rel_height_mean']; rs = z['tkm_rel_side_mean']
    ivb = z['tkm_induced_vert_break_mean']/12.0
    hb = z['tkm_horz_break_mean']/12.0
    dist = (60.5-ext).clip(lower=50.0)
    spin = z['tkm_spin_rate_mean'].clip(lower=500.0)
    z['phys_effective_velocity'] = (v*(60.5/dist)).astype('float32')
    z['phys_vaa_proxy'] = (np.arctan((rh-2.5+ivb)/dist)*180/np.pi).astype('float32')
    z['phys_haa_proxy'] = (np.arctan((rs+hb)/dist)*180/np.pi).astype('float32')
    z['phys_spin_efficiency'] = (np.sqrt((ivb*12)**2+(hb*12)**2)/spin).astype('float32')
    b=raw.balls_before.fillna(0).to_numpy(); s=raw.strikes_before.fillna(0).to_numpy()
    li=raw.li.fillna(1).to_numpy(); r2=(raw.runner_on_2b.fillna(0)>0).to_numpy(float)
    r3=(raw.runner_on_3b.fillna(0)>0).to_numpy(float)
    sd=raw.score_diff_pitcher_team.fillna(0).to_numpy(); inn=raw.inning.fillna(1).to_numpy()
    same=(raw.pitcher_hand.astype(str)==raw.batter_hand.astype(str)).to_numpy(float)
    z['feat_count_advantage']=(s-1.5*b).astype('float32')
    z['feat_full_count']=((b==3)&(s==2)).astype('float32')
    z['feat_pitcher_ahead']=((s>b)&(s>=2)).astype('float32')
    z['feat_pitcher_behind']=((b>s)&(b>=2)).astype('float32')
    z['feat_clutch_pressure']=(li*(1+r2+r3)*np.exp(-np.clip(sd**2/10,0,5))).astype('float32')
    z['feat_scoring_position']=(r2+r3).astype('float32')
    z['feat_platoon_fastball_inter']=(same*raw.asof_pitcher_fastball_rate.fillna(.5)).astype('float32')
    z['feat_platoon_breaking_inter']=(same*raw.asof_pitcher_breaking_rate.fillna(.3)).astype('float32')
    z['feat_platoon_offspeed_inter']=(same*raw.asof_pitcher_offspeed_rate.fillna(.2)).astype('float32')
    z['feat_late_inning_clutch']=((inn>=7)*li).astype('float32')
    return z

def main():
    df=pd.read_csv(os.path.join(ROOT,'open/data/train.csv')); out={}
    for yr in [2022, 2023, 2024]:
        tr=df[df.season<yr].copy(); va=df[df.season==yr].copy()
        xt,xv=add_features(tr,va,yr)
        xt,xv=augment(xt,tr),augment(xv,va)
        preds=[]
        for seed in SEEDS:
            m=lgb.LGBMRegressor(objective='regression',n_estimators=350,num_leaves=63,
                learning_rate=.05,colsample_bytree=.8,subsample=.8,subsample_freq=1,
                random_state=seed,verbosity=-1,n_jobs=-1)
            m.fit(xt,tr.control_success.to_numpy())
            preds.append(np.clip(m.predict(xv),1e-6,1-1e-6))
            print(time.strftime('%H:%M:%S'),yr,seed,flush=True)
        pm=np.mean(preds,axis=0); cache=np.load(os.path.join(ROOT,'scratch/audit_v16_exact',f'val{yr}.npz'))
        g=.20*(cache['p_lgb']-.007)+.72*(cache['p_cb']-.008)+.08*(cache['p_xgb']-.006)
        base=.65*g+.35*cache['p_mlp']; y=va.control_success.to_numpy()
        rows=[]
        for w in [0,.05,.10,.15,.20,.25,.30,.40,.50]:
            raw=(1-w)*base+w*pm
            p=np.clip(.5+1.10*(raw-.5)-.0045192086,1e-6,1-1e-6)
            s,b=skill(y,p); rows.append({'weight':w,'skill':s,'brier':b})
        out[str(yr)]={'mse_solo':skill(y,pm)[0],'grid':rows,'best':max(rows,key=lambda q:q['skill'])}
        with open(OUT,'w') as f: json.dump(out,f,indent=2)
        print(out[str(yr)],flush=True)

if __name__=='__main__': main()
