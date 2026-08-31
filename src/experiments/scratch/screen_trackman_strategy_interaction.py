"""Strict screen: historical pitcher-batter Trackman strategy lookups.

All lookup tables are built from Trackman seasons before the held-out season.
At scoring time each row only uses its own pitcher, batter, count and hand.
The candidate replaces only the CatBoost branch of the exact v33 OOF blend.
"""
import gc, json, os, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = os.path.expanduser("~/LG_data")
sys.path[:0] = [os.path.join(ROOT, "scratch"), os.path.join(ROOT, "track_claude_z"), ROOT]
from audit_v16_exact_cv import add_features
from agent3_tkm_sit import attach as attach_pitcher
from agent3_tkm_bat import attach_bat, attach_matchup

CACHE = os.path.join(ROOT, "scratch", "agent3_cache")
SEASONS = [2022, 2023]
ASOF = {2022: 2021, 2023: 2022}
CATS = ["top_bottom", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "count_code", "platoon_matchup", "tkm_match", "count_x_base"]

def skill(y, p):
    br = float(np.mean((y-p)**2)); base = float(y.mean()*(1-y.mean()))
    return 100000*(1-br/base), br

def cb_frame(x):
    z=x.copy(); cats=[c for c in CATS if c in z]
    for c in cats: z[c]=pd.to_numeric(z[c],errors="coerce").fillna(-1).astype(int).astype(str)
    for c in z.columns:
        if c not in cats: z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0).astype(np.float32)
    return z,cats

def main():
    raw=pd.read_csv(os.path.join(ROOT,"open","data","train.csv")); out={}
    for season in SEASONS:
        tr=raw[raw.season<season].copy(); va=raw[raw.season==season].copy()
        xtr,xva=add_features(tr,va,season)
        a=ASOF[season]
        fp=pd.read_parquet(os.path.join(CACHE,f"tkm_sit_{a}.parquet"))
        fb=pd.read_parquet(os.path.join(CACHE,f"tkm_bsit_{a}.parquet"))
        fm=pd.read_parquet(os.path.join(CACHE,f"tkm_mu_{a}.parquet"))
        xtr=attach_pitcher(fp,tr,xtr,list(fp.columns)); xva=attach_pitcher(fp,va,xva,list(fp.columns))
        xtr=attach_bat(fb,tr,xtr); xva=attach_bat(fb,va,xva)
        xtr=attach_matchup(fm,tr,xtr); xva=attach_matchup(fm,va,xva)
        xtr,cats=cb_frame(xtr); xva,_=cb_frame(xva)
        p=np.zeros(len(va))
        for seed in [7,2025]:
            m=CatBoostClassifier(iterations=300,learning_rate=.06,depth=6,cat_features=cats,random_seed=seed,verbose=0)
            m.fit(xtr,tr.control_success.to_numpy()); p+=m.predict_proba(xva)[:,1]/2
        base=np.load(os.path.join(ROOT,"scratch","audit_v16_exact",f"val{season}.npz"))
        pg=.15*base["p_lgb"]+.75*base["p_cb"]+.10*base["p_xgb"]
        rows=[]
        for w in [0,.05,.10,.15,.20,.30,.50,1.0]:
            # replace part of the CB component, retain exact other OOF branches.
            g=.15*base["p_lgb"]+.75*((1-w)*base["p_cb"]+w*p)+.10*base["p_xgb"]
            pred=np.clip(.5+1.10*((.65*g+.35*base["p_mlp"])-.5)-.0045192086,1e-6,1-1e-6)
            sk,br=skill(va.control_success.to_numpy(),pred); rows.append({"cb_replace":w,"skill":sk,"brier":br})
        out[str(season)]={"features":int(xtr.shape[1]),"best":max(rows,key=lambda r:r["skill"]),"grid":rows}
        print(season,out[str(season)],flush=True)
        del tr,va,xtr,xva,p; gc.collect()
    with open(os.path.join(ROOT,"scratch","trackman_strategy_interaction_results.json"),"w") as f: json.dump(out,f,indent=2)

if __name__=="__main__": main()
