"""Prospective OOF meta-calibration of the production GBDT prediction.

For validation season S, the stacker is trained only on genuine OOF predictions
from seasons earlier than S. Inference features are strictly current-row fields.
"""
import json, os, sys, time
ROOT=os.path.expanduser("~/LG_data")
sys.path[:0]=[ROOT,os.path.join(ROOT,"core")]
import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from eval_utils import calc_brier_skill_score

OUT=os.path.join(ROOT,"scratch","oof_residual_stacker_results.json")
SEEDS=[7,123,2025]
DROP={"row_id","season","control_success","pitcher_id","batter_id"}

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)

def base_and_components(year):
    z=np.load(os.path.join(ROOT,"scratch","cache_final",f"final_val{year}.npz"))
    p=.15*(z["p_lgb"]-.007)+.75*(z["p_cb"]-.008)+.10*(z["p_xgb"]-.006)
    return np.clip(p,1e-5,1-1e-5),z["p_lgb"],z["p_cb"],z["p_xgb"],z["y"]

def make_frames(df, years):
    frames=[]
    for yr in years:
        raw=df[df.season==yr].copy().reset_index(drop=True)
        p,pl,pc,px,y=base_and_components(yr)
        assert len(raw)==len(y) and np.array_equal(raw.control_success.to_numpy(),y)
        x=raw[[c for c in raw.columns if c not in DROP]].copy()
        x["base_logit"]=np.log(p/(1-p)); x["base_p"]=p
        x["model_lgb"]=pl; x["model_cb"]=pc; x["model_xgb"]=px
        frames.append((yr,x,y,p))
    return frames

def encode(train,valid):
    a,b=train.copy(),valid.copy(); cats=[]
    for c in a.columns:
        if a[c].dtype==object:
            vals=pd.Index(a[c].fillna("__NA__").astype(str).unique())
            mp={v:i for i,v in enumerate(vals)}
            a[c]=a[c].fillna("__NA__").astype(str).map(mp).fillna(-1).astype("int32")
            b[c]=b[c].fillna("__NA__").astype(str).map(mp).fillna(-1).astype("int32")
            cats.append(c)
        else:
            a[c]=pd.to_numeric(a[c],errors="coerce").astype("float32")
            b[c]=pd.to_numeric(b[c],errors="coerce").astype("float32")
    return a,b,cats

def main():
    df=pd.read_csv(config.TRAIN_PATH)
    fs=make_frames(df,[2021,2022,2023,2024]); results={}
    for j in range(1,len(fs)):
        yr,xv,yv,pbase=fs[j]
        xt=pd.concat([q[1] for q in fs[:j]],ignore_index=True)
        yt=np.concatenate([q[2] for q in fs[:j]])
        xt,xv2,cats=encode(xt,xv)
        preds=[]
        for seed in SEEDS:
            m=lgb.LGBMClassifier(n_estimators=180,num_leaves=15,max_depth=5,
                learning_rate=.025,min_child_samples=500,min_split_gain=1e-5,
                reg_lambda=20,reg_alpha=2,colsample_bytree=.75,subsample=.8,
                random_state=seed,n_jobs=-1,verbosity=-1)
            m.fit(xt,yt,categorical_feature=cats)
            preds.append(m.predict_proba(xv2)[:,1]); log(f"year={yr} seed={seed}")
        pm=np.mean(preds,axis=0); base_score=calc_brier_skill_score(yv,pbase)[0]
        grid={}
        for w in np.linspace(0,1,21):
            p=(1-w)*pbase+w*pm
            sk,br,_,_=calc_brier_skill_score(yv,p)
            grid[f"{w:.2f}"]={"skill":float(sk),"gain":float(sk-base_score),"brier":float(br)}
        results[str(yr)]={"base":float(base_score),"meta":float(calc_brier_skill_score(yv,pm)[0]),
                          "best":max(grid.items(),key=lambda z:z[1]["skill"]),"grid":grid}
        json.dump(results,open(OUT,"w"),indent=2); log(f"RESULT {yr}: {results[str(yr)]}")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
