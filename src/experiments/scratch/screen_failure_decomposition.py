"""Inner screen: reconstruct success as one minus four failure-subtype models."""
import json, sys, time
sys.path[:0] = ["~/LG_data/scratch", os.path.expanduser("~/LG_data")]
import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from agent2_common import build_base_features, base_cat_cols
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from core.eval_utils import calc_brier_skill_score

OUT="~/LG_data/scratch/failure_decomposition_results.json"
SEEDS=[7,123]

def log(x): print(f"[{time.strftime('%H:%M:%S')}] {x}",flush=True)
def model(seed):
    return lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=.05,
        min_child_samples=20,colsample_bytree=.7,subsample=.7,random_state=seed,
        verbosity=-1,n_jobs=-1)

def main():
    df=pd.read_csv(config.TRAIN_PATH); L=recover(df); res={}
    r=L.lab_reverse.to_numpy(); m=L.lab_middle.to_numpy(); y=df.control_success.to_numpy()
    known=np.isfinite(r)&np.isfinite(m)
    cats=np.full(len(df),-1,np.int8)
    cats[known&(y==1)]=0; cats[known&(y==0)&(r==1)&(m==0)]=1
    cats[known&(y==0)&(r==0)&(m==1)]=2; cats[known&(y==0)&(r==1)&(m==1)]=3
    cats[known&(y==0)&(r==0)&(m==0)]=4
    # Frozen-candidate outer confirmation after inner selected w_subtype=0.30.
    for vs in (2024,):
        tr=(df.season<vs).to_numpy(); va=(df.season==vs).to_numpy()
        dtr=df[tr].copy(); dva=df[va].copy()
        xt,xv=build_base_features(dtr,dva,vs-1,fix_index=True)
        dec=AsofDecomposer2().fit(dtr,vs); at,av=dec.transform(dtr),dec.transform(dva)
        xt=pd.concat([xt,at],axis=1); xv=pd.concat([xv,av],axis=1)
        ci=[xt.columns.get_loc(c) for c in base_cat_cols(xt)]
        direct=[]; subtype=[]
        for seed in SEEDS:
            md=model(seed).fit(xt,y[tr],categorical_feature=ci)
            direct.append(md.predict_proba(xv)[:,1])
            q=[]
            for k in (1,2,3,4):
                mk=model(seed).fit(xt,(cats[tr]==k).astype(np.int8),categorical_feature=ci)
                q.append(mk.predict_proba(xv)[:,1])
            subtype.append(np.clip(1-np.sum(q,axis=0),1e-6,1-1e-6))
            log(f"val={vs} seed={seed} done")
        pdirect=np.mean(direct,axis=0); psub=np.mean(subtype,axis=0); yy=y[va]
        rows={}
        for w in np.linspace(0,1,21):
            p=(1-w)*pdirect+w*psub
            sk,br,_,_=calc_brier_skill_score(yy,p)
            rows[str(round(float(w),2))]={"skill":sk,"brier":br}
        best=max(rows.items(),key=lambda z:z[1]["skill"])
        res[str(vs)]={"direct":calc_brier_skill_score(yy,pdirect)[0],
                      "subtype":calc_brier_skill_score(yy,psub)[0],"best":best,"grid":rows}
        json.dump(res,open(OUT,"w"),indent=2); log(f"RESULT {vs}: {res[str(vs)]}")
    print(json.dumps(res,indent=2))
if __name__=="__main__": main()
