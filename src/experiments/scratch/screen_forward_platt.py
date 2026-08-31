"""Strict forward-only calibration of the established ensemble."""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = os.path.expanduser("~/LG_data")

def pred(vs):
    c=np.load(os.path.join(ROOT,"scratch/cache_final",f"final_val{vs}.npz"))
    m=np.load(os.path.join(ROOT,"scratch/multitask_aux_preds",f"val{vs}.npz"))["p"]
    g=.15*(c["p_lgb"]-.007)+.75*(c["p_cb"]-.008)+.10*(c["p_xgb"]-.006)
    p=np.clip(.5+1.1*((.68*g+.32*m)-.5)-.0045192086,1e-6,1-1e-6)
    return c["y"].astype(int),p

def sk(y,p):
    r=y.mean();return 100000*(1-np.mean((p-y)**2)/(r*(1-r)))

def main():
    D={s:pred(s) for s in (2022,2023,2024)}; out={}
    for vs in (2023,2024):
        yy=np.concatenate([D[s][0] for s in D if s<vs]); pp=np.concatenate([D[s][1] for s in D if s<vs])
        y,p=D[vs]; logit=np.log(pp/(1-pp)).reshape(-1,1); z=np.log(p/(1-p)).reshape(-1,1)
        m=LogisticRegression(C=1e6,max_iter=200).fit(logit,yy)
        q=m.predict_proba(z)[:,1]
        grid={}
        for w in (0,.1,.25,.5,.75,1):
            b=(1-w)*p+w*q;grid[str(w)]={"skill":float(sk(y,b)),"gain":float(sk(y,b)-sk(y,p))}
        out[str(vs)]={"coef":float(m.coef_[0,0]),"intercept":float(m.intercept_[0]),"base":float(sk(y,p)),"platt":float(sk(y,q)),"grid":grid}
        print(vs,out[str(vs)],flush=True)
    json.dump(out,open(os.path.join(ROOT,"scratch/forward_platt_results.json"),"w"),indent=2)
if __name__=="__main__":main()
