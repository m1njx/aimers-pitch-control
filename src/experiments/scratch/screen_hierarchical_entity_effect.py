"""Strict pre-season empirical-Bayes pitcher/batter effect screen.

No test data is read.  For season s, all entity rates are fit only on rows
with season < s, then blended with the saved exact-v33 temporal OOF prediction.
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/LG_data")

def score(y, p):
    br = float(np.mean((y-p)**2)); base = float(y.mean()*(1-y.mean()))
    return 100000*(1-br/base), br

def logit(x):
    x = np.clip(x, 1e-5, 1-1e-5); return np.log(x/(1-x))

def main():
    raw = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"))
    result = {}
    for season in (2022, 2023, 2024):
        tr = raw[raw.season < season]
        va = raw[raw.season == season].copy()
        g = tr.control_success.mean()
        # Shrink player rates strongly enough to make unseen/small-sample rows safe.
        tables = {}
        for col in ("pitcher_id", "batter_id"):
            t = tr.groupby(col).control_success.agg(["sum", "count"])
            for m in (75., 200., 500.):
                tables[(col,m)] = ((t["sum"] + m*g)/(t["count"]+m)).to_dict()
        base = np.load(os.path.join(ROOT, "scratch", "audit_v16_exact", f"val{season}.npz"))
        assert len(base["y"]) == len(va)
        p_gbdt = .15*base["p_lgb"] + .75*base["p_cb"] + .10*base["p_xgb"]
        p0 = np.clip(.5 + 1.10*((.65*p_gbdt + .35*base["p_mlp"])-.5)-.0045192086, 1e-6, 1-1e-6)
        rows=[]
        for m in (75.,200.,500.):
            pp = va.pitcher_id.map(tables[("pitcher_id",m)]).fillna(g).to_numpy()
            pb = va.batter_id.map(tables[("batter_id",m)]).fillna(g).to_numpy()
            # Additive log-odds player effects; no matchup table or validation labels.
            pe = 1/(1+np.exp(-(logit(g)+(logit(pp)-logit(g))+(logit(pb)-logit(g)))))
            for w in (0., .025, .05, .10, .15, .20):
                p=np.clip((1-w)*p0+w*pe,1e-6,1-1e-6); sk,br=score(va.control_success.to_numpy(),p)
                rows.append({"m":m,"weight":w,"skill":sk,"brier":br})
        result[str(season)]=rows
        best=max(rows,key=lambda x:x["skill"])
        print(f"season={season} baseline={rows[0]['skill']:.3f} best={best}",flush=True)
    path=os.path.join(ROOT,"scratch","hierarchical_entity_effect_results.json")
    with open(path,"w") as f: json.dump(result,f,indent=2)
    print(path,flush=True)
if __name__ == "__main__": main()
