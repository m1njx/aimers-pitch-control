"""Legal compact Trackman-state candidate against the clean v23 OOF baseline.

Only historical Trackman seasons and fixed train-derived ID links are used.
The candidate replaces part of CatBoost only; LGB/XGB/MLP remain the exact
119-feature clean-baseline predictions saved by audit_v16_exact_cv.py.
"""
import json, os, sys, time
ROOT = os.path.expanduser("~/LG_data"); PKG = os.path.join(ROOT, "work", "submit_v16")
sys.path[:0] = [PKG, os.path.join(ROOT, "scratch")]
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from audit_v16_exact_cv import add_features

MAP = os.path.join(ROOT, "scratch", "agent2_map_pitcher.csv")
TM = os.path.join(ROOT, "open", "data", "trackman_history.csv")
SEEDS = (7, 123)
PHYS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed")

def state_table(asof):
    mp = pd.read_csv(MAP)
    mp = mp[(mp.margin >= .15) & (mp.n_a >= 20) & mp.mutual]
    hand = ((mp.hand_a == 1) & (mp.hand_b == "Left")) | ((mp.hand_a == 2) & (mp.hand_b == "Right"))
    lookup = dict(zip(mp.loc[hand, "b_id"].astype(int), mp.loc[hand, "a_id"].astype(int)))
    tm = pd.read_csv(TM, usecols=["season", "pitcher_trackman_id", *PHYS])
    tm = tm[tm.season <= asof].copy(); tm["pid"] = tm.pitcher_trackman_id.map(lookup); tm = tm.dropna(subset=["pid"])
    def agg(d, tag):
        x = d.groupby("pid")[list(PHYS)].agg(["mean", "std"]); x.columns = [f"{tag}_{a}_{b}" for a,b in x.columns]
        x[f"{tag}_n"] = d.groupby("pid").size(); return x
    recent = agg(tm[tm.season == asof], "tm_recent")
    hist = agg(tm[tm.season < asof], "tm_hist")
    out = hist.join(recent, how="outer")
    # Recent-minus-history is only meaningful with enough prior observations.
    for c in PHYS:
        out[f"tm_delta_{c}"] = out[f"tm_recent_{c}_mean"] - out[f"tm_hist_{c}_mean"]
    out["tm_recent_log_n"] = np.log1p(out["tm_recent_n"].fillna(0))
    out["tm_hist_log_n"] = np.log1p(out["tm_hist_n"].fillna(0))
    return out

def attach(X, raw, tab):
    q = tab.reindex(raw.pitcher_id.to_numpy()); q.index = X.index
    # Keep only robust mechanics changes, not a high-dimensional raw profile.
    keep = ["tm_recent_log_n", "tm_hist_log_n"] + [f"tm_delta_{c}" for c in PHYS]
    q = q.reindex(columns=keep)
    q["tm_state_mapped"] = q.notna().any(axis=1).astype(np.int8)
    for c in keep: q[c] = q[c].fillna(q[c].median()).fillna(0).astype(np.float32)
    # Interaction is pre-pitch: official as-of control state times past mechanics drift.
    q["tm_state_speed_x_control"] = q["tm_delta_rel_speed"] * X["cs_p_succ_eb"].to_numpy()
    q["tm_state_release_x_control"] = np.hypot(q["tm_delta_rel_height"], q["tm_delta_rel_side"]) * X["cs_p_succ_eb"].to_numpy()
    return pd.concat([X, q], axis=1)

def cb_frame(X, cats):
    z = X.copy()
    for c in cats: z[c] = z[c].astype(int).astype(str)
    for c in z.columns.difference(cats): z[c] = z[c].astype(np.float32)
    return z

def score(y, p):
    base = np.mean((y-y.mean())**2); return 10000*(1-np.mean((y-p)**2)/base)

def main():
    df = pd.read_csv(os.path.join(ROOT,"open","data","train.csv")); out={}
    for vs in (2022, 2023):
        tr=df[df.season<vs].copy(); va=df[df.season==vs].copy(); xtr,xva=add_features(tr,va,vs)
        tab=state_table(vs-1); xtr=attach(xtr,tr,tab); xva=attach(xva,va,tab)
        cats=[c for c in xtr if c in ("top_bottom","base_state","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","count_code","platoon_matchup","tkm_match","count_x_base")]
        a,b=cb_frame(xtr,cats),cb_frame(xva,cats); ps=[]
        for s in SEEDS:
            print(f"{vs} seed={s} features={a.shape[1]}",flush=True)
            m=CatBoostClassifier(iterations=300,learning_rate=.06,depth=6,cat_features=cats,random_seed=s,verbose=0)
            m.fit(a,tr.control_success); ps.append(m.predict_proba(b)[:,1])
        cand=np.mean(ps,0)
        z=np.load(os.path.join(ROOT,"scratch","audit_v16_exact",f"val{vs}.npz")); assert np.array_equal(z["y"],va.control_success.to_numpy())
        oldcb=z["p_cb"]; l=.15*np.clip(z["p_lgb"]-.007,1e-6,1-1e-6); x=.10*np.clip(z["p_xgb"]-.006,1e-6,1-1e-6); m=.32*z["p_mlp"]
        res={"coverage":float(xva.tm_state_mapped.mean()),"scores":{}}
        for w in (0,.15,.3,.5,.75,1):
            g=l+.75*((1-w)*np.clip(oldcb-.008,1e-6,1-1e-6)+w*np.clip(cand-.008,1e-6,1-1e-6))+x
            p=.68*g+m; res["scores"][str(w)]=score(z["y"],np.clip(p,1e-6,1-1e-6))
        out[str(vs)]=res; print(vs,res,flush=True)
    path=os.path.join(ROOT,"scratch","trackman_state_cb_results.json"); json.dump(out,open(path,"w"),indent=2); print(path,out)
if __name__=="__main__": main()
