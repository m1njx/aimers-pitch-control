"""Forward-season screen: CatBoost success vs decomposed failure probabilities.

All auxiliary targets are recovered exclusively from each labelled training row.
At inference every model consumes only the current row's allowed pre-pitch fields.
"""
import gc, json, os, sys, time

ROOT = os.path.expanduser("~/LG_data")
sys.path[:0] = [os.path.join(ROOT, "track_claude_z"), os.path.join(ROOT, "scratch"), ROOT]

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from eval_utils import calc_brier_skill_score

OUT = os.path.join(ROOT, "scratch", "failure_decomposition_catboost_seed123_results.json")
PRED_DIR = os.path.join(ROOT, "scratch", "failure_decomposition_catboost_preds")
SEEDS = [123]
BASE_CATS = (config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS +
             [config.TRACKMAN_MATCH_FLAG_COL, "count_x_base"])

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def frame(raw_tr, raw_va, as_of, val_season):
    prep = PitchPreprocessor().fit(raw_tr, as_of_season=as_of, is_final=False)
    a, b = prep.transform(raw_tr), prep.transform(raw_va)
    def count_base(raw):
        bases = ((raw.runner_on_1b.fillna(0)>0).astype(int).astype(str)+"_"+
                 (raw.runner_on_2b.fillna(0)>0).astype(int).astype(str)+"_"+
                 (raw.runner_on_3b.fillna(0)>0).astype(int).astype(str))
        return (raw.balls_before.fillna(0).astype(int).astype(str)+"_"+
                raw.strikes_before.fillna(0).astype(int).astype(str)+"_"+bases)
    mp = {v:i for i,v in enumerate(count_base(raw_tr).unique())}
    a["count_x_base"] = count_base(raw_tr).map(mp).fillna(-1).astype(np.int32)
    b["count_x_base"] = count_base(raw_va).map(mp).fillna(-1).astype(np.int32)
    dec = AsofDecomposer2().fit(raw_tr, val_season=val_season)
    da, db = dec.transform(raw_tr), dec.transform(raw_va)
    da.index, db.index = a.index, b.index
    return pd.concat([a,da],axis=1), pd.concat([b,db],axis=1)

def prepare(Xtr, Xva):
    tr, va = Xtr.copy(), Xva.copy()
    cats = [c for c in BASE_CATS if c in tr.columns]
    for c in cats:
        tr[c] = tr[c].fillna(-1).astype(str); va[c] = va[c].fillna(-1).astype(str)
    for c in tr.columns:
        if c not in cats:
            tr[c] = tr[c].astype(np.float32); va[c] = va[c].astype(np.float32)
    return tr, va, cats

def predict(Xtr, Xva, target, seed):
    m = CatBoostClassifier(iterations=250, depth=6, learning_rate=.06,
        l2_leaf_reg=10, loss_function="Logloss", verbose=0, random_seed=seed,
        cat_features=[c for c in BASE_CATS if c in Xtr.columns], thread_count=-1,
        allow_writing_files=False)
    m.fit(Xtr, target)
    return m.predict_proba(Xva)[:,1]

def main():
    df = pd.read_csv(config.TRAIN_PATH)
    labels = recover(df)
    r = labels.lab_reverse.to_numpy(); mid = labels.lab_middle.to_numpy()
    y = df.control_success.to_numpy().astype(np.int8)
    known = np.isfinite(r) & np.isfinite(mid)
    category = np.full(len(df), -1, np.int8)
    category[known & (y==1)] = 0
    category[known & (y==0) & (r==1) & (mid==0)] = 1
    category[known & (y==0) & (r==0) & (mid==1)] = 2
    category[known & (y==0) & (r==1) & (mid==1)] = 3
    category[known & (y==0) & (r==0) & (mid==0)] = 4
    os.makedirs(PRED_DIR, exist_ok=True)
    result = {}
    # Independent seed confirmation over both inner folds and the untouched outer fold.
    for fold in get_cv_folds(df):
        ti, vi = fold.train_idx, fold.val_idx
        raw_tr, raw_va = df.iloc[ti].copy(), df.iloc[vi].copy()
        log(f"build fold={fold.val_season}")
        Xtr, Xva = frame(raw_tr, raw_va, fold.fold_max_season, fold.val_season)
        Xtr, Xva, _ = prepare(Xtr, Xva)
        direct, subtype = [], []
        for seed in SEEDS:
            log(f"fold={fold.val_season} seed={seed} direct")
            direct.append(predict(Xtr, Xva, y[ti], seed))
            qs=[]
            for k in (1,2,3,4):
                log(f"fold={fold.val_season} seed={seed} subtype={k}")
                qs.append(predict(Xtr, Xva, (category[ti]==k).astype(np.int8), seed+100*k))
            subtype.append(np.clip(1-np.sum(qs,axis=0),1e-6,1-1e-6))
        pdirect=np.mean(direct,axis=0); psub=np.mean(subtype,axis=0); yy=y[vi]
        np.savez_compressed(os.path.join(PRED_DIR, f"val{fold.val_season}_seed123.npz"),
                            y=yy, direct=pdirect, subtype=psub, row_idx=vi)
        grid={}
        for w in np.linspace(0,0.6,13):
            p=(1-w)*pdirect+w*psub
            sk,br,_,_=calc_brier_skill_score(yy,p)
            grid[f"{w:.2f}"]={"skill":float(sk),"brier":float(br)}
        best=max(grid.items(), key=lambda z:z[1]["skill"])
        result[str(fold.val_season)]={"direct":float(calc_brier_skill_score(yy,pdirect)[0]),
          "subtype":float(calc_brier_skill_score(yy,psub)[0]),"best":best,"grid":grid}
        with open(OUT,"w") as f: json.dump(result,f,indent=2)
        log(f"RESULT {fold.val_season}: {result[str(fold.val_season)]}")
        del Xtr,Xva,raw_tr,raw_va; gc.collect()
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
