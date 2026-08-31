"""Round-4 INNER-ONLY screening (val 2022, 2023 only).

s0 control        = decomp v2 + form + hist_cond   (round-2 winner)
s1 +log5          = Bill James log5 matchup odds combining the pitcher's and the
                    batter's current-season ability against an extrapolated
                    league level (train-only extrapolation, no test statistics)
s2 +era_ability   = league-drift-adjusted career ability (per-season rates minus
                    that season's league rate, then pooled) - a clean "true
                    skill" estimate to complement the noisy season-to-date rate
s3 +both
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.expanduser('~/LG_data')); sys.path.insert(0,'~/LG_data/scratch')
import numpy as np, pandas as pd, lightgbm as lgb, config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from agent2_exp7_extra import form_ladder, HistCondRates
from agent2_entity_rel import SeasonRelativeAbility
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123]; TGT=config.TARGET_COL

def log5(A, L):
    """P = (p*b/L) / (p*b/L + (1-p)(1-b)/(1-L)) — classic matchup odds formula."""
    p=np.clip(A['cs_p_succ_eb'].values,1e-3,1-1e-3)
    b=np.clip(A['cs_b_succ_eb'].values,1e-3,1-1e-3)
    num=p*b/L; den=num+(1-p)*(1-b)/(1-L)
    X=pd.DataFrame(index=A.index)
    X['log5']=num/den
    X['log5_odds']=np.log(np.clip(num/np.maximum(den-num,1e-9),1e-6,1e6))
    return X.astype(np.float32)

def run(val_seasons=(2022,2023)):
    df=pd.read_csv(config.TRAIN_PATH); L=recover(df); res={}
    for vs in val_seasons:
        tr=((df.season>=2019)&(df.season<vs)).values; va=(df.season==vs).values
        df_tr=df[tr].copy(); df_val=df[va].copy()
        Xb_tr,Xb_val=build_base_features(df_tr,df_val,vs-1,fix_index=True); cc=base_cat_cols(Xb_tr)
        d=AsofDecomposer2().fit(df_tr,vs); A_tr=d.transform(df_tr); A_val=d.transform(df_val)
        hc=HistCondRates().fit(df_tr,L[tr],vs)
        base_tr=pd.concat([Xb_tr,A_tr,form_ladder(df_tr,A_tr),hc.transform(df_tr)],axis=1)
        base_val=pd.concat([Xb_val,A_val,form_ladder(df_val,A_val),hc.transform(df_val)],axis=1)
        # league level: extrapolated from TRAIN seasons only, per row's own season
        lg=df_tr.groupby('season')[TGT].mean()
        yrs=lg.index.values.astype(float); co=np.polyfit(yrs,lg.values,1)
        Lrow_tr=np.array([lg.get(s, np.polyval(co,s)) for s in df_tr.season.values])
        Lrow_val=np.full(len(df_val), float(np.polyval(co,vs)))
        sr=SeasonRelativeAbility().fit(df_tr, vs)
        blocks={'log5':(log5(A_tr,Lrow_tr),log5(A_val,Lrow_val)),
                'era':(sr.transform(df_tr),sr.transform(df_val))}
        y_tr=df[TGT].values[tr]; y_val=df[TGT].values[va]
        log(f"val={vs} base X={base_tr.shape}  L_hat(val)={Lrow_val[0]:.4f} actual={y_val.mean():.4f}")
        for name,keys in [('s0_control',[]),('s1_log5',['log5']),('s2_era',['era']),
                          ('s3_both',['log5','era'])]:
            Xt=pd.concat([base_tr]+[blocks[k][0] for k in keys],axis=1)
            Xv=pd.concat([base_val]+[blocks[k][1] for k in keys],axis=1)
            ci=[Xt.columns.get_loc(c) for c in cc]
            acc=np.zeros(len(Xv)); t0=time.time()
            for seed in SEEDS:
                m=lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=0.05,
                    min_child_samples=20,colsample_bytree=0.7,subsample=0.7,
                    random_state=seed,verbosity=-1,n_jobs=-1)
                m.fit(Xt,y_tr,categorical_feature=ci); acc+=m.predict_proba(Xv)[:,1]
            p=acc/len(SEEDS); sk=calc_brier_skill_score(y_val,np.clip(p,1e-6,1-1e-6))
            res.setdefault(name,{})[vs]=sk[0]
            log(f"  [{name}] val={vs} skill={sk[0]:.2f} ({Xt.shape[1]}f, {time.time()-t0:.0f}s)")
        json.dump(res,open('~/LG_data/scratch/agent2_exp12.json','w'),indent=2)
    print("\n=== ROUND-4 INNER-ONLY SUMMARY ===")
    for n,dd in res.items():
        if len(dd)==len(val_seasons): print(f"  {n:<14} inner mean = {np.mean(list(dd.values())):9.2f}")
if __name__=='__main__': run()
