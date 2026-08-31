"""Round-3 INNER-ONLY screening: training-window / objective choices on the new
feature set. val = 2022, 2023 only."""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.expanduser('~/LG_data')); sys.path.insert(0,'~/LG_data/scratch')
import numpy as np, pandas as pd, lightgbm as lgb, config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from agent2_exp7_extra import form_ladder, HistCondRates
from core.eval_utils import calc_brier_skill_score
SEEDS=[7,123]; TGT=config.TARGET_COL
CFG={'r0_control':{}, 'r1_decay07':{'decay':0.7}, 'r2_decay05':{'decay':0.5},
     'r3_last3':{'min_season_off':3}, 'r4_l2':{'obj':'regression'},
     'r5_multi_eb':{'multi_eb':True}}
def run(val_seasons=(2022,2023)):
    df=pd.read_csv(config.TRAIN_PATH); L=recover(df); res={}
    for vs in val_seasons:
        tr=((df.season>=2019)&(df.season<vs)).values; va=(df.season==vs).values
        df_tr=df[tr].copy(); df_val=df[va].copy()
        Xb_tr,Xb_val=build_base_features(df_tr,df_val,vs-1,fix_index=True); cc=base_cat_cols(Xb_tr)
        parts_tr=[Xb_tr]; parts_val=[Xb_val]
        A_tr=A_val=None
        for m in [150.0]:
            d=AsofDecomposer2(eb_m=m).fit(df_tr,vs); A_tr=d.transform(df_tr); A_val=d.transform(df_val)
        hc=HistCondRates().fit(df_tr,L[tr],vs)
        X_tr=pd.concat([Xb_tr,A_tr,form_ladder(df_tr,A_tr),hc.transform(df_tr)],axis=1)
        X_val=pd.concat([Xb_val,A_val,form_ladder(df_val,A_val),hc.transform(df_val)],axis=1)
        extra={}
        for m in [40.0,600.0]:
            d=AsofDecomposer2(eb_m=m).fit(df_tr,vs)
            for tag,src,dst in [('tr',df_tr,'tr'),('va',df_val,'va')]:
                T=d.transform(src)[['cs_p_succ_eb','cs_b_succ_eb']].add_suffix(f'_m{int(m)}')
                extra.setdefault(tag,[]).append(T)
        y_tr=df[TGT].values[tr]; y_val=df[TGT].values[va]; season_tr=df_tr['season'].values
        cat_idx=[X_tr.columns.get_loc(c) for c in cc]
        log(f"val={vs} X={X_tr.shape}")
        for name,cfg in CFG.items():
            Xt,Xv=X_tr,X_val; ci=cat_idx; yt=y_tr; sw=None; msk=np.ones(len(Xt),bool)
            if cfg.get('multi_eb'):
                Xt=pd.concat([X_tr]+extra['tr'],axis=1); Xv=pd.concat([X_val]+extra['va'],axis=1)
                ci=[Xt.columns.get_loc(c) for c in cc]
            if cfg.get('decay'):
                gap=np.clip(vs-1-season_tr,0,None); sw=np.power(cfg['decay'],gap); sw=sw/sw.mean()
            if cfg.get('min_season_off'):
                msk=season_tr>=vs-cfg['min_season_off']
            acc=np.zeros(len(Xv)); t0=time.time()
            for seed in SEEDS:
                if cfg.get('obj')=='regression':
                    m_=lgb.LGBMRegressor(objective='regression',n_estimators=250,num_leaves=45,
                        learning_rate=0.05,min_child_samples=20,colsample_bytree=0.7,subsample=0.7,
                        random_state=seed,verbosity=-1,n_jobs=-1)
                    m_.fit(Xt[msk],yt[msk],categorical_feature=ci,
                           sample_weight=None if sw is None else sw[msk])
                    acc+=np.clip(m_.predict(Xv),0.001,0.999)
                else:
                    m_=lgb.LGBMClassifier(n_estimators=250,num_leaves=45,learning_rate=0.05,
                        min_child_samples=20,colsample_bytree=0.7,subsample=0.7,
                        random_state=seed,verbosity=-1,n_jobs=-1)
                    m_.fit(Xt[msk],yt[msk],categorical_feature=ci,
                           sample_weight=None if sw is None else sw[msk])
                    acc+=m_.predict_proba(Xv)[:,1]
            p=acc/len(SEEDS)
            sk=calc_brier_skill_score(y_val,np.clip(p,1e-6,1-1e-6))
            res.setdefault(name,{})[vs]=sk[0]
            log(f"  [{name}] val={vs} skill={sk[0]:.2f} ({Xt.shape[1]}f, n={msk.sum():,}, {time.time()-t0:.0f}s)")
        json.dump(res,open('~/LG_data/scratch/agent2_exp11.json','w'),indent=2)
    print("\n=== ROUND-3 INNER-ONLY SUMMARY ===")
    for n,d in res.items():
        if len(d)==len(val_seasons): print(f"  {n:<14} inner mean = {np.mean(list(d.values())):9.2f}")
if __name__=='__main__': run()
