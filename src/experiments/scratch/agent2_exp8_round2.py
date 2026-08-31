"""
agent2_exp8_round2.py — Round-2 INNER-ONLY screening (LightGBM, 2 seeds,
val = 2022 and 2023 only; 2024 is never touched here).

Builds on the round-1 winners:
  decomp v2 (off-by-one + tiny-cur_n fixes)  +  form ladder  +  historical
  count-conditional rates from the recovered hidden labels.
Also screens model capacity, since the feature set got much stronger and the
250-tree / depth-6 configuration was tuned for the old weak features.
"""
import sys, os, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from agent2_common import build_base_features, base_cat_cols, log
from agent2_asof_decomp import AsofDecomposer
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from agent2_exp7_extra import form_ladder, HistCondRates
from core.eval_utils import calc_brier_skill_score

SEEDS = [7, 123]
TGT = config.TARGET_COL
LGB_BASE = dict(n_estimators=250, num_leaves=45, learning_rate=0.05,
                min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                verbosity=-1, n_jobs=-1)

VARIANTS = {
    'w0_v1':        dict(dec='v1', blocks=[], lgb={}),
    'w1_v2':        dict(dec='v2', blocks=[], lgb={}),
    'w2_v2_fh':     dict(dec='v2', blocks=['form', 'hist'], lgb={}),
    'w3_cap500':    dict(dec='v2', blocks=['form', 'hist'],
                         lgb=dict(n_estimators=500, learning_rate=0.035)),
    'w4_leaves127': dict(dec='v2', blocks=['form', 'hist'],
                         lgb=dict(num_leaves=127, n_estimators=400, learning_rate=0.035,
                                  min_child_samples=100)),
    'w5_recency':   dict(dec='v2', blocks=['form', 'hist'], lgb={}, decay=0.9),
    'w6_cap1000':   dict(dec='v2', blocks=['form', 'hist'],
                         lgb=dict(n_estimators=1000, learning_rate=0.02)),
}


def run(val_seasons=(2022, 2023), seeds=SEEDS, variants=None):
    variants = variants or VARIANTS
    df = pd.read_csv(config.TRAIN_PATH)
    L = recover(df)
    res = {}
    for vs in val_seasons:
        tr = ((df.season >= 2019) & (df.season < vs)).values
        va = (df.season == vs).values
        df_tr = df[tr].copy(); df_val = df[va].copy()
        Xb_tr, Xb_val = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(Xb_tr)
        y_tr = df[TGT].values[tr]; y_val = df[TGT].values[va]

        cache = {}
        for tag, cls in [('v1', AsofDecomposer), ('v2', AsofDecomposer2)]:
            d = cls().fit(df_tr, vs)
            cache[tag] = (d.transform(df_tr), d.transform(df_val))
        hc = HistCondRates().fit(df_tr, L[tr], vs)
        H = (hc.transform(df_tr), hc.transform(df_val))
        log(f"val={vs}: feature blocks ready")

        for name, cfg in variants.items():
            A_tr, A_val = cache[cfg['dec']]
            parts_tr = [Xb_tr, A_tr]; parts_val = [Xb_val, A_val]
            if 'form' in cfg['blocks']:
                parts_tr.append(form_ladder(df_tr, A_tr))
                parts_val.append(form_ladder(df_val, A_val))
            if 'hist' in cfg['blocks']:
                parts_tr.append(H[0]); parts_val.append(H[1])
            X_tr = pd.concat(parts_tr, axis=1); X_val = pd.concat(parts_val, axis=1)
            cat_idx = [X_tr.columns.get_loc(c) for c in cc]
            sw = None
            if cfg.get('decay'):
                gap = (vs - 1 - df_tr['season']).clip(lower=0).values
                sw = np.power(cfg['decay'], gap); sw = sw / sw.mean()
            params = dict(LGB_BASE); params.update(cfg['lgb'])
            acc = np.zeros(len(X_val)); t0 = time.time()
            for seed in seeds:
                m = lgb.LGBMClassifier(random_state=seed, **params)
                m.fit(X_tr, y_tr, categorical_feature=cat_idx, sample_weight=sw)
                acc += m.predict_proba(X_val)[:, 1]
            p = acc / len(seeds)
            sk = calc_brier_skill_score(y_val, np.clip(p, 1e-6, 1 - 1e-6))
            best = max(((calc_brier_skill_score(y_val, np.clip(p + s, 1e-6, 1-1e-6))[0], round(s, 3))
                        for s in np.arange(-0.03, 0.021, 0.001)))
            res.setdefault(name, {})[vs] = dict(skill0=sk[0], best_skill=best[0],
                                                best_shift=best[1], nfeat=X_tr.shape[1])
            log(f"  [{name}] val={vs} skill={sk[0]:.2f} best={best[0]:.2f}@{best[1]:+.3f} "
                f"({X_tr.shape[1]}f, {time.time()-t0:.0f}s)")
            with open('~/LG_data/scratch/agent2_exp8.json', 'w') as f:
                json.dump(res, f, indent=2)
    print("\n=== ROUND-2 INNER-ONLY SUMMARY (mean of val 2022, 2023) ===")
    for name, d in res.items():
        if len(d) == len(val_seasons):
            print(f"  {name:<14} skill={np.mean([v['skill0'] for v in d.values()]):9.2f}  "
                  f"best={np.mean([v['best_skill'] for v in d.values()]):9.2f}")
    return res


if __name__ == '__main__':
    run()
