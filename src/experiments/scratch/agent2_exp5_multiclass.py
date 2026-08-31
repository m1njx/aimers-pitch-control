"""
agent2_exp5_multiclass.py — Exploit the RECOVERED per-pitch labels
(agent2_recover_labels.py) by training on the FULL 5-category outcome instead of
the binary coarsening, then reading off P(success).

Outcome categories (mutually exclusive, verified):
  0 success        52.37%
  1 reverse only   19.49%
  2 middle only    11.55%
  3 reverse+middle  3.41%
  4 none-of-these  13.18%

Rationale: control_success is a coarsening of a 5-way outcome. Each training row
carries strictly more label information in the 5-way form, so a multinomial fit
should estimate P(success | x) with lower variance than a binary fit - which
matters enormously here because the binary signal is tiny (AUC ~0.57).

Also tested: 3-class pitch-type group as an auxiliary target (fastball/breaking/
offspeed is recoverable for 100% of rows and explains 5.95e-4 of the target
variance = ~238 skill points if known).

LightGBM-only screening (the ensemble weight is 75% CatBoost, but we only need
the DIRECTION here and multiclass is 3-5x the cost).
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
from agent2_recover_labels import recover
from core.eval_utils import calc_brier_skill_score

OUT = '~/LG_data/scratch/cache_mc'
os.makedirs(OUT, exist_ok=True)
SEEDS = [7, 123]


def make_outcome5(L, y):
    """0=success 1=rev only 2=mid only 3=rev+mid 4=none"""
    r = L['lab_reverse'].values
    m = L['lab_middle'].values
    o = np.full(len(y), -1, dtype=np.int8)
    known = np.isfinite(r) & np.isfinite(m)
    o[known & (y == 1)] = 0
    o[known & (y == 0) & (r == 1) & (m == 0)] = 1
    o[known & (y == 0) & (r == 0) & (m == 1)] = 2
    o[known & (y == 0) & (r == 1) & (m == 1)] = 3
    o[known & (y == 0) & (r == 0) & (m == 0)] = 4
    return o


def make_pt3(L):
    fb = L['lab_fastball'].values
    br = L['lab_breaking'].values
    os_ = L['lab_offspeed'].values
    t = np.full(len(fb), -1, dtype=np.int8)
    t[fb == 1] = 0; t[br == 1] = 1; t[os_ == 1] = 2
    return t


def lgb_binary(X_tr, y_tr, X_val, cat_idx, seed, n_est=250):
    m = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=45, learning_rate=0.05,
                           min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                           random_state=seed, verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, categorical_feature=cat_idx)
    return m.predict_proba(X_val)[:, 1]


def lgb_multi(X_tr, y_tr, X_val, cat_idx, seed, n_class, n_est=250, success_idx=0):
    m = lgb.LGBMClassifier(n_estimators=n_est, num_leaves=45, learning_rate=0.05,
                           min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                           random_state=seed, verbosity=-1, n_jobs=-1,
                           objective='multiclass', num_class=n_class)
    m.fit(X_tr, y_tr, categorical_feature=cat_idx)
    return m.predict_proba(X_val)[:, success_idx]


def run(val_seasons=(2022, 2023, 2024), seeds=SEEDS):
    df = pd.read_csv(config.TRAIN_PATH)
    log("recovering hidden labels ...")
    L = recover(df)
    y_all = df[config.TARGET_COL].values
    out5 = make_outcome5(L, y_all)
    pt3 = make_pt3(L)
    log(f"outcome5 dist: {np.bincount(out5[out5>=0], minlength=5) / (out5>=0).sum()}")

    results = {}
    for vs in val_seasons:
        tr = ((df.season >= 2019) & (df.season < vs)).values
        va = (df.season == vs).values
        df_tr = df[tr].copy(); df_val = df[va].copy()
        X_tr, X_val = build_base_features(df_tr, df_val, vs - 1, fix_index=True)
        cc = base_cat_cols(X_tr)
        dec = AsofDecomposer().fit(df_tr, vs)
        X_tr = pd.concat([X_tr, dec.transform(df_tr)], axis=1)
        X_val = pd.concat([X_val, dec.transform(df_val)], axis=1)
        cat_idx = [X_tr.columns.get_loc(c) for c in cc]
        y_tr = y_all[tr]; y_val = y_all[va]
        o_tr = out5[tr]; t_tr = pt3[tr]
        ok5 = o_tr >= 0; ok3 = t_tr >= 0
        log(f"val={vs}: X={X_tr.shape}, 5class usable={ok5.mean():.4f}")

        preds = {}
        for name in ['binary', 'mc5', 'mc5_deep']:
            acc = np.zeros(len(X_val))
            for seed in seeds:
                t0 = time.time()
                if name == 'binary':
                    p = lgb_binary(X_tr, y_tr, X_val, cat_idx, seed)
                elif name == 'mc5':
                    p = lgb_multi(X_tr.loc[ok5], o_tr[ok5], X_val, cat_idx, seed, 5)
                elif name == 'mc5_deep':
                    p = lgb_multi(X_tr.loc[ok5], o_tr[ok5], X_val, cat_idx, seed, 5, n_est=400)
                acc += p
                log(f"  {name} seed={seed} done in {time.time()-t0:.0f}s")
            preds[name] = acc / len(seeds)

        res = {}
        for name, p in preds.items():
            best = max(((calc_brier_skill_score(y_val, np.clip(p + s, 1e-6, 1-1e-6))[0], s)
                        for s in np.arange(-0.05, 0.031, 0.001)))
            sk0 = calc_brier_skill_score(y_val, np.clip(p, 1e-6, 1-1e-6))
            res[name] = dict(skill_noshift=sk0[0], raw_brier=sk0[1], p_mean=float(p.mean()),
                             p_std=float(p.std()), best_skill=best[0], best_shift=float(best[1]))
            log(f"  [{name}] val={vs} skill(no shift)={sk0[0]:.2f} best={best[0]:.2f} @shift={best[1]:+.3f}")
        results[vs] = res
        np.savez_compressed(f'{OUT}/mc_val{vs}.npz', y=y_val.astype(np.int8),
                            **{k: v.astype(np.float32) for k, v in preds.items()})
        with open(f'{OUT}/summary.json', 'w') as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == '__main__':
    vs = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [2022, 2023, 2024]
    r = run(vs)
    print(json.dumps(r, indent=2))
