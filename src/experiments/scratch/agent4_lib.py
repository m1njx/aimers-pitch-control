"""
agent4_lib.py - Screening harness for agent4 experiments.

Extends agent2_common.py's run_variants pattern (base features computed once
per fold, multiple variants reuse them) with:
  - LightGBM hyperparameter overrides (num_leaves sweep etc)
  - asof_dec (v2) features included as the control baseline for every variant
  - HistCondRates.recover() scoped strictly to the current fold's df_tr_f
    (fixes the fold-boundary lookahead bug documented in agent4_findings.md
    Discovery 0 -- agent2's original implementation called recover() on the
    FULL df_train before subsetting, which let fold1/fold2 peek past their
    own val season boundary).
"""
import sys, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_exp7_extra import form_ladder, HistCondRates
from agent2_recover_labels import recover

FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
SCREEN_SEEDS = [7, 123]
W = (0.15, 0.75, 0.10)
SHIFTS = (-0.007, -0.008, -0.006)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_base_features(df_tr_f, df_val_f, as_of, fix_index=True):
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=as_of, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)
    if fix_index:
        X_tr_f.index = df_tr_f.index
        X_val_f.index = df_val_f.index
    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        b_str = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                 (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                 (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        c_str = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                 df_src['strikes_before'].fillna(0).astype(int).astype(str))
        X_dst['count_x_base'] = (c_str + '_' + b_str)
    cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
    X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    return X_tr_f, X_val_f


def base_cat_cols(X):
    return [c for c in X.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']


def add_asof_dec(df_tr_f, df_val_f, val_season, X_tr_f, X_val_f):
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    A_tr = dec.transform(df_tr_f); A_tr.index = X_tr_f.index
    A_val = dec.transform(df_val_f); A_val.index = X_val_f.index
    return pd.concat([X_tr_f, A_tr], axis=1), pd.concat([X_val_f, A_val], axis=1), A_tr, A_val


def add_form_ladder(df_tr_f, df_val_f, A_tr, A_val, X_tr_f, X_val_f):
    F_tr = form_ladder(df_tr_f, A_tr); F_tr.index = X_tr_f.index
    F_val = form_ladder(df_val_f, A_val); F_val.index = X_val_f.index
    return pd.concat([X_tr_f, F_tr], axis=1), pd.concat([X_val_f, F_val], axis=1)


def add_hist_cond_FIXED(df_tr_f, df_val_f, val_season, X_tr_f, X_val_f, m=60.0):
    """FIXED version: recover() scoped to df_tr_f only (no fold-boundary lookahead)."""
    L_tr = recover(df_tr_f)  # <-- only this fold's train slice, not full df_train
    hc = HistCondRates(m=m).fit(df_tr_f, L_tr, val_season)
    H_tr = hc.transform(df_tr_f); H_tr.index = X_tr_f.index
    H_val = hc.transform(df_val_f); H_val.index = X_val_f.index
    return pd.concat([X_tr_f, H_tr], axis=1), pd.concat([X_val_f, H_val], axis=1)


def add_hist_cond_ORIG_BUGGY(df_train_full, df_tr_f, df_val_f, val_season, X_tr_f, X_val_f, m=60.0):
    """ORIGINAL (buggy) version: recover() on the FULL df_train, for comparison only."""
    L_full = recover(df_train_full)
    L_tr = L_full.loc[df_tr_f.index]
    hc = HistCondRates(m=m).fit(df_tr_f, L_tr, val_season)
    H_tr = hc.transform(df_tr_f); H_tr.index = X_tr_f.index
    H_val = hc.transform(df_val_f); H_val.index = X_val_f.index
    return pd.concat([X_tr_f, H_tr], axis=1), pd.concat([X_val_f, H_val], axis=1)


def fit_predict_ensemble(X_tr, y_tr, X_val, cat_cols, seed, lgb_overrides=None,
                          cb_overrides=None, xgb_overrides=None, sample_weight=None):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05,
                      min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                      random_state=seed, verbosity=-1, n_jobs=-1)
    if lgb_overrides:
        lgb_params.update(lgb_overrides)
        lgb_params['random_state'] = seed
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X_tr, y_tr, categorical_feature=cat_idx, sample_weight=sample_weight)
    p_lgb = np.clip(m.predict_proba(X_val)[:, 1] + SHIFTS[0], 1e-6, 1 - 1e-6)

    X_tr_cb = X_tr.copy(); X_val_cb = X_val.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)
    cb_params = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                     random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
    if cb_overrides:
        cb_params.update(cb_overrides)
        cb_params['random_seed'] = seed
        cb_params['cat_features'] = cat_cols
    m = CatBoostClassifier(**cb_params)
    m.fit(X_tr_cb, y_tr, sample_weight=sample_weight)
    p_cb = np.clip(m.predict_proba(X_val_cb)[:, 1] + SHIFTS[1], 1e-6, 1 - 1e-6)

    X_tr_x = X_tr.copy(); X_val_x = X_val.copy()
    for c in cat_cols:
        X_tr_x[c] = X_tr_x[c].astype(np.float32)
        X_val_x[c] = X_val_x[c].astype(np.float32)
    xgb_params = dict(n_estimators=250, max_depth=5, learning_rate=0.05,
                      colsample_bytree=0.8, subsample=0.8, random_state=seed,
                      n_jobs=-1, eval_metric='logloss')
    if xgb_overrides:
        xgb_params.update(xgb_overrides)
        xgb_params['random_state'] = seed
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X_tr_x.astype(np.float32), y_tr, sample_weight=sample_weight)
    p_xgb = np.clip(m.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS[2], 1e-6, 1 - 1e-6)

    return W[0] * p_lgb + W[1] * p_cb + W[2] * p_xgb


def run_variants(df_train, variant_fns, seeds=FULL_SEEDS, out_json=None, folds_subset=None):
    """
    variant_fns: dict name -> fn(df_tr_f, df_val_f, val_season, X_tr_b, X_val_b, cc_b, A_tr, A_val)
                                 -> (X_tr2, X_val2, cc2)
    Base features (asof_dec included) computed once per fold; each variant adds on top.
    """
    folds = get_cv_folds(df_train)
    if folds_subset is not None:
        folds = [folds[i] for i in folds_subset]
    results = {name: [] for name in variant_fns}

    for k, fold in enumerate(folds):
        t0 = time.time()
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
        as_of = fold.fold_max_season
        val_season = fold.val_season
        X_tr_b, X_val_b = build_base_features(df_tr_f, df_val_f, as_of)
        X_tr_b, X_val_b, A_tr, A_val = add_asof_dec(df_tr_f, df_val_f, val_season, X_tr_b, X_val_b)
        cc_b = base_cat_cols(X_tr_b)
        y_tr = df_tr_f[config.TARGET_COL].values
        y_val = df_val_f[config.TARGET_COL].values
        log(f"fold val={val_season}: base+asof_dec features ready in {time.time()-t0:.0f}s ({X_tr_b.shape[1]}f)")

        for name, fn in variant_fns.items():
            t1 = time.time()
            X_tr, X_val, cc, lgb_ov = fn(df_tr_f, df_val_f, val_season, X_tr_b.copy(), X_val_b.copy(),
                                          list(cc_b), A_tr, A_val)
            bag = np.zeros(len(X_val))
            for seed in seeds:
                bag += fit_predict_ensemble(X_tr, y_tr, X_val, cc, seed, lgb_overrides=lgb_ov)
            p = np.clip(bag / len(seeds), 1e-6, 1 - 1e-6)
            skill, raw, base, r = calc_brier_skill_score(y_val, p)
            results[name].append(dict(val_season=int(val_season), skill=skill,
                                      raw_brier=raw, baseline=base, r=r))
            log(f"  [{name}] val={val_season} skill={skill:.2f} raw_brier={raw:.6f} ({time.time()-t1:.0f}s)")
            if out_json:
                with open(out_json, 'w') as f:
                    json.dump(results, f, indent=2)

    summary = {}
    for name, fds in results.items():
        inner = [f['skill'] for f in fds if f['val_season'] in (2022, 2023)]
        outer = [f['skill'] for f in fds if f['val_season'] == 2024]
        summary[name] = dict(fold_details=fds,
                             mean_skill=float(np.mean([f['skill'] for f in fds])) if fds else None,
                             inner_mean=float(np.mean(inner)) if inner else None,
                             outer=float(outer[0]) if outer else None)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump({'results': results, 'summary': summary}, f, indent=2)
    return summary


def print_summary(summary, ref=None):
    print("\n" + "=" * 78)
    print(f"{'variant':<34}{'3fold':>10}{'inner(22/23)':>14}{'outer(2024)':>13}")
    print("-" * 78)
    for name, s in summary.items():
        d = f"  ({s['mean_skill']-ref:+.2f})" if (ref and s['mean_skill']) else ""
        ms = s['mean_skill'] if s['mean_skill'] is not None else float('nan')
        im = s['inner_mean'] if s['inner_mean'] is not None else float('nan')
        ot = s['outer'] if s['outer'] is not None else float('nan')
        print(f"{name:<34}{ms:>10.2f}{im:>14.2f}{ot:>13.2f}{d}")
    print("=" * 78)
