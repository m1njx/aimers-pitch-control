"""
agent2_common.py - Independent evaluation harness (Agent 2).

Reproduces the confirmed 843.69 GBDT classification baseline:
  lgb(colsample=0.7, subsample=0.7) 15% + catboost 75% + xgb 10%
  shifts (-0.007, -0.008, -0.006), seeds [7,123,2025,31415,8675309]

Allows injecting arbitrary extra features per fold via a callback, and
evaluating MULTIPLE variants inside the same fold/seed loop (preprocessing
is done once per fold, which is the expensive part).
"""
import sys, time, json, os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score

FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
PROBE_SEEDS = [11, 222, 3333, 44444, 555555]
W = (0.15, 0.75, 0.10)
SHIFTS = (-0.007, -0.008, -0.006)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_base_features(df_tr_f, df_val_f, as_of, fix_index=False):
    """Standard SSOT preprocessing + count_x_base.

    fix_index=False reproduces the SSOT engine EXACTLY, including its index
    misalignment bug: TrackmanFeatureBuilder.transform() uses pd.merge(), which
    resets the index to a RangeIndex, so assigning count_x_base from df_src
    (which carries the original .iloc positions as its index) silently produces
    all-NaN -> fillna(-1) for the VALIDATION split. Train split is unaffected
    because train rows are always the leading rows of the file.

    fix_index=True realigns the index so count_x_base is correct on both splits
    (which is what actually happens at submission time, where df_test is read
    fresh with a 0..n-1 index).
    """
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


def fit_predict_ensemble(X_tr, y_tr, X_val, cat_cols, seed, sample_weight=None):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05,
                      min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                      random_state=seed, verbosity=-1, n_jobs=-1)
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
    m = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                           random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
    m.fit(X_tr_cb, y_tr, sample_weight=sample_weight)
    p_cb = np.clip(m.predict_proba(X_val_cb)[:, 1] + SHIFTS[1], 1e-6, 1 - 1e-6)

    X_tr_x = X_tr.copy(); X_val_x = X_val.copy()
    # BUGFIX (agent2): the original code called .astype('category').cat.codes
    # SEPARATELY on train and val, so any categorical whose value SET differs
    # between the two splits gets a different integer for the same real
    # category. Measured on val=2024: 11 of 12 pitcher_team_id codes and 11 of
    # 12 batter_team_id codes were scrambled. The preprocessor already emits
    # consistent integer codes, so pass them through unchanged.
    for c in cat_cols:
        X_tr_x[c] = X_tr_x[c].astype(np.float32)
        X_val_x[c] = X_val_x[c].astype(np.float32)
    m = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                          colsample_bytree=0.8, subsample=0.8, random_state=seed,
                          n_jobs=-1, eval_metric='logloss')
    m.fit(X_tr_x.astype(np.float32), y_tr, sample_weight=sample_weight)
    p_xgb = np.clip(m.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS[2], 1e-6, 1 - 1e-6)

    return W[0] * p_lgb + W[1] * p_cb + W[2] * p_xgb


def run_variants(df_train, variant_fns, seeds=FULL_SEEDS, out_json=None, folds_subset=None):
    """
    variant_fns: dict name -> fn(df_tr_f, df_val_f, as_of, X_tr, X_val, cat_cols)
                                 -> (X_tr2, X_val2, cat_cols2)  (may return same objects)
    Returns dict name -> {fold_details, mean_skill, inner_mean, outer}
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
        X_tr_b, X_val_b = build_base_features(df_tr_f, df_val_f, as_of)
        cc_b = base_cat_cols(X_tr_b)
        y_tr = df_tr_f[config.TARGET_COL].values
        y_val = df_val_f[config.TARGET_COL].values
        log(f"fold val={fold.val_season}: base features ready in {time.time()-t0:.0f}s")

        for name, fn in variant_fns.items():
            X_tr, X_val, cc = fn(df_tr_f, df_val_f, as_of, X_tr_b.copy(), X_val_b.copy(), list(cc_b))
            bag = np.zeros(len(X_val))
            for seed in seeds:
                bag += fit_predict_ensemble(X_tr, y_tr, X_val, cc, seed)
            p = np.clip(bag / len(seeds), 1e-6, 1 - 1e-6)
            skill, raw, base, r = calc_brier_skill_score(y_val, p)
            results[name].append(dict(val_season=int(fold.val_season), skill=skill,
                                      raw_brier=raw, baseline=base, r=r))
            log(f"  [{name}] val={fold.val_season} skill={skill:.2f} raw_brier={raw:.6f}")

    summary = {}
    for name, fds in results.items():
        inner = [f['skill'] for f in fds if f['val_season'] in (2022, 2023)]
        outer = [f['skill'] for f in fds if f['val_season'] == 2024]
        summary[name] = dict(fold_details=fds,
                             mean_skill=float(np.mean([f['skill'] for f in fds])),
                             mean_raw_brier=float(np.mean([f['raw_brier'] for f in fds])),
                             inner_mean=float(np.mean(inner)) if inner else None,
                             outer=float(outer[0]) if outer else None)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(summary, f, indent=2)
    return summary


def print_summary(summary, ref=None):
    print("\n" + "=" * 78)
    print(f"{'variant':<34}{'3fold':>10}{'inner(22/23)':>14}{'outer(2024)':>13}")
    print("-" * 78)
    for name, s in summary.items():
        d = f"  ({s['mean_skill']-ref:+.2f})" if ref else ""
        print(f"{name:<34}{s['mean_skill']:>10.2f}{s['inner_mean'] or float('nan'):>14.2f}"
              f"{s['outer'] or float('nan'):>13.2f}{d}")
    print("=" * 78)
