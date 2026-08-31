"""
agent5_exp_shift_extrap.py

Orchestrator-directed candidate: reproduce agent2's inner-only shift
extrapolation procedure (agent2_findings.md section 12-1), but CORRECTLY on
the actually-deployed PURE asof_dec (v2, AsofDecomposer2) feature set instead
of the confounded "w2" (asof_dec+form_ladder+hist_cond) set that report 172
found section 12-1's numbers actually came from (cache_final/*.npz).

Procedure (code-level, matches agent2_exp2_drift_diag.py's per-fold shift
sweep, generalized from solo-LightGBM to the full 3-GBDT ensemble; matches
191/192's official-harness feature construction exactly):

  1. Get RAW (shift=0) 5-seed ensemble predictions for val=2021 (manual fold,
     train=2019-2020, matching 192's construction) and val=2022/2023/2024
     (official run_standard_sota_evaluation, matching 191's construction),
     using ONLY AsofDecomposer2 as extra_feature_fn (no form_ladder, no
     hist_cond).
  2. For each of 2021, 2022, 2023 (all NOT outer/2024), grid-sweep a single
     scalar ensemble-level shift to find that fold's skill-maximizing shift.
  3. Linearly extrapolate the 3-point (year, best_shift) trend to 2024.
  4. Apply that ONE extrapolated shift to the ALREADY-computed 2024 raw
     predictions -- single outer(2024) touch, no re-tuning after seeing it.

Reference (pure asof_dec, official harness, report 191/192):
  outer(2024) @ fixed shift -0.007 = 805.74 (the correct asof_dec-alone label
  per 172's audit -- NOT 812.21, which was actually the w2 confound).
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from preprocessing import PitchPreprocessor
from core.eval_utils import run_standard_sota_evaluation, calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
WEIGHTS = (0.15, 0.75, 0.10)
W_LGB, W_CB, W_XGB = WEIGHTS
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
ZERO_SHIFTS = {'lgb': 0.0, 'cb': 0.0, 'xgb': 0.0}
FIXED_SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
ASOF_DEC_OUTER_REF = 805.74  # correct label (191, pure asof_dec, fixed shift)


def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f); val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index; val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


def run_2021_raw(df_train, seeds):
    """Manual fold train=2019-2020, val=2021, RAW (shift=0) predictions, pure asof_dec v2.
    Mirrors 192_asof_dec_2021_holdout.py's construction exactly, minus the fixed shift."""
    VAL_SEASON = 2021
    df_tr_f = df_train[(df_train['season'] >= 2019) & (df_train['season'] < VAL_SEASON)].copy()
    df_val_f = df_train[df_train['season'] == VAL_SEASON].copy()
    as_of = VAL_SEASON - 1

    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=as_of, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)
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

    X_tr_f, X_val_f = add_asof_dec_features(df_tr_f, df_val_f, as_of, X_tr_f, X_val_f)

    y_tr_f = df_tr_f[config.TARGET_COL].values
    y_val_f = df_val_f[config.TARGET_COL].values
    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    bag_p_lgb = np.zeros(len(df_val_f)); bag_p_cb = np.zeros(len(df_val_f)); bag_p_xgb = np.zeros(len(df_val_f))
    for seed in seeds:
        lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                           **BASE_MP['lgb'], random_state=seed, verbosity=-1, n_jobs=-1)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
        bag_p_lgb += m.predict_proba(X_val_f)[:, 1]

        X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str); X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
        for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32); X_val_cb[c] = X_val_cb[c].astype(np.float32)
        cb_params = dict(random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1, **BASE_MP['cb'])
        m = CatBoostClassifier(**cb_params)
        m.fit(X_tr_cb, y_tr_f)
        bag_p_cb += m.predict_proba(X_val_cb)[:, 1]

        X_tr_x, X_val_x = X_tr_f.copy(), X_val_f.copy()
        for c in cat_cols:
            X_tr_x[c] = X_tr_x[c].astype('category').cat.codes.astype(np.float32)
            X_val_x[c] = X_val_x[c].astype('category').cat.codes.astype(np.float32)
        xgb_params = dict(random_state=seed, n_jobs=-1, eval_metric='logloss', **BASE_MP['xgb'])
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr_x.astype(np.float32), y_tr_f)
        bag_p_xgb += m.predict_proba(X_val_x.astype(np.float32))[:, 1]

    n = len(seeds)
    return dict(y=y_val_f, p_lgb=bag_p_lgb / n, p_cb=bag_p_cb / n, p_xgb=bag_p_xgb / n)


def best_shift_sweep(y, p_raw, grid=None):
    if grid is None:
        grid = np.arange(-0.08, 0.021, 0.001)
    best_s, best_sk = None, -1e18
    for s in grid:
        sk, *_ = calc_brier_skill_score(y, np.clip(p_raw + s, 1e-6, 1 - 1e-6))
        if sk > best_sk:
            best_sk, best_s = sk, s
    return float(best_s), float(best_sk)


if __name__ == '__main__':
    df_train = pd.read_csv(config.TRAIN_PATH)
    results = {}

    log("=== Step 1: RAW (shift=0) 5-seed official-harness run, pure asof_dec v2, folds 2022/2023/2024 ===")
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=ZERO_SHIFTS, random_seeds=FULL_SEEDS,
                                      extra_feature_fn=add_asof_dec_features)
    log(f"done in {(time.time()-t0)/60:.1f}min; raw(shift=0) fold skills="
        f"{[round(fd['skill_k'],2) for fd in r['fold_details']]}")

    for vs in [2022, 2023, 2024]:
        mask = (df_train['season'] == vs).values
        y = df_train.loc[mask, config.TARGET_COL].values
        p_raw = np.clip(W_LGB * r['oof_preds_lgb'][mask] + W_CB * r['oof_preds_cb'][mask]
                         + W_XGB * r['oof_preds_xgb'][mask], 1e-6, 1 - 1e-6)
        results[vs] = dict(y=y, p_raw=p_raw)

    log("=== Step 2: RAW (shift=0) 5-seed manual fold, val=2021 (train=2019-2020) ===")
    t0 = time.time()
    r2021 = run_2021_raw(df_train, FULL_SEEDS)
    log(f"done in {(time.time()-t0)/60:.1f}min")
    p_raw_2021 = np.clip(W_LGB * r2021['p_lgb'] + W_CB * r2021['p_cb'] + W_XGB * r2021['p_xgb'], 1e-6, 1 - 1e-6)
    results[2021] = dict(y=r2021['y'], p_raw=p_raw_2021)

    log("=== Step 3: per-year optimal single ensemble shift (2021, 2022, 2023 only -- NOT 2024) ===")
    trend_years = [2021, 2022, 2023]
    trend_shifts = []
    for vs in trend_years:
        s, sk = best_shift_sweep(results[vs]['y'], results[vs]['p_raw'])
        trend_shifts.append(s)
        # also report skill at the fixed -0.007-equivalent single shift for reference
        sk_fixed, *_ = calc_brier_skill_score(results[vs]['y'], np.clip(results[vs]['p_raw'] - 0.007, 1e-6, 1-1e-6))
        log(f"  val={vs}: best_shift={s:+.4f} skill_at_best={sk:.2f} skill_at_fixed(-0.007)={sk_fixed:.2f}")

    coef = np.polyfit(trend_years, trend_shifts, 1)
    extrap_shift_2024 = float(np.polyval(coef, 2024))
    log(f"linear fit: slope={coef[0]:.5f} intercept={coef[1]:.5f} -> extrapolated 2024 shift = {extrap_shift_2024:+.4f}")

    log("=== Step 4: SINGLE outer(2024) application of extrapolated shift (no re-tuning) ===")
    y2024 = results[2024]['y']; p_raw_2024 = results[2024]['p_raw']
    sk_extrap, raw_b, base_b, rr = calc_brier_skill_score(y2024, np.clip(p_raw_2024 + extrap_shift_2024, 1e-6, 1 - 1e-6))
    sk_fixed007, *_ = calc_brier_skill_score(y2024, np.clip(p_raw_2024 - 0.007, 1e-6, 1 - 1e-6))
    sk_zero, *_ = calc_brier_skill_score(y2024, np.clip(p_raw_2024, 1e-6, 1 - 1e-6))
    oracle_s, oracle_sk = best_shift_sweep(y2024, p_raw_2024)

    log(f"outer(2024) @ shift=0        skill = {sk_zero:.2f}")
    log(f"outer(2024) @ shift=-0.007   skill = {sk_fixed007:.2f}  (should be close to ref {ASOF_DEC_OUTER_REF})")
    log(f"outer(2024) @ shift=extrap({extrap_shift_2024:+.4f})  skill = {sk_extrap:.2f}  "
        f"(delta vs ref {ASOF_DEC_OUTER_REF} = {sk_extrap-ASOF_DEC_OUTER_REF:+.2f}; "
        f"delta vs this-run's own -0.007 baseline = {sk_extrap-sk_fixed007:+.2f})")
    log(f"outer(2024) @ oracle shift({oracle_s:+.4f}) skill = {oracle_sk:.2f}  (upper bound, cheating reference only)")

    out = dict(trend_years=trend_years, trend_shifts=trend_shifts, coef=list(coef),
               extrap_shift_2024=extrap_shift_2024,
               outer_2024=dict(skill_shift0=sk_zero, skill_fixed007=sk_fixed007, skill_extrap=sk_extrap,
                                oracle_shift=oracle_s, oracle_skill=oracle_sk),
               asof_dec_ref_outer=ASOF_DEC_OUTER_REF)
    with open('/tmp/agent5_shift_extrap_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    log("=== DONE ===")
