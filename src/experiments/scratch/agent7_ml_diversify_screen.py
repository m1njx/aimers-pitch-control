"""
agent7_ml_diversify_screen.py

Step 2 (ML diversification) of agent7's task: screen NEW ensemble-member
architectures (never yet tried on top of the asof_dec feature set) against the
asof_dec GBDT reference, INNER-ONLY (val=2022, val=2023). 2024 (outer) is never
built or touched here.

Motivation: agent3_findings.md (pre-asof_dec era) found that a pure Ridge model
has a very different error structure from GBDT -- much stronger on fold=2023 in
particular -- suggesting value as a blending diversifier, though it was never
retested on the (much stronger) asof_dec feature set. This script retests that
idea plus one more model family (ExtraTrees) that hasn't been used as an
ensemble member anywhere in this project.

Row-independence discipline (report 203 principle): every encoding here is
fit on the TRAIN split only and applied to val via a fixed lookup/transform,
never recomputed from whichever values happen to be present in val:
  - OneHotEncoder(handle_unknown='ignore'): categories_ fixed from fit(train).
  - StandardScaler: mean/std fixed from fit(train).
  - ExtraTrees / GBDT: reuse the existing PitchPreprocessor integer category
    codes (already a fixed train-only dict, see preprocessing.py / dl_common's
    count_x_base pattern) -- NOT pandas .astype('category').cat.codes.
  - XGBoost reference (if used) uses the value-1 fixed arithmetic transform,
    matching the report-203 fix in core/eval_utils.py.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2

SEED = 7
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
BASE_MP = {
    'lgb': dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                colsample_bytree=0.7, subsample=0.7, random_state=SEED, verbosity=-1, n_jobs=-1),
    'cb': dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
               random_seed=SEED, verbose=0, thread_count=-1),
    'xgb': dict(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8,
                subsample=0.8, random_state=SEED, n_jobs=-1, eval_metric='logloss'),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f)
    val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index
    val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


def build_asofdec_fold_frames(df_train, fold):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
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
    X_tr_f, X_val_f = add_asof_dec_features(df_tr_f, df_val_f, fold.fold_max_season, X_tr_f, X_val_f)
    y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)
    y_val_f = df_val_f[config.TARGET_COL].values.astype(np.float32)
    return X_tr_f, X_val_f, y_tr_f, y_val_f


def get_cat_cols(X_tr_f):
    return [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']


def fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f, cat_cols):
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    m_lgb = lgb.LGBMClassifier(**BASE_MP['lgb'])
    m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
    p_lgb = np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + SHIFTS['lgb'], 1e-6, 1 - 1e-6)

    X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)
    m_cb = CatBoostClassifier(cat_features=cat_cols, **BASE_MP['cb'])
    m_cb.fit(X_tr_cb, y_tr_f)
    p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + SHIFTS['cb'], 1e-6, 1 - 1e-6)

    # row-independent XGB encoding: fixed value-1 arithmetic (report 203 fix),
    # NOT .astype('category').cat.codes (which is split-dependent).
    X_tr_x, X_val_x = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        if c == 'count_x_base':
            X_tr_x[c] = X_tr_x[c].astype(np.float32)
            X_val_x[c] = X_val_x[c].astype(np.float32)
        else:
            X_tr_x[c] = (X_tr_x[c] - 1).astype(np.float32)
            X_val_x[c] = (X_val_x[c] - 1).astype(np.float32)
    m_xgb = xgb.XGBClassifier(**BASE_MP['xgb'])
    m_xgb.fit(X_tr_x.astype(np.float32), y_tr_f)
    p_xgb = np.clip(m_xgb.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS['xgb'], 1e-6, 1 - 1e-6)

    w_lgb, w_cb, w_xgb = WEIGHTS
    p_ens = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
    return p_ens


def fit_predict_logistic(X_tr_f, X_val_f, y_tr_f, cat_cols):
    num_cols = [c for c in X_tr_f.columns if c not in cat_cols]
    # Fixed train-only fit for both encoders -- row-independent by construction
    # (categories_/mean_/scale_ are frozen before any val row is transformed).
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    Xtr_cat = ohe.fit_transform(X_tr_f[cat_cols].astype(str))
    Xval_cat = ohe.transform(X_val_f[cat_cols].astype(str))

    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(X_tr_f[num_cols].fillna(0.0).astype(np.float32))
    Xval_num = scaler.transform(X_val_f[num_cols].fillna(0.0).astype(np.float32))

    from scipy.sparse import hstack, csr_matrix
    Xtr = hstack([csr_matrix(Xtr_num), Xtr_cat]).tocsr()
    Xval = hstack([csr_matrix(Xval_num), Xval_cat]).tocsr()

    m = LogisticRegression(C=1.0, max_iter=300, solver='lbfgs', n_jobs=-1)
    m.fit(Xtr, y_tr_f)
    p = np.clip(m.predict_proba(Xval)[:, 1], 1e-6, 1 - 1e-6)
    return p


def fit_predict_extratrees(X_tr_f, X_val_f, y_tr_f, cat_cols):
    # Reuse the SAME fixed integer category codes already present in X_tr_f/X_val_f
    # (produced by PitchPreprocessor / the count_x_base train-only dict) -- no
    # re-derivation from val's own value set, so this is row-independent.
    Xtr = X_tr_f.fillna(0.0).astype(np.float32)
    Xval = X_val_f.fillna(0.0).astype(np.float32)
    m = ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_leaf=20,
                              n_jobs=-1, random_state=SEED)
    m.fit(Xtr, y_tr_f)
    p = np.clip(m.predict_proba(Xval)[:, 1], 1e-6, 1 - 1e-6)
    return p


def best_shared_w(results, model_key, folds=(2022, 2023)):
    best_w, best_avg = 0.0, float(np.mean([results[vs]['sk_gbdt'] for vs in folds]))
    gbdt_only = best_avg
    for w in np.linspace(0, 0.5, 26):
        sks = []
        for vs in folds:
            r = results[vs]
            p_blend = np.clip((1 - w) * r['p_gbdt'] + w * r[model_key], 1e-6, 1 - 1e-6)
            sk, *_ = calc_brier_skill_score(r['y'], p_blend)
            sks.append(sk)
        avg = float(np.mean(sks))
        if avg > best_avg:
            best_avg, best_w = avg, float(w)
    return best_w, best_avg, gbdt_only


def main():
    t_start = time.time()
    df_train = pd.read_csv(config.TRAIN_PATH)
    folds = get_cv_folds(df_train)
    inner_folds = [f for f in folds if f.val_season in (2022, 2023)]

    results = {}
    for fold in inner_folds:
        vs = fold.val_season
        log(f"=== fold val={vs}: building asof_dec (v2) feature frames ===")
        t0 = time.time()
        X_tr_f, X_val_f, y_tr_f, y_val_f = build_asofdec_fold_frames(df_train, fold)
        cat_cols = get_cat_cols(X_tr_f)
        log(f"fold val={vs}: X_tr={X_tr_f.shape} X_val={X_val_f.shape} built in {time.time()-t0:.1f}s")

        log(f"fold val={vs}: fitting GBDT reference (single seed={SEED}) ...")
        t0 = time.time()
        p_gbdt = fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f, cat_cols)
        sk_gbdt, *_ = calc_brier_skill_score(y_val_f, p_gbdt)
        log(f"fold val={vs}: GBDT alone skill={sk_gbdt:.2f} ({time.time()-t0:.1f}s)")

        log(f"fold val={vs}: fitting Logistic Regression (one-hot + scaled) ...")
        t0 = time.time()
        p_log = fit_predict_logistic(X_tr_f, X_val_f, y_tr_f, cat_cols)
        sk_log, *_ = calc_brier_skill_score(y_val_f, p_log)
        corr_log = float(np.corrcoef(p_log, p_gbdt)[0, 1])
        log(f"fold val={vs}: LogisticRegression alone skill={sk_log:.2f} corr(vs GBDT)={corr_log:.4f} "
            f"({time.time()-t0:.1f}s)")

        log(f"fold val={vs}: fitting ExtraTrees ...")
        t0 = time.time()
        p_et = fit_predict_extratrees(X_tr_f, X_val_f, y_tr_f, cat_cols)
        sk_et, *_ = calc_brier_skill_score(y_val_f, p_et)
        corr_et = float(np.corrcoef(p_et, p_gbdt)[0, 1])
        log(f"fold val={vs}: ExtraTrees alone skill={sk_et:.2f} corr(vs GBDT)={corr_et:.4f} "
            f"({time.time()-t0:.1f}s)")

        results[vs] = dict(p_gbdt=p_gbdt, sk_gbdt=sk_gbdt, p_log=p_log, sk_log=sk_log, corr_log=corr_log,
                            p_et=p_et, sk_et=sk_et, corr_et=corr_et, y=y_val_f)

    print("\n\n=== SUMMARY (INNER ONLY -- 2024/outer never built) ===")
    for vs in (2022, 2023):
        r = results[vs]
        print(f"val={vs}: GBDT={r['sk_gbdt']:.2f} | Logistic={r['sk_log']:.2f} (corr={r['corr_log']:.4f}) | "
              f"ExtraTrees={r['sk_et']:.2f} (corr={r['corr_et']:.4f})")

    for model_key, name in [('p_log', 'LogisticRegression'), ('p_et', 'ExtraTrees')]:
        w, avg, gbdt_only = best_shared_w(results, model_key)
        print(f"\n{name}: shared inner-only best w={w:.2f} -> inner avg skill={avg:.2f} "
              f"(GBDT-alone inner avg={gbdt_only:.2f}, gain={avg-gbdt_only:+.2f})")

    np.savez('/tmp/agent7_ml_diversify_screen.npz',
              **{f'p_gbdt_{vs}': results[vs]['p_gbdt'] for vs in (2022, 2023)},
              **{f'p_log_{vs}': results[vs]['p_log'] for vs in (2022, 2023)},
              **{f'p_et_{vs}': results[vs]['p_et'] for vs in (2022, 2023)},
              **{f'y_{vs}': results[vs]['y'] for vs in (2022, 2023)})
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
