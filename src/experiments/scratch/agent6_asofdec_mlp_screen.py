"""
agent6_asofdec_mlp_screen.py

Low-risk screening: does a simple MLP (CPU-forced, to avoid the MPS stall
history documented in report 161) trained on top of the asof_dec feature set
(the current deployed SSOT, AsofDecomposer2 v2) add anything when blended
with the asof_dec GBDT ensemble?

Scope: INNER folds ONLY (val=2022, val=2023). The 2024 (outer) fold is never
built or touched by this script -- that keeps the "outer exactly once, only
for a genuinely promising frozen candidate" principle intact. If this
screening shows a real signal, a SEPARATE follow-up script would apply the
frozen blend weight to outer exactly once.

Both the GBDT reference and the MLP are trained on the EXACT SAME per-fold
feature matrix (PitchPreprocessor + count_x_base + AsofDecomposer2), built
once per fold and reused for both, single seed (seed=7) for speed -- this is
a screening probe, not a final confirmation.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')  # force CPU -- MPS has a reproducible hang w/ TabM (report 161)
SEED = 7
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
# same model params as the deployed v11/v12 GBDT (agent5_exp_shift_extrap.py BASE_MP)
LGB_PARAMS = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                   colsample_bytree=0.7, subsample=0.7, random_state=SEED, verbosity=-1, n_jobs=-1)
CB_PARAMS = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                  random_seed=SEED, verbose=0, thread_count=-1)
XGB_PARAMS = dict(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8,
                   subsample=0.8, random_state=SEED, n_jobs=-1, eval_metric='logloss')


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
    dlc.add_count_x_base(df_tr_f, X_tr_f)
    dlc.add_count_x_base(df_val_f, X_val_f)
    cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
    X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_tr_f, X_val_f = add_asof_dec_features(df_tr_f, df_val_f, fold.fold_max_season, X_tr_f, X_val_f)
    y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)
    y_val_f = df_val_f[config.TARGET_COL].values.astype(np.float32)
    return X_tr_f, X_val_f, y_tr_f, y_val_f


def fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f):
    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
    p_lgb = np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + SHIFTS['lgb'], 1e-6, 1 - 1e-6)

    X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)
    m_cb = CatBoostClassifier(cat_features=cat_cols, **CB_PARAMS)
    m_cb.fit(X_tr_cb, y_tr_f)
    p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + SHIFTS['cb'], 1e-6, 1 - 1e-6)

    X_tr_x, X_val_x = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        X_tr_x[c] = X_tr_x[c].astype('category').cat.codes.astype(np.float32)
        X_val_x[c] = X_val_x[c].astype('category').cat.codes.astype(np.float32)
    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(X_tr_x.astype(np.float32), y_tr_f)
    p_xgb = np.clip(m_xgb.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS['xgb'], 1e-6, 1 - 1e-6)

    w_lgb, w_cb, w_xgb = WEIGHTS
    p_ens = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
    return p_ens


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
        log(f"fold val={vs}: X_tr={X_tr_f.shape} X_val={X_val_f.shape} built in {time.time()-t0:.1f}s")

        log(f"fold val={vs}: fitting GBDT reference (single seed={SEED}) ...")
        t0 = time.time()
        p_gbdt = fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f)
        sk_gbdt, br_gbdt, _, _ = calc_brier_skill_score(y_val_f, p_gbdt)
        log(f"fold val={vs}: GBDT alone skill={sk_gbdt:.2f} (raw brier={br_gbdt:.6f}) in {time.time()-t0:.1f}s")

        log(f"fold val={vs}: training MLP (CPU, single seed={SEED}) ...")
        t0 = time.time()
        tens = dlc.to_tensors(X_tr_f, X_val_f)
        num_tr, num_val = tens['num_tr'], tens['num_val']
        cat_tr, cat_val = tens['cat_tr'], tens['cat_val']
        y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)
        num_dim = num_tr.shape[1]
        cat_cardinalities = tens['cat_cardinalities']

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = dlc.SimpleMLP(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15)
        model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=8, lr=1e-3,
                                          batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                          verbose_prefix=f"[mlp val={vs}] ")
        p_mlp = dlc.predict(model, num_val, cat_val, DEVICE, shift)
        sk_mlp, br_mlp, _, _ = calc_brier_skill_score(y_val_f, p_mlp)
        log(f"fold val={vs}: MLP alone skill={sk_mlp:.2f} (raw brier={br_mlp:.6f}) in {time.time()-t0:.1f}s")

        corr = float(np.corrcoef(p_mlp, p_gbdt)[0, 1])
        log(f"fold val={vs}: corr(MLP, GBDT) = {corr:.4f}")

        best_w, best_sk = 0.0, sk_gbdt
        for w in np.linspace(0, 0.5, 26):
            p_blend = np.clip((1 - w) * p_gbdt + w * p_mlp, 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(y_val_f, p_blend)
            if sk > best_sk:
                best_sk, best_w = sk, float(w)
        log(f"fold val={vs}: per-fold best blend w_mlp={best_w:.2f} skill={best_sk:.2f} "
            f"(gain over GBDT-alone: {best_sk - sk_gbdt:+.2f})")

        results[vs] = dict(sk_mlp=sk_mlp, sk_gbdt=sk_gbdt, corr=corr, best_w=best_w, best_sk=best_sk,
                            gain=best_sk - sk_gbdt, p_mlp=p_mlp, p_gbdt=p_gbdt, y=y_val_f)

    log("=== inner-only SHARED-weight selection (single w maximizing avg of 2022,2023) ===")
    best_w_shared, best_avg = 0.0, float(np.mean([results[vs]['sk_gbdt'] for vs in (2022, 2023)]))
    gbdt_only_avg = best_avg
    for w in np.linspace(0, 0.5, 26):
        sks = []
        for vs in (2022, 2023):
            r = results[vs]
            p_blend = np.clip((1 - w) * r['p_gbdt'] + w * r['p_mlp'], 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(r['y'], p_blend)
            sks.append(sk)
        avg = float(np.mean(sks))
        if avg > best_avg:
            best_avg, best_w_shared = avg, float(w)
    log(f"shared best_w_mlp={best_w_shared:.2f}: inner avg skill={best_avg:.2f} vs GBDT-alone inner avg="
        f"{gbdt_only_avg:.2f} (gain {best_avg - gbdt_only_avg:+.2f})")

    print("\n\n=== SUMMARY (INNER ONLY -- 2024/outer never built) ===")
    for vs in (2022, 2023):
        r = results[vs]
        print(f"val={vs}: MLP alone={r['sk_mlp']:.2f} | GBDT alone={r['sk_gbdt']:.2f} | "
              f"corr={r['corr']:.4f} | per-fold best blend={r['best_sk']:.2f} (w={r['best_w']:.2f}, "
              f"gain={r['gain']:+.2f})")
    print(f"\nSHARED inner-only weight: w_mlp={best_w_shared:.2f} -> inner avg skill={best_avg:.2f} "
          f"(GBDT-alone inner avg={gbdt_only_avg:.2f}, gain={best_avg - gbdt_only_avg:+.2f})")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    np.savez('/tmp/agent6_mlp_asofdec_screen.npz',
              **{f'p_mlp_{vs}': results[vs]['p_mlp'] for vs in (2022, 2023)},
              **{f'p_gbdt_{vs}': results[vs]['p_gbdt'] for vs in (2022, 2023)},
              **{f'y_{vs}': results[vs]['y'] for vs in (2022, 2023)},
              best_w_shared=best_w_shared, gbdt_only_avg=gbdt_only_avg, best_avg=best_avg)


if __name__ == '__main__':
    main()
