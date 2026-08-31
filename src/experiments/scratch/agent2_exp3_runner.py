"""
agent2_exp3_runner.py — Caches RAW (unshifted) per-model validation predictions
for several feature variants across an EXTENDED set of expanding-window folds
(val = 2020..2024).

Why raw + extended folds:
 1. Raw predictions let every shift / recalibration rule be explored offline
    for free, instead of one expensive CV run per candidate.
 2. Extra folds (2020, 2021) give a season SERIES of optimal shifts, so the
    shift for the next unseen season can be EXTRAPOLATED instead of averaged.
    This is fully legal (train labels only, no test statistics).

Variants:
  base      - current SSOT 69 features
  tkm_prof  - + per-pitcher trackman physical/mechanics profile (NEW; enabled by
              pitcher_id <-> pitcher_trackman_id entity resolution)
  tkm_all   - + per-(pitcher,count) and per-(pitcher,batter_hand) arsenal mix
"""
import sys, os, time, json
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
from agent2_common import build_base_features, base_cat_cols, log
from agent2_tkm_profile import PitcherTrackmanProfile, load_pitcher_map

OUT = os.environ.get('AGENT2_CACHE', '~/LG_data/scratch/agent2_cache')
os.makedirs(OUT, exist_ok=True)
SEEDS = [7, 123]
FIX_INDEX = os.environ.get('FIX_INDEX', '0') == '1'
VAL_SEASONS = [2021, 2022, 2023, 2024]


def fit_predict_raw(X_tr, y_tr, X_val, cat_cols, seed):
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    m = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                           min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                           random_state=seed, verbosity=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, categorical_feature=cat_idx)
    p_lgb = m.predict_proba(X_val)[:, 1]

    X_tr_cb = X_tr.copy(); X_val_cb = X_val.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32); X_val_cb[c] = X_val_cb[c].astype(np.float32)
    m = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                           random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
    m.fit(X_tr_cb, y_tr)
    p_cb = m.predict_proba(X_val_cb)[:, 1]

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
    m.fit(X_tr_x.astype(np.float32), y_tr)
    p_xgb = m.predict_proba(X_val_x.astype(np.float32))[:, 1]
    return p_lgb, p_cb, p_xgb


def main(variants, seeds=SEEDS, val_seasons=VAL_SEASONS, tag=''):
    df = pd.read_csv(config.TRAIN_PATH)
    pmap = load_pitcher_map()
    tm_full = pd.read_csv(config.TRACKMAN_PATH)
    log(f"mapped pitchers: {len(pmap)}")

    for vs in val_seasons:
        tr_mask = (df['season'] >= 2019) & (df['season'] < vs)
        va_mask = df['season'] == vs
        df_tr = df[tr_mask].copy(); df_val = df[va_mask].copy()
        as_of = vs - 1
        t0 = time.time()
        X_tr_b, X_val_b = build_base_features(df_tr, df_val, as_of, fix_index=FIX_INDEX)
        cc_b = base_cat_cols(X_tr_b)
        y_tr = df_tr[config.TARGET_COL].values
        y_val = df_val[config.TARGET_COL].values
        log(f"val={vs}: base ready ({len(df_tr):,} train) in {time.time()-t0:.0f}s")

        prof = None
        if any(v != 'base' for v in variants):
            prof = PitcherTrackmanProfile(pmap).fit(tm_full[tm_full.season <= as_of])

        for vname in variants:
            f = f'{OUT}/{tag}{vname}_val{vs}.npz'
            if os.path.exists(f):
                log(f"  skip existing {f}"); continue
            X_tr, X_val, cc = X_tr_b, X_val_b, cc_b
            if vname == 'tkm_prof':
                A = prof.transform(df_tr, groups=('prof',)); B = prof.transform(df_val, groups=('prof',))
                X_tr = pd.concat([X_tr_b, A], axis=1); X_val = pd.concat([X_val_b, B], axis=1)
            elif vname == 'tkm_all':
                A = prof.transform(df_tr); B = prof.transform(df_val)
                X_tr = pd.concat([X_tr_b, A], axis=1); X_val = pd.concat([X_val_b, B], axis=1)
            t1 = time.time()
            P = np.zeros((3, len(X_val)))
            for seed in seeds:
                pl, pc, px = fit_predict_raw(X_tr, y_tr, X_val, cc, seed)
                P[0] += pl; P[1] += pc; P[2] += px
            P /= len(seeds)
            np.savez_compressed(f, y=y_val.astype(np.int8), p_lgb=P[0].astype(np.float32),
                                p_cb=P[1].astype(np.float32), p_xgb=P[2].astype(np.float32),
                                val_season=vs, n_features=X_tr.shape[1])
            log(f"  [{vname}] val={vs} saved ({X_tr.shape[1]} feats) in {time.time()-t1:.0f}s")


if __name__ == '__main__':
    variants = sys.argv[1].split(',') if len(sys.argv) > 1 else ['base', 'tkm_prof', 'tkm_all']
    vs = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else VAL_SEASONS
    main(variants, val_seasons=vs)
