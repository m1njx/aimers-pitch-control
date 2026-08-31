"""
192_asof_dec_2021_holdout.py
asof_dec(agent2/191번에서 outer(2024) +207~213점으로 확인된 피처셋)를 네 번째
독립 연도(2021)로 추가 검증. train=2019~2020(2개 시즌만), val=2021.
기존 3-fold CV(2022/2023/2024)에는 전혀 없던 홀드아웃이라, 진짜 신뢰도를 더 높일 수 있음.
공식 하네스(core/eval_utils.py)와 동일한 3-GBDT 앙상블(15/75/10, classification objective)
구조를 그대로 쓰되, get_cv_folds() 대신 수동으로 2021 fold를 구성.
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
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
VAL_SEASON = 2021


def run_eval_2021(df_train, seeds, use_asof_dec):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006

    df_tr_f = df_train[(df_train['season'] >= 2019) & (df_train['season'] < VAL_SEASON)].copy()
    df_val_f = df_train[df_train['season'] == VAL_SEASON].copy()
    as_of = VAL_SEASON - 1  # 2020

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

    if use_asof_dec:
        dec = AsofDecomposer2().fit(df_tr_f, val_season=VAL_SEASON)
        A_tr = dec.transform(df_tr_f)
        A_val = dec.transform(df_val_f)
        A_tr.index = X_tr_f.index
        A_val.index = X_val_f.index
        X_tr_f = pd.concat([X_tr_f, A_tr], axis=1)
        X_val_f = pd.concat([X_val_f, A_val], axis=1)

    y_tr_f = df_tr_f[config.TARGET_COL].values
    y_val_f = df_val_f[config.TARGET_COL].values
    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    bag_p_lgb = np.zeros(len(df_val_f))
    bag_p_cb = np.zeros(len(df_val_f))
    bag_p_xgb = np.zeros(len(df_val_f))
    for seed in seeds:
        m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                                    colsample_bytree=0.7, subsample=0.7, random_state=seed, verbosity=-1, n_jobs=-1)
        m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
        bag_p_lgb += np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

        X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
        for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_val_cb[c] = X_val_cb[c].astype(np.float32)
        m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                   random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
        m_cb.fit(X_tr_cb, y_tr_f)
        bag_p_cb += np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

        X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
        m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8,
                                   subsample=0.8, random_state=seed, n_jobs=-1, eval_metric='logloss')
        m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f)
        bag_p_xgb += np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

    n_seeds = len(seeds)
    p_ens = np.clip(w_lgb * (bag_p_lgb / n_seeds) + w_cb * (bag_p_cb / n_seeds) + w_xgb * (bag_p_xgb / n_seeds), 1e-6, 1 - 1e-6)
    sk, raw_brier, base_brier, r = calc_brier_skill_score(y_val_f, p_ens)
    return {'skill': sk, 'raw_brier': raw_brier, 'r': r, 'n_train': len(df_tr_f), 'n_val': len(df_val_f)}


df_train = pd.read_csv(config.TRAIN_PATH)
log(f"=== 192: asof_dec 2021 홀드아웃 검증 (train=2019~2020, val=2021, 5-seed) ===")

t0 = time.time()
r_base = run_eval_2021(df_train, FULL_SEEDS, use_asof_dec=False)
log(f"[baseline] skill={r_base['skill']:.2f} raw_brier={r_base['raw_brier']:.6f} r={r_base['r']:.4f} "
    f"n_train={r_base['n_train']} n_val={r_base['n_val']} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_dec = run_eval_2021(df_train, FULL_SEEDS, use_asof_dec=True)
log(f"[+asof_dec] skill={r_dec['skill']:.2f} raw_brier={r_dec['raw_brier']:.6f} r={r_dec['r']:.4f} "
    f"({(time.time()-t0)/60:.1f}min)")

log(f"\n=== 결과 요약 ===")
log(f"2021 홀드아웃(train=2019~2020만): baseline={r_base['skill']:.2f} asof_dec={r_dec['skill']:.2f} "
    f"delta={r_dec['skill']-r_base['skill']:+.2f}")
log(f"참고: 2024 outer 결과(191번) baseline=598.97 asof_dec=805.74 delta=+206.77")

with open('/tmp/192_result.json', 'w') as f:
    json.dump({'baseline': r_base, 'asof_dec': r_dec, 'delta': r_dec['skill'] - r_base['skill']}, f, indent=2)
log("=== 192 DONE ===")
