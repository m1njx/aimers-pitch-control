"""
180_l2recency_tabm_blend.py
179번의 GBDT(L2목적함수+recency decay=0.7 결합, 869.90점)을 OOF 저장하도록
재실행하고, 캐시된 TabM 5-seed OOF(main-env, report157/158)와 nested-honest
블렌딩(inner 2022/23 선택 -> outer 2024 최초 적용) 시도.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

BEST_SHIFT = (-0.025, 0.0, -0.05)
DECAY = 0.7
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]


def train_l2recency_and_save_oof(df_train, seeds, decay, shift):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = shift
    folds = get_cv_folds(df_train)
    n = len(df_train)
    oof_ens = np.full(n, np.nan)

    for k, fold in enumerate(folds):
        idx_tr, idx_val = fold.train_idx, fold.val_idx
        df_tr_f = df_train.iloc[idx_tr].copy()
        df_val_f = df_train.iloc[idx_val].copy()
        as_of = fold.fold_max_season

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

        y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)

        season_gap = (as_of - df_tr_f['season']).clip(lower=0).values
        sw = np.power(decay, season_gap).astype(np.float64)
        sw = sw / sw.mean()

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))

        for seed in seeds:
            cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]
            m_lgb = lgb.LGBMRegressor(objective='regression', n_estimators=250, num_leaves=45,
                                       learning_rate=0.05, min_child_samples=20,
                                       colsample_bytree=0.7, subsample=0.7,
                                       random_state=seed, verbosity=-1, n_jobs=-1)
            m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx, sample_weight=sw)
            bag_p_lgb += np.clip(m_lgb.predict(X_val_f) + s_lgb, 1e-6, 1 - 1e-6)

            X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
                X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
            for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
                X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
                X_val_cb[c] = X_val_cb[c].astype(np.float32)
            m_cb = CatBoostRegressor(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                      loss_function='RMSE', random_seed=seed, verbose=0,
                                      cat_features=cat_cols, thread_count=-1)
            m_cb.fit(X_tr_cb, y_tr_f, sample_weight=sw)
            bag_p_cb += np.clip(m_cb.predict(X_val_cb) + s_cb, 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            m_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, max_depth=5,
                                      learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
                                      random_state=seed, n_jobs=-1)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f, sample_weight=sw)
            bag_p_xgb += np.clip(m_xgb.predict(X_val_xgb.astype(np.float32)) + s_xgb, 1e-6, 1 - 1e-6)

        n_seeds = len(seeds)
        p_lgb, p_cb, p_xgb = bag_p_lgb / n_seeds, bag_p_cb / n_seeds, bag_p_xgb / n_seeds
        p_ens = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
        oof_ens[idx_val] = p_ens
        log(f"  fold{k+1}({fold.val_season}) 완료")

    return oof_ens


df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train)
y_full = df_train[config.TARGET_COL].values
n = len(df_train)
inner_folds = [f for f in folds if f.val_season in (2022, 2023)]
outer_fold = [f for f in folds if f.val_season == 2024][0]


def fold_skill_mean(p_full, fold_list):
    return float(np.mean([calc_brier_skill_score(y_full[f.val_idx], np.clip(p_full[f.val_idx], 1e-6, 1-1e-6))[0]
                           for f in fold_list]))


log("=== 180: GBDT(L2+recency) OOF 저장 + TabM 블렌딩 ===")
t0 = time.time()
p_gbdt_new = train_l2recency_and_save_oof(df_train, FULL_SEEDS, DECAY, BEST_SHIFT)
log(f"GBDT(L2+recency) 학습 완료 ({(time.time()-t0)/60:.1f}min)")
np.save('/tmp/180_gbdt_l2recency_oof.npy', p_gbdt_new)

verify_skill = fold_skill_mean(p_gbdt_new, folds)
log(f"[검증] nested-full skill={verify_skill:.2f} (179번 기록값: 869.90)")

d_tabm = np.load('/tmp/tabm_5seed_oof.npz')
p_tabm = d_tabm['oof']

mask = ~np.isnan(p_gbdt_new) & ~np.isnan(p_tabm)
corr = np.corrcoef(p_gbdt_new[mask], p_tabm[mask])[0, 1]
log(f"GBDT(L2+recency) vs TabM 상관계수: {corr:.4f} (참고: 기존 GBDT-classification vs TabM는 0.9256)")

inner_idx = np.concatenate([f.val_idx for f in inner_folds])
best_w, best_inner = 0.0, -1e9
for w in np.linspace(0, 0.6, 31):
    p_blend = np.clip((1 - w) * p_gbdt_new + w * p_tabm, 1e-6, 1 - 1e-6)
    s = fold_skill_mean(p_blend, inner_folds)
    if s > best_inner:
        best_inner, best_w = s, float(w)

p_blend_full = np.clip((1 - best_w) * p_gbdt_new + best_w * p_tabm, 1e-6, 1 - 1e-6)
honest_full = fold_skill_mean(p_blend_full, folds)
outer_only = fold_skill_mean(p_blend_full, [outer_fold])
gbdt_alone_outer = fold_skill_mean(p_gbdt_new, [outer_fold])

log(f"\nbest_w_tabm(inner-선택)={best_w:.2f}")
log(f"nested-honest full skill(블렌딩)={honest_full:.2f} (GBDT단독 869.90, 기존 GBDT+TabM 블렌딩 888.43)")
log(f"outer(2024)-only: 블렌딩={outer_only:.2f}  GBDT단독={gbdt_alone_outer:.2f}")

result = {
    'gbdt_l2recency_verify': verify_skill,
    'corr_with_tabm': float(corr),
    'best_w_tabm': best_w,
    'blend_nested_full': honest_full,
    'blend_outer_only': outer_only,
    'gbdt_alone_outer_only': gbdt_alone_outer,
}
with open('/tmp/180_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 180 DONE ===")
