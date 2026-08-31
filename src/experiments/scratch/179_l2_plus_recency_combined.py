"""
179_l2_plus_recency_combined.py
168번(L2/Brier 직접 목적함수, +9.70점=853.39)과 174번(recency decay=0.7 가중치,
+23.52점=867.21)을 결합. 서로 다른 축(학습 목적함수 vs 샘플 가중치)이라 함께
적용하면 더 좋아질 가능성. LGBMRegressor/CatBoostRegressor/XGBRegressor에
sample_weight=recency_weight를 동시에 적용.
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
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

BEST_SHIFT = (-0.025, 0.0, -0.05)  # 168번에서 nested-선택된 shift (L2 objective 전용)
DECAY = 0.7


def run_eval_l2_recency(df_train, random_seeds, decay, shift):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = shift
    folds = get_cv_folds(df_train)
    fold_details = []

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
        y_val_f = df_val_f[config.TARGET_COL].values

        season_gap = (as_of - df_tr_f['season']).clip(lower=0).values
        sw = np.power(decay, season_gap).astype(np.float64)
        sw = sw / sw.mean()

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))

        for seed in random_seeds:
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

        n_seeds = len(random_seeds)
        p_lgb, p_cb, p_xgb = bag_p_lgb / n_seeds, bag_p_cb / n_seeds, bag_p_xgb / n_seeds
        p_ens_fold = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
        skill_k, _, _, _ = calc_brier_skill_score(y_val_f, p_ens_fold)
        fold_details.append({"fold": k + 1, "val_season": fold.val_season, "skill_k": float(skill_k)})

    return {"mean_fold_skill": evaluate_fold_skills(fold_details), "fold_details": fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

log("=== 179: L2 목적함수 + recency(decay=0.7) 가중치 결합, 2-seed 스크리닝 ===")
log("(참고: L2단독=853.39, recency단독=867.21)")
t0 = time.time()
r = run_eval_l2_recency(df_train, SCREEN_SEEDS, DECAY, BEST_SHIFT)
dt = (time.time() - t0) / 60
log(f"[L2+recency] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")

result = {'screen_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

log("\n5-seed 정식 확인 진행 (결합 아이디어라 스크리닝 문턱 없이 바로 확인)")
t0 = time.time()
r_full = run_eval_l2_recency(df_train, FULL_SEEDS, DECAY, BEST_SHIFT)
dt = (time.time() - t0) / 60
log(f"[FULL 5-seed L2+recency] skill={r_full['mean_fold_skill']:.2f} "
    f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}, "
    f"vs L2단독 853.39={r_full['mean_fold_skill']-853.39:+.2f}, vs recency단독 867.21={r_full['mean_fold_skill']-867.21:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({dt:.1f}min)")

result['full_5seed_skill'] = r_full['mean_fold_skill']
result['full_fold_details'] = r_full['fold_details']
with open('/tmp/179_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 179 DONE ===")
