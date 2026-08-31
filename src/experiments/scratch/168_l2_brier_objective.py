"""
168_l2_brier_objective.py
근본적으로 안 해본 각도: 대회 지표는 Brier score(=MSE)인데, 지금까지 모든 GBDT는
binary/logloss objective로 학습해왔음(분류기 predict_proba 사용). 대신 회귀
(L2/RMSE) objective로 직접 Brier를 최소화하도록 학습하면 평가지표와 학습목표가
일치해 더 나을 수 있다는 가설을 검증.
LGBMRegressor(objective='regression'), CatBoostRegressor(loss_function='RMSE'),
XGBRegressor(objective='reg:squarederror').
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
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_eval_l2(df_train, random_seeds, apply_shift_search=True):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    folds = get_cv_folds(df_train)
    oof_preds_lgb = np.zeros(len(df_train))
    oof_preds_cb = np.zeros(len(df_train))
    oof_preds_xgb = np.zeros(len(df_train))
    val_indices = []
    fold_details = []

    for k, fold in enumerate(folds):
        idx_tr, idx_val = fold.train_idx, fold.val_idx
        val_indices.extend(idx_val)
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

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))

        for seed in random_seeds:
            m_lgb = lgb.LGBMRegressor(objective='regression', n_estimators=250, num_leaves=45,
                                       learning_rate=0.05, min_child_samples=20,
                                       colsample_bytree=0.7, subsample=0.7,
                                       random_state=seed, verbosity=-1, n_jobs=-1)
            m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
            bag_p_lgb += np.clip(m_lgb.predict(X_val_f), 1e-6, 1 - 1e-6)

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
            m_cb.fit(X_tr_cb, y_tr_f)
            bag_p_cb += np.clip(m_cb.predict(X_val_cb), 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            m_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, max_depth=5,
                                      learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
                                      random_state=seed, n_jobs=-1)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f)
            bag_p_xgb += np.clip(m_xgb.predict(X_val_xgb.astype(np.float32)), 1e-6, 1 - 1e-6)

        n_seeds = len(random_seeds)
        p_lgb, p_cb, p_xgb = bag_p_lgb / n_seeds, bag_p_cb / n_seeds, bag_p_xgb / n_seeds
        oof_preds_lgb[idx_val], oof_preds_cb[idx_val], oof_preds_xgb[idx_val] = p_lgb, p_cb, p_xgb
        val_indices_arr_local = np.array(idx_val)
        fold_details.append({"fold": k + 1, "val_season": fold.val_season,
                              "p_lgb": p_lgb, "p_cb": p_cb, "p_xgb": p_xgb, "y": y_val_f})

    val_idx_arr = np.array(val_indices)

    # Shift search (nested: search shift on inner folds 2022/23 only, apply to all)
    def blend_skill(shift_lgb, shift_cb, shift_xgb, fold_list):
        skills = []
        for fd in fold_list:
            p = np.clip(w_lgb * (fd['p_lgb'] + shift_lgb) + w_cb * (fd['p_cb'] + shift_cb) +
                        w_xgb * (fd['p_xgb'] + shift_xgb), 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(fd['y'], p)
            skills.append(sk)
        return float(np.mean(skills))

    inner_fds = [fd for fd in fold_details if fd['val_season'] in (2022, 2023)]

    if apply_shift_search:
        best_shift, best_sk = (0.0, 0.0, 0.0), -1e9
        for s_lgb in np.linspace(-0.05, 0.05, 5):
            for s_cb in np.linspace(-0.05, 0.05, 5):
                for s_xgb in np.linspace(-0.05, 0.05, 5):
                    sk = blend_skill(s_lgb, s_cb, s_xgb, inner_fds)
                    if sk > best_sk:
                        best_sk, best_shift = sk, (s_lgb, s_cb, s_xgb)
    else:
        best_shift = (0.0, 0.0, 0.0)

    full_skill = blend_skill(*best_shift, fold_details)
    all_fold_skills = []
    for fd in fold_details:
        p = np.clip(w_lgb * (fd['p_lgb'] + best_shift[0]) + w_cb * (fd['p_cb'] + best_shift[1]) +
                    w_xgb * (fd['p_xgb'] + best_shift[2]), 1e-6, 1 - 1e-6)
        sk, _, _, _ = calc_brier_skill_score(fd['y'], p)
        all_fold_skills.append({"fold": fd['fold'], "val_season": fd['val_season'], "skill_k": sk})

    return {"mean_fold_skill": full_skill, "best_shift": best_shift, "fold_details": all_fold_skills}


df_train = pd.read_csv(config.TRAIN_PATH)
SCREEN_SEEDS = [7, 123]
BASELINE_REF = 843.69

log("=== 168: L2/Brier(RMSE) 직접 목적함수 GBDT (2-seed 스크리닝) ===")
t0 = time.time()
r = run_eval_l2(df_train, SCREEN_SEEDS, apply_shift_search=True)
dt = (time.time() - t0) / 60
log(f"[L2-objective, shift-tuned(nested inner)] 2-seed skill={r['mean_fold_skill']:.2f} "
    f"(delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) best_shift={r['best_shift']} "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")

result = {'l2_objective_screen': r['mean_fold_skill'], 'best_shift': r['best_shift'],
          'fold_details': r['fold_details'], 'minutes': dt}
with open('/tmp/168_result.json', 'w') as f:
    json.dump(result, f, indent=2)

if r['mean_fold_skill'] > BASELINE_REF - 5.0:
    log("\nPromising (screen close to/above baseline), promoting to full 5-seed confirm ...")
    FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
    t0 = time.time()
    r_full = run_eval_l2(df_train, FULL_SEEDS, apply_shift_search=True)
    dt = (time.time() - t0) / 60
    log(f"[FULL 5-seed L2-objective] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({dt:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
    with open('/tmp/168_result.json', 'w') as f:
        json.dump(result, f, indent=2)
else:
    log(f"\nNot promising enough, skipping full confirm.")

log("\n=== 168 DONE ===")
