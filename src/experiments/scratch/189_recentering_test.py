"""
189_recentering_test.py
agent3(독립 에이전트)가 발견한 "recentering" 기법을 메인 파이프라인(843.69 classification
기준)에 검증. 각 fold의 예측 평균을, 직전 시즌(fold_max_season)의 실제 타겟 비율에
맞춰 재중심화(mean-matching shift). r_last는 inner/outer 구분 없이 각 fold 자신의
"학습 데이터 마지막 시즌" 값만 쓰므로 리키지 없음(항상 그 fold의 train 구간 안의 정보).
"""
import sys, time, json, warnings
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
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_eval_recenter(df_train, seeds, recenter=False):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
    folds = get_cv_folds(df_train)
    fold_details = []
    for k, fold in enumerate(folds):
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
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
        y_tr_f = df_tr_f[config.TARGET_COL].values
        y_val_f = df_val_f[config.TARGET_COL].values

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        bag_p_lgb = np.zeros(len(fold.val_idx))
        bag_p_cb = np.zeros(len(fold.val_idx))
        bag_p_xgb = np.zeros(len(fold.val_idx))
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

        if recenter:
            r_last = df_train[df_train['season'] == as_of][config.TARGET_COL].mean()
            p_mean = p_ens.mean()
            p_ens = np.clip(p_ens + (r_last - p_mean), 1e-6, 1 - 1e-6)
            log(f"  fold{k+1}: as_of={as_of} r_last={r_last:.4f} p_mean(원래)={p_mean:.4f} shift={r_last-p_mean:+.4f}")

        sk, _, _, _ = calc_brier_skill_score(y_val_f, p_ens)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk})
    return {'mean_fold_skill': evaluate_fold_skills(fold_details), 'fold_details': fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]

log("=== 189: recentering(직전시즌 실제비율로 예측평균 재중심화) 검증, classification 843.69 기준 ===")
t0 = time.time()
r_base = run_eval_recenter(df_train, FULL_SEEDS, recenter=False)
log(f"[baseline] skill={r_base['mean_fold_skill']:.2f} folds={[round(fd['skill_k'],2) for fd in r_base['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_rc = run_eval_recenter(df_train, FULL_SEEDS, recenter=True)
log(f"[+recentering] skill={r_rc['mean_fold_skill']:.2f} folds={[round(fd['skill_k'],2) for fd in r_rc['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

log(f"\n요약: baseline outer(fold3)={r_base['fold_details'][2]['skill_k']:.2f}, "
    f"recentering outer(fold3)={r_rc['fold_details'][2]['skill_k']:.2f}, "
    f"delta={r_rc['fold_details'][2]['skill_k']-r_base['fold_details'][2]['skill_k']:+.2f}")
log(f"nested-full: baseline={r_base['mean_fold_skill']:.2f} recentering={r_rc['mean_fold_skill']:.2f} "
    f"delta={r_rc['mean_fold_skill']-r_base['mean_fold_skill']:+.2f}")

with open('/tmp/189_result.json', 'w') as f:
    json.dump({'baseline': r_base, 'recentering': r_rc}, f, indent=2)
log("=== 189 DONE ===")
