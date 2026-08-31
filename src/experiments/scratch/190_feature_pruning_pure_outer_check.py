"""
190_feature_pruning_pure_outer_check.py
187번의 하위20% 피처 제거가 L2+recency(오염된 decay=0.7)와 섞여서 순수 효과를
알 수 없었음. classification 순수 baseline(843.69, recency 없음)에 동일한
14개 피처(하위20%, 187번의 importances_bottom20 순서 그대로) 제거만 단독 적용해
outer(2024)를 정직하게 재검증.
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

DROP_COLS = ['runner_on_1b', 'runner_on_3b', 'tkm_match', 'tkm_extension_std', 'is_scoring_position',
             'top_bottom', 'is_leading', 'runner_on_2b', 'tkm_spin_rate_mean', 'pitcher_hand',
             'tkm_zone_speed_mean', 'tkm_rel_side_std', 'tkm_horz_break_mean', 'tkm_rel_side_mean']
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]


def run_eval(df_train, seeds, drop_cols=None):
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

        if drop_cols:
            cols_to_drop = [c for c in drop_cols if c in X_tr_f.columns]
            X_tr_f = X_tr_f.drop(columns=cols_to_drop)
            X_val_f = X_val_f.drop(columns=cols_to_drop)

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
        sk, _, _, _ = calc_brier_skill_score(y_val_f, p_ens)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk})
    return {'mean_fold_skill': evaluate_fold_skills(fold_details), 'fold_details': fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)

log("=== 190: 순수 classification baseline + 하위20%피처 제거, outer 정직검증(5-seed) ===")
t0 = time.time()
r_base = run_eval(df_train, FULL_SEEDS, drop_cols=None)
log(f"[classification baseline] skill={r_base['mean_fold_skill']:.2f} "
    f"folds={[round(fd['skill_k'],2) for fd in r_base['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_drop = run_eval(df_train, FULL_SEEDS, drop_cols=DROP_COLS)
log(f"[+하위20%제거, {len(DROP_COLS)}개] skill={r_drop['mean_fold_skill']:.2f} "
    f"folds={[round(fd['skill_k'],2) for fd in r_drop['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

log(f"\n요약: baseline outer(fold3)={r_base['fold_details'][2]['skill_k']:.2f}, "
    f"드롭 outer(fold3)={r_drop['fold_details'][2]['skill_k']:.2f}, "
    f"delta={r_drop['fold_details'][2]['skill_k']-r_base['fold_details'][2]['skill_k']:+.2f} "
    f"(노이즈 프로브 추정 폭 ±31.75 참고)")
log(f"nested-full: baseline={r_base['mean_fold_skill']:.2f} 드롭={r_drop['mean_fold_skill']:.2f} "
    f"delta={r_drop['mean_fold_skill']-r_base['mean_fold_skill']:+.2f}")

with open('/tmp/190_result.json', 'w') as f:
    json.dump({'baseline': r_base, 'drop_20pct': r_drop, 'drop_cols': DROP_COLS}, f, indent=2)
log("=== 190 DONE ===")
