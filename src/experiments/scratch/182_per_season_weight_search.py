"""
182_per_season_weight_search.py
174번의 매끈한 지수감쇠(decay=0.7) 대신, 시즌별로 완전히 독립적인 가중치를
inner(2022,2023) 폴드에서 nested 탐색. 규칙변화가 비선형(특정 시즌만 급격히
다름)이라면 매끈한 decay보다 더 잘 포착할 수 있음.
탐색공간 축소를 위해 각 시즌 가중치를 {0.3, 0.5, 0.7, 1.0, 1.3} 중에서 좌표하강
(coordinate descent) 방식으로 탐색 (전수탐색은 5^5=3125가지라 비쌈).
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

CANDIDATE_LEVELS = [0.3, 0.5, 0.7, 1.0, 1.3]
SCREEN_SEEDS = [7, 123]
BASELINE_REF = 843.69


def run_eval_season_weights(df_train, season_weight_map, seeds):
    mp = {
        'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
        'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
        'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
    }
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
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

        y_tr_f = df_tr_f[config.TARGET_COL].values
        y_val_f = df_val_f[config.TARGET_COL].values
        sw = df_tr_f['season'].map(season_weight_map).fillna(1.0).astype(np.float64).values
        sw = sw / sw.mean()

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))
        for seed in seeds:
            lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                               random_state=seed, verbosity=-1, n_jobs=-1)
            lgb_params.update(mp.get('lgb', {}))
            lgb_params['random_state'] = seed
            m_lgb = lgb.LGBMClassifier(**lgb_params)
            m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx, sample_weight=sw)
            bag_p_lgb += np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

            X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
                X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
            for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
                X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
                X_val_cb[c] = X_val_cb[c].astype(np.float32)
            cb_params = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                              random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
            cb_params.update(mp.get('cb', {}))
            cb_params['random_seed'] = seed
            cb_params['cat_features'] = cat_cols
            m_cb = CatBoostClassifier(**cb_params)
            m_cb.fit(X_tr_cb, y_tr_f, sample_weight=sw)
            bag_p_cb += np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            xgb_params = dict(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=seed,
                               n_jobs=-1, eval_metric='logloss')
            xgb_params.update(mp.get('xgb', {}))
            xgb_params['random_state'] = seed
            m_xgb = xgb.XGBClassifier(**xgb_params)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f, sample_weight=sw)
            bag_p_xgb += np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

        n_seeds = len(seeds)
        p_lgb, p_cb, p_xgb = bag_p_lgb / n_seeds, bag_p_cb / n_seeds, bag_p_xgb / n_seeds
        p_ens_fold = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
        skill_k, _, _, _ = calc_brier_skill_score(y_val_f, p_ens_fold)
        fold_details.append({"fold": k + 1, "val_season": fold.val_season, "skill_k": float(skill_k)})

    return {"mean_fold_skill": evaluate_fold_skills(fold_details), "fold_details": fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)
all_seasons = sorted(df_train['season'].unique().tolist())
log(f"=== 182: 시즌별 개별 가중치 좌표하강 탐색 (seasons={all_seasons}) ===")

# 시작점: 174번의 decay=0.7과 유사한 형태 (최신 시즌일수록 큰 가중치)
current_weights = {s: 1.0 for s in all_seasons}
# 초기화는 decay=0.7 형태로 시작 (as_of=2023 기준, inner-fold 관점)
for s in all_seasons:
    gap = max(0, 2023 - s)
    current_weights[s] = round(0.7 ** gap, 2)

r0 = run_eval_season_weights(df_train, current_weights, SCREEN_SEEDS)
log(f"[초기값(decay=0.7 근사) {current_weights}] skill={r0['mean_fold_skill']:.2f}")
best_skill = r0['mean_fold_skill']

for coord_season in all_seasons:
    best_level_here = current_weights[coord_season]
    for level in CANDIDATE_LEVELS:
        if abs(level - current_weights[coord_season]) < 1e-6:
            continue
        trial_weights = dict(current_weights)
        trial_weights[coord_season] = level
        r = run_eval_season_weights(df_train, trial_weights, SCREEN_SEEDS)
        log(f"  season={coord_season} weight={level}: skill={r['mean_fold_skill']:.2f}")
        if r['mean_fold_skill'] > best_skill:
            best_skill, best_level_here = r['mean_fold_skill'], level
    current_weights[coord_season] = best_level_here
    log(f"[coord={coord_season} 확정] best_level={best_level_here}, running_best_skill={best_skill:.2f}, weights={current_weights}")

log(f"\n최종 시즌별 가중치: {current_weights}")
log(f"최종 2-seed skill: {best_skill:.2f} (참고: decay=0.7 단독 862.60, L2+recency 856.67)")

with open('/tmp/182_result.json', 'w') as f:
    json.dump({'final_weights': current_weights, 'final_skill_2seed': best_skill}, f, indent=2)
log("\n=== 182 DONE ===")
