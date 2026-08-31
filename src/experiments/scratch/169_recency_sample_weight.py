"""
169_recency_sample_weight.py
안 해본 각도: li(레버리지) 재가중과는 다른 축으로, "최근 시즌일수록 테스트 분포에
더 가깝다"는 가정 하에 학습 데이터에 recency(최신도) 가중치를 부여.
weight = decay^(fold_max_season - row_season), decay<1이면 오래된 시즌일수록 감쇠.
94번(hard-example 재가중), 165번(li 재가중) 모두 REJECT였으므로 재가중 자체가
이 프로젝트 데이터에서 구조적으로 불리할 가능성이 높다는 사전 확률을 감안하고 검증.
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
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_eval_recency(df_train, decay, random_seeds):
    mp = {
        'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
        'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
        'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
    }
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
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

        y_tr_f = df_tr_f[config.TARGET_COL].values
        y_val_f = df_val_f[config.TARGET_COL].values

        if decay is not None:
            season_gap = (as_of - df_tr_f['season']).clip(lower=0).values
            sw = np.power(decay, season_gap).astype(np.float64)
            sw = sw / sw.mean()
        else:
            sw = None

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))

        for seed in random_seeds:
            lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05,
                               min_child_samples=20, random_state=seed, verbosity=-1, n_jobs=-1)
            lgb_params.update(mp.get('lgb', {}))
            lgb_params['random_state'] = seed
            cb_params = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                              random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
            cb_params.update(mp.get('cb', {}))
            cb_params['random_seed'] = seed
            cb_params['cat_features'] = cat_cols
            xgb_params = dict(n_estimators=250, max_depth=5, learning_rate=0.05,
                               random_state=seed, n_jobs=-1, eval_metric='logloss')
            xgb_params.update(mp.get('xgb', {}))
            xgb_params['random_state'] = seed

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
            m_cb = CatBoostClassifier(**cb_params)
            m_cb.fit(X_tr_cb, y_tr_f, sample_weight=sw)
            bag_p_cb += np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            m_xgb = xgb.XGBClassifier(**xgb_params)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f, sample_weight=sw)
            bag_p_xgb += np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

        n_seeds = len(random_seeds)
        p_lgb, p_cb, p_xgb = bag_p_lgb / n_seeds, bag_p_cb / n_seeds, bag_p_xgb / n_seeds
        oof_preds_lgb[idx_val], oof_preds_cb[idx_val], oof_preds_xgb[idx_val] = p_lgb, p_cb, p_xgb
        p_ens_fold = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
        skill_k, raw_brier_k, brier_base_k, r_k = calc_brier_skill_score(y_val_f, p_ens_fold)
        fold_details.append({"fold": k + 1, "val_season": fold.val_season, "skill_k": float(skill_k)})

    return {"mean_fold_skill": evaluate_fold_skills(fold_details), "fold_details": fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)
SCREEN_SEEDS = [7, 123]
BASELINE_REF = 843.69

log("=== 169: 시즌 recency(최신도) sample_weight 스크리닝 (2-seed) ===")

configs = [
    ('baseline(no weight)', None),
    ('decay=0.7', 0.7),
    ('decay=0.85', 0.85),
    ('decay=0.95(약함)', 0.95),
]

results = {}
for name, decay in configs:
    t0 = time.time()
    r = run_eval_recency(df_train, decay, SCREEN_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[{name}] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")
    results[name] = {'screen_skill': r['mean_fold_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

with open('/tmp/169_result.json', 'w') as f:
    json.dump(results, f, indent=2)
log("\n=== 169 DONE ===")
