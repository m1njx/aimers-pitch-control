"""
explore_mono_full_ensemble.py
XGBoost 단독 5-seed 검증에서 단조제약이 +32.79점(전부 양수, std 9.98)으로 강하게 재현됨을 확인.
XGBoost는 SSOT 앙상블에서 10% 비중밖에 안 되므로, 실제 전체 앙상블(15/75/10)에 적용했을 때도
효과가 살아남는지 확인. LightGBM 단조제약도 함께 재확인. 여러 시드로 비교.
"""
import sys, os, time, warnings
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

df_train = pd.read_csv(config.TRAIN_PATH)
target_col = config.TARGET_COL
folds = get_cv_folds(df_train)

MONO_FEATURES = {
    'asof_pitcher_success_rate': 1,
    'asof_pitcher_reverse_rate': -1,
    'asof_batter_success_rate': 1,
    'asof_pitcher_prev5_game_success_rate': 1,
}
SSOT_SKILL = 853.62
w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
SEEDS = [42, 100, 2024]

configs = {
    'baseline': {'lgb_mono': False, 'xgb_mono': False},
    'xgb_mono_only': {'lgb_mono': False, 'xgb_mono': True},
    'lgb_mono_only': {'lgb_mono': True, 'xgb_mono': False},
    'both_mono': {'lgb_mono': True, 'xgb_mono': True},
}

results = {cfg: {s: [] for s in SEEDS} for cfg in configs}

t0 = time.time()
for k, fold in enumerate(folds):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()

    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
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

    y_tr_f = df_tr_f[target_col].values
    y_val_f = df_val_f[target_col].values

    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    mono_vec = [0] * len(X_tr_f.columns)
    for feat, direction in MONO_FEATURES.items():
        if feat in X_tr_f.columns:
            mono_vec[X_tr_f.columns.get_loc(feat)] = direction
    mono_tuple = '(' + ','.join(str(v) for v in mono_vec) + ')'

    X_tr_cb = X_tr_f.copy(); X_val_cb = X_val_f.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)

    X_tr_xgb = X_tr_f.copy(); X_val_xgb = X_val_f.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
    X_tr_xgb = X_tr_xgb.astype(np.float32); X_val_xgb = X_val_xgb.astype(np.float32)

    for seed in SEEDS:
        m_lgb_free = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                         min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                         random_state=seed, verbosity=-1, n_jobs=-1)
        m_lgb_free.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
        p_lgb_free = np.clip(m_lgb_free.predict_proba(X_val_f)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

        m_lgb_mono = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                         min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                         random_state=seed, verbosity=-1, n_jobs=-1,
                                         monotone_constraints=mono_vec, monotone_constraints_method='basic')
        m_lgb_mono.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
        p_lgb_mono = np.clip(m_lgb_mono.predict_proba(X_val_f)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

        m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                   random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
        m_cb.fit(X_tr_cb, y_tr_f)
        p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

        m_xgb_free = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                        colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                        n_jobs=-1, eval_metric='logloss')
        m_xgb_free.fit(X_tr_xgb, y_tr_f)
        p_xgb_free = np.clip(m_xgb_free.predict_proba(X_val_xgb)[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

        m_xgb_mono = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                        colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                        n_jobs=-1, eval_metric='logloss', monotone_constraints=mono_tuple)
        m_xgb_mono.fit(X_tr_xgb, y_tr_f)
        p_xgb_mono = np.clip(m_xgb_mono.predict_proba(X_val_xgb)[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

        for cfg_name, cfg in configs.items():
            p_lgb_use = p_lgb_mono if cfg['lgb_mono'] else p_lgb_free
            p_xgb_use = p_xgb_mono if cfg['xgb_mono'] else p_xgb_free
            p_ens = np.clip(w_lgb * p_lgb_use + w_cb * p_cb + w_xgb * p_xgb_use, 1e-6, 1 - 1e-6)
            sk, br, _, _ = calc_brier_skill_score(y_val_f, p_ens)
            results[cfg_name][seed].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    print(f"Fold {k+1} ({fold.val_season}) done. elapsed={time.time()-t0:.1f}s")

print("\n=== FULL ENSEMBLE RESULTS (multi-seed) ===")
summary_lines = []
for cfg_name in configs:
    seed_skills = [evaluate_fold_skills(results[cfg_name][s]) for s in SEEDS]
    mean_skill = float(np.mean(seed_skills))
    delta = mean_skill - SSOT_SKILL
    line = f"{cfg_name:20s}: seeds={[f'{s:.2f}' for s in seed_skills]} mean={mean_skill:.2f}점 delta_vs_SSOT={delta:+.2f}점"
    print(line)
    summary_lines.append(line)

with open('/tmp/mono_full_ensemble_result.txt', 'w') as f:
    f.write('\n'.join(summary_lines))
print("\nDone. Summary saved to /tmp/mono_full_ensemble_result.txt")
