"""
explore_catboost_monotone.py
CatBoost는 SSOT 앙상블에서 75% 비중을 차지하는 핵심 모델. LGBM/XGBoost에서 검증한 동일한
4개 도메인 피처 단조제약을 CatBoost에도 적용했을 때 효과가 있는지 확인 (single model, multi-seed).
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

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
SEEDS = [42, 100, 2024]
results = {'free': {s: [] for s in SEEDS}, 'mono': {s: [] for s in SEEDS}}

t0 = time.time()
for k, fold in enumerate(folds):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()

    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)

    y_tr_f = df_tr_f[target_col].values
    y_val_f = df_val_f[target_col].values

    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL]

    mono_vec = [0] * len(X_tr_f.columns)
    for feat, direction in MONO_FEATURES.items():
        if feat in X_tr_f.columns:
            mono_vec[X_tr_f.columns.get_loc(feat)] = direction

    X_tr_cb = X_tr_f.copy(); X_val_cb = X_val_f.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)

    for seed in SEEDS:
        m_free = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                     random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
        m_free.fit(X_tr_cb, y_tr_f)
        p_free = np.clip(m_free.predict_proba(X_val_cb)[:, 1] - 0.008, 1e-6, 1 - 1e-6)
        sk, br, _, _ = calc_brier_skill_score(y_val_f, p_free)
        results['free'][seed].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

        m_mono = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                     random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1,
                                     monotone_constraints=mono_vec)
        m_mono.fit(X_tr_cb, y_tr_f)
        p_mono = np.clip(m_mono.predict_proba(X_val_cb)[:, 1] - 0.008, 1e-6, 1 - 1e-6)
        sk, br, _, _ = calc_brier_skill_score(y_val_f, p_mono)
        results['mono'][seed].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    print(f"Fold {k+1} ({fold.val_season}) done. elapsed={time.time()-t0:.1f}s")

print("\n=== CATBOOST MONOTONE RESULTS (multi-seed) ===")
deltas = []
for seed in SEEDS:
    sk_free = evaluate_fold_skills(results['free'][seed])
    sk_mono = evaluate_fold_skills(results['mono'][seed])
    d = sk_mono - sk_free
    deltas.append(d)
    print(f"  seed={seed}: free={sk_free:.2f}점 mono={sk_mono:.2f}점 delta={d:+.2f}점")
mean_delta = float(np.mean(deltas))
std_delta = float(np.std(deltas))
print(f"\n3-seed mean delta: {mean_delta:+.2f}점 (std={std_delta:.2f}점)")

with open('/tmp/catboost_mono_result.txt', 'w') as f:
    f.write(f"CatBoost monotone 3-seed mean delta: {mean_delta:+.2f}점 (std={std_delta:.2f}점)\n")
    f.write(f"Individual: {deltas}\n")
print("Done.")
