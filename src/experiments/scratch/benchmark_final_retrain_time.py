"""
benchmark_final_retrain_time.py
5-seed bagging이 실전 제출 10분(600초) 제한 내에 가능한지 확인하기 위해,
현재 SSOT 하이퍼파라미터로 전체 학습 데이터(2019~2024, fold 없음)를 1회 학습하는 데
걸리는 실제 시간을 측정. 평가 서버는 CPU 6개(로컬은 10개)이므로 여유 마진을 감안해야 함.
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
from preprocessing import PitchPreprocessor

df_train = pd.read_csv(config.TRAIN_PATH)
target_col = config.TARGET_COL
print(f"Full train: {len(df_train):,} rows")

t0 = time.time()
prep = PitchPreprocessor()
prep.fit(df_train, as_of_season=None, is_final=True)
X_tr = prep.transform(df_train)
t_prep = time.time() - t0
print(f"Preprocessing (fit+transform, final mode): {t_prep:.1f}s")

for df_src, X_dst in [(df_train, X_tr)]:
    b_str = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
             (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
             (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    c_str = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
             df_src['strikes_before'].fillna(0).astype(int).astype(str))
    X_dst['count_x_base'] = (c_str + '_' + b_str)
cat_map = {v: i for i, v in enumerate(X_tr['count_x_base'].unique())}
X_tr['count_x_base'] = X_tr['count_x_base'].map(cat_map).fillna(-1).astype(int)

y_tr = df_train[target_col].values
cat_cols = [c for c in X_tr.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

seed = 42

t1 = time.time()
m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                            min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                            random_state=seed, verbosity=-1, n_jobs=-1)
m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
t_lgb = time.time() - t1
print(f"LightGBM single fit: {t_lgb:.1f}s")

X_tr_cb = X_tr.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
    X_tr_cb[c] = X_tr_cb[c].astype(np.float32)

t2 = time.time()
m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                           random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
m_cb.fit(X_tr_cb, y_tr)
t_cb = time.time() - t2
print(f"CatBoost single fit: {t_cb:.1f}s")

X_tr_xgb = X_tr.copy()
for c in cat_cols:
    X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
X_tr_xgb = X_tr_xgb.astype(np.float32)

t3 = time.time()
m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                           colsample_bytree=0.8, subsample=0.8, random_state=seed,
                           n_jobs=-1, eval_metric='logloss')
m_xgb.fit(X_tr_xgb, y_tr)
t_xgb = time.time() - t3
print(f"XGBoost single fit: {t_xgb:.1f}s")

t_one_seed_fit_only = t_lgb + t_cb + t_xgb
t_one_seed_total = t_prep + t_one_seed_fit_only

print(f"\n=== SUMMARY (local, 10 CPU) ===")
print(f"Preprocessing (one-time, seed-independent): {t_prep:.1f}s")
print(f"Per-seed fit (LGBM+CB+XGB): {t_one_seed_fit_only:.1f}s")
print(f"1-seed total (prep + fit): {t_one_seed_total:.1f}s")
print(f"3-seed total (prep once + 3x fit): {t_prep + 3*t_one_seed_fit_only:.1f}s")
print(f"5-seed total (prep once + 5x fit): {t_prep + 5*t_one_seed_fit_only:.1f}s")
print(f"Server limit: 600s, server CPUs=6 vs local CPUs=10 (expect slower on server)")

with open('/tmp/retrain_benchmark_result.txt', 'w') as f:
    f.write(f"prep={t_prep:.1f}s lgb={t_lgb:.1f}s cb={t_cb:.1f}s xgb={t_xgb:.1f}s\n")
    f.write(f"per_seed_fit={t_one_seed_fit_only:.1f}s\n")
    f.write(f"1seed_total={t_one_seed_total:.1f}s\n")
    f.write(f"3seed_total={t_prep + 3*t_one_seed_fit_only:.1f}s\n")
    f.write(f"5seed_total={t_prep + 5*t_one_seed_fit_only:.1f}s\n")

print("\nDone.")
