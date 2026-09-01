# ==============================================================================
# DACON 1100+ Breakthrough: Super B (x1.8 Iteration Optimum) & GPU XGBoost Arm
# Target: Google Colab (T4 GPU)
# Dataset: arm_b_features.parquet (82 Complete Features, 203MB)
# ==============================================================================

!pip -q install catboost xgboost pyarrow

import os, sys, time, gc, json, zipfile
import numpy as np, pandas as pd
from google.colab import files

PARQUET = 'arm_b_features.parquet'
if not os.path.exists(PARQUET):
    print("\n[안내] 다운로드 폴더의 'arm_b_features.parquet' (203MB)를 업로드해주세요:")
    uploaded = files.upload()

t0 = time.time()
print("\n--- 82개 완전체 피처 로드 중 ---")
df = pd.read_parquet(PARQUET)
print(f"✓ 로드 완료: {len(df):,} 행 x {df.shape[1]} 열 ({time.time()-t0:.1f}초)")

# Feature preparation
TARGET = 'control_success'
drop_cols = ['control_success', 'season']
features = [c for c in df.columns if c not in drop_cols]
print(f"학습 피처 수: {len(features)}개")

# Categorical handling
CAT_COLS = ['top_bottom', 'game_type', 'pitcher_hand', 'batter_hand']
for c in CAT_COLS:
    if c in df.columns:
        df[c] = df[c].astype('category').cat.codes.astype(np.int32)

X = df[features].fillna(-999.0)
y = df[TARGET].values.astype(np.int8)
del df; gc.collect()

# ------------------------------------------------------------------------------
# Part 1: Super B (GPU CatBoost x1.8 Scale - 1800 Iterations, 20 Seeds)
# ------------------------------------------------------------------------------
from catboost import CatBoostClassifier

CB_SEEDS = [7, 42, 77, 123, 202, 365, 777, 1024, 1234, 2024,
            2025, 3141, 4096, 5555, 7890, 8888, 9001, 9999, 31337, 8675309]

CB_PARAMS = dict(
    iterations=1800,            # Empirical x1.8 optimum (+16.3 pts peak)
    learning_rate=0.015,
    depth=8,
    l2_leaf_reg=384.0,
    min_data_in_leaf=400,
    bootstrap_type='Bernoulli',
    subsample=0.9,
    border_count=128,
    task_type='GPU',
    devices='0',
    loss_function='Logloss',
    verbose=False
)

os.makedirs('super_b_models', exist_ok=True)
print(f"\n--- Part 1: Super B (1800회 x 20시드) GPU 학습 시작 ---")
t_cb = time.time()
for i, s in enumerate(CB_SEEDS, 1):
    t_s = time.time()
    cb = CatBoostClassifier(random_seed=s, **CB_PARAMS)
    cb.fit(X, y)
    cb.save_model(f'super_b_models/cb_super_seed{s}.cbm')
    print(f"  [CatBoost {i:02d}/20] Seed {s:>8} 완료 ({time.time()-t_s:.1f}초)")

print(f"✓ Super B 학습 완료! 소요시간: {time.time()-t_cb:.1f}초")

# ------------------------------------------------------------------------------
# Part 2: GPU XGBoost Arm (Depthwise Tree Diversity, 10 Seeds)
# ------------------------------------------------------------------------------
from xgboost import XGBClassifier

XGB_SEEDS = [7, 42, 123, 202, 777, 1024, 2024, 5555, 9001, 31415]

XGB_PARAMS = dict(
    tree_method='hist',
    device='cuda',
    n_estimators=1800,
    learning_rate=0.015,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=50.0,
    min_child_weight=200,
    eval_metric='logloss'
)

os.makedirs('xgb_gpu_models', exist_ok=True)
print(f"\n--- Part 2: GPU XGBoost (1800회 x 10시드) 학습 시작 ---")
t_xgb = time.time()
for i, s in enumerate(XGB_SEEDS, 1):
    t_s = time.time()
    xgb = XGBClassifier(random_state=s, **XGB_PARAMS)
    xgb.fit(X, y)
    xgb.save_model(f'xgb_gpu_models/xgb_seed{s}.json')
    print(f"  [XGBoost {i:02d}/10] Seed {s:>8} 완료 ({time.time()-t_s:.1f}초)")

print(f"✓ GPU XGBoost 학습 완료! 소요시간: {time.time()-t_xgb:.1f}초")

# ------------------------------------------------------------------------------
# Part 3: Package & Download All SOTA Breakthrough Models
# ------------------------------------------------------------------------------
print("\n산출물 압축 중...")
with zipfile.ZipFile('super_models_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir('super_b_models'):
        z.write(os.path.join('super_b_models', f), arcname=os.path.join('super_b', f))
    for f in os.listdir('xgb_gpu_models'):
        z.write(os.path.join('xgb_gpu_models', f), arcname=os.path.join('xgb', f))

print(f"✓ super_models_bundle.zip 생성 완료 ({os.path.getsize('super_models_bundle.zip')/(1024*1024):.2f} MB)")
print("\n[안내] 맥북으로 다운로드를 시작합니다...")
files.download('super_models_bundle.zip')
print("🎉 1100+ Breakthrough Super Models Ready!")
