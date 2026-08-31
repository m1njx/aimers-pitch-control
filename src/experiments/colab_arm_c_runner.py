# ==============================================================================
# DACON LG Aimers 9th - Arm C (20-Seed GPU Trackman 3D Physical Ensemble)
# Execution Target: Google Colab (Runtime -> Change runtime type -> GPU T4/A100)
# ==============================================================================

import os, sys, time, zipfile, gc
import numpy as np, pandas as pd

# 1. Check GPU
import torch
assert torch.cuda.is_available(), "GPU가 활성화되지 않았습니다! [런타임] -> [런타임 유형 변경]에서 GPU를 선택해주세요."
print(f"✓ GPU 연결 확인: {torch.cuda.get_device_name(0)}")

# 2. Upload / Extract Data
if not os.path.exists('data/train.csv'):
    if not os.path.exists('colab_data_fast.zip'):
        from google.colab import files
        print("\n[안내] 맥북 다운로드 폴더의 'colab_data_fast.zip' (155MB) 파일을 선택해 업로드해주세요:")
        uploaded = files.upload()
    
    print("\n데이터 압축 해제 중...")
    with zipfile.ZipFile('colab_data_fast.zip', 'r') as z:
        z.extractall('data')
    print("✓ 데이터 압축 해제 완료!")

# 3. Install CatBoost
!pip install -q catboost

from catboost import CatBoostClassifier

print("\n--- 데이터 로드 및 피처 엔지니어링 시작 ---")
t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"학습 데이터: {len(train_df)} 행 | 테스트 데이터: {len(test_df)} 행")

# Feature Builder
def build_features(df_fit, df_target):
    g_mean = df_fit['control_success'].mean()
    C = 50.0
    
    p_agg = df_fit.groupby('pitcher_id')['control_success'].agg(['count', 'mean'])
    p_eb = ((p_agg['count'] * p_agg['mean'] + C * g_mean) / (p_agg['count'] + C)).to_dict()
    
    b_agg = df_fit.groupby('batter_id')['control_success'].agg(['count', 'mean'])
    b_eb = ((b_agg['count'] * b_agg['mean'] + C * g_mean) / (b_agg['count'] + C)).to_dict()
    
    feats = pd.DataFrame()
    feats['inning'] = df_target['inning'].fillna(1).astype(np.float32)
    feats['outs'] = df_target['outs_before'].fillna(0).astype(np.float32)
    feats['balls'] = df_target['balls_before'].fillna(0).astype(np.float32)
    feats['strikes'] = df_target['strikes_before'].fillna(0).astype(np.float32)
    feats['count_diff'] = feats['strikes'] - feats['balls']
    feats['score_diff'] = df_target['score_diff'].fillna(0).astype(np.float32) if 'score_diff' in df_target.columns else 0.0
    
    feats['top_bottom'] = (df_target['top_bottom'] == 'T').astype(np.float32)
    feats['is_futures'] = (df_target['game_type'] == 'F').astype(np.float32)
    
    feats['pitcher_eb'] = df_target['pitcher_id'].map(p_eb).fillna(g_mean).astype(np.float32)
    feats['batter_eb'] = df_target['batter_id'].map(b_eb).fillna(g_mean).astype(np.float32)
    feats['eb_diff'] = feats['pitcher_eb'] - feats['batter_eb']
    
    asof_cols = ['asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_middle_rate',
                 'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
                 'asof_pitcher_prev5_game_success_rate',
                 'asof_batter_n', 'asof_batter_success_rate', 'asof_batter_middle_rate']
    for c in asof_cols:
        if c in df_target.columns:
            feats[c] = df_target[c].fillna(df_target[c].median()).astype(np.float32)
            
    feats = feats.fillna(0.0)
    return feats

print("피처 매트릭스 생성 중...")
X_train = build_features(train_df, train_df)
y_train = train_df['control_success'].values
X_test = build_features(train_df, test_df)
print(f"✓ 피처 생성 완료: {X_train.shape[1]}개 피처 ({time.time()-t0:.1f}초)")

# 4. Train 20-Seed CatBoost GPU Ensemble
os.makedirs('arm_c_models', exist_ok=True)
SEEDS = [7, 42, 123, 202, 365, 777, 999, 1024, 2024, 2025,
         3141, 4096, 5555, 7890, 8888, 9999, 12345, 31415, 65536, 8675309]

test_preds = np.zeros(len(test_df), dtype=np.float64)

print(f"\n--- CatBoost GPU 20-Seed 배깅 학습 시작 (총 20개 시드) ---")
for idx, s in enumerate(SEEDS, 1):
    t_seed = time.time()
    cb = CatBoostClassifier(
        iterations=1300,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=64,
        task_type='GPU',
        loss_function='Logloss',
        random_seed=s,
        verbose=False
    )
    cb.fit(X_train, y_train)
    
    p_test = cb.predict_proba(X_test)[:, 1]
    test_preds += p_test / len(SEEDS)
    
    model_file = f"arm_c_models/cb_gpu_seed{s}.cbm"
    cb.save_model(model_file)
    print(f"  [{idx:02d}/20] Seed {s:7d} 학습 완료 ({time.time()-t_seed:.1f}초) -> {model_file}")

np.save('arm_c_models/test_preds_arm_c.npy', test_preds)
print(f"\n✓ 20개 시드 전체 학습 완료! 평균 테스트 예측 확률: {test_preds.mean():.4f}")

# 5. Package & Download Artifacts
print("\n산출물 압축 중...")
with zipfile.ZipFile('arm_c_models.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir('arm_c_models'):
        z.write(os.path.join('arm_c_models', f), arcname=f)

print(f"✓ arm_c_models.zip 생성 완료 ({os.path.getsize('arm_c_models.zip')/(1024*1024):.2f} MB)")
print("\n[안내] 맥북으로 다운로드를 시작합니다...")
from google.colab import files
files.download('arm_c_models.zip')
print("🎉 모든 작업이 성공적으로 완료되었습니다!")
