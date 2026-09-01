# =============================================================================
#  Arm C 전방검증용 재학습 — Google Colab (T4 GPU) 용
#
#  목적: 기존 arm_c_models.zip 은 `train<2024` 로 학습돼 **2024 가 유일한 정직 폴드**다.
#        그래서 재보정 상수(shift/scale)를 2024 라벨로 적합할 수밖에 없었고,
#        그 상수가 2025 로 전이되는지 시험할 방법이 없었다.
#        여기서 `train<2023` 으로 같은 arm 을 한 번 더 학습하면
#        2023·2024 두 개의 정직 폴드가 생기고,
#        "2023 에서 적합한 재보정을 2024 에 적용" 이라는 전방검증이 가능해진다.
#
#  판정(착수 전 확정):
#    2023 에서 적합한 아핀 재보정 + 블렌드 가중치를 2024 에 적용했을 때
#    3-arm 이 2-arm 을 이기면 배포 근거 성립, 지면 이 리드 종결.
#
#  런타임: 런타임 유형 → 하드웨어 가속기 → **T4 GPU** 로 먼저 바꿀 것.
#  소요: 20시드 × 1300트리, T4 기준 대략 15~30분.
# =============================================================================

# ---- [셀 1] 설치 & 업로드 ---------------------------------------------------
# !pip -q install catboost
#
# from google.colab import files
# up = files.upload()          # armc_slim.csv.gz (33.5MB) 를 선택
#
# 또는 드라이브 사용 시:
# from google.colab import drive; drive.mount('/content/drive')
# CSV = '/content/drive/MyDrive/armc_slim.csv.gz'

CSV = 'armc_slim.csv.gz'

# ---- [셀 2] 본체 ------------------------------------------------------------
import numpy as np, pandas as pd, os, json, time
from catboost import CatBoostClassifier

SEEDS = [7,42,123,202,365,777,999,1024,2024,2025,
         3141,4096,5555,7890,8888,9999,12345,31415,65536,8675309]

# 원본 arm_c_models.zip 에서 읽어낸 설정 그대로 (변경 금지 — 비교 가능성이 깨진다)
PARAMS = dict(iterations=1300, learning_rate=0.03, depth=7, l2_leaf_reg=64,
              loss_function='Logloss', task_type='GPU', devices='0', verbose=False)

FN = ['inning','outs','balls','strikes','count_diff','score_diff','top_bottom','is_futures',
      'pitcher_eb','batter_eb','eb_diff','asof_pitcher_n','asof_pitcher_success_rate',
      'asof_pitcher_middle_rate','asof_pitcher_prev1_game_success_rate',
      'asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate',
      'asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate']


def build(fit_df, trans_df):
    """원본 train_arm_c_clean.py 의 extract_features 와 동일한 20피처.
    EB 인코딩은 fit_df(=학습 시즌)에서만 적합한다."""
    g = fit_df.control_success.mean(); C = 50.0
    pa = fit_df.groupby('pitcher_id')['control_success'].agg(['count','mean'])
    pe = ((pa['count']*pa['mean'] + C*g)/(pa['count'] + C)).to_dict()
    ba = fit_df.groupby('batter_id')['control_success'].agg(['count','mean'])
    be = ((ba['count']*ba['mean'] + C*g)/(ba['count'] + C)).to_dict()

    f = pd.DataFrame(index=trans_df.index)
    f['inning']   = trans_df['inning'].fillna(1).astype(float)
    f['outs']     = trans_df['outs_before'].fillna(0).astype(float)
    f['balls']    = trans_df['balls_before'].fillna(0).astype(float)
    f['strikes']  = trans_df['strikes_before'].fillna(0).astype(float)
    f['count_diff'] = f['strikes'] - f['balls']
    # train.csv 에 score_diff 컬럼이 없다 → 원본 학습에서도 상수 0 이었다
    f['score_diff'] = trans_df['score_diff'].fillna(0).astype(float) \
                      if 'score_diff' in trans_df.columns else 0.0
    f['top_bottom'] = (trans_df['top_bottom'] == 'T').astype(float)
    f['is_futures'] = (trans_df['game_type'] == 'F').astype(float)
    f['pitcher_eb'] = trans_df['pitcher_id'].map(pe).fillna(g).astype(float)
    f['batter_eb']  = trans_df['batter_id'].map(be).fillna(g).astype(float)
    f['eb_diff']    = f['pitcher_eb'] - f['batter_eb']
    for c in FN[11:]:
        f[c] = trans_df[c].astype(float) if c in trans_df.columns else np.nan
    return f[FN].fillna(f[FN].median())


def skill(p, y):
    r = y.mean()
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


df = pd.read_csv(CSV)
print(f'loaded {len(df):,} rows, seasons {sorted(df.season.unique())}')

CUT = 2023                       # ← 핵심: 2023 을 두 번째 정직 폴드로 만든다
past = df[df.season <  CUT]
f23  = df[df.season == 2023].reset_index(drop=True)
f24  = df[df.season == 2024].reset_index(drop=True)
print(f'train<{CUT}: {len(past):,}   holdout 2023: {len(f23):,}   holdout 2024: {len(f24):,}')

X_tr = build(past, past)
y_tr = past.control_success.values
X23, y23 = build(past, f23), f23.control_success.values.astype(float)
X24, y24 = build(past, f24), f24.control_success.values.astype(float)

p23, p24, t0 = [], [], time.time()
for i, s in enumerate(SEEDS, 1):
    m = CatBoostClassifier(random_seed=s, **PARAMS)
    m.fit(X_tr, y_tr)                      # 조기종료 없음 — 평가 라벨을 쓰지 않는다
    a = m.predict_proba(X23)[:, 1]
    b = m.predict_proba(X24)[:, 1]
    p23.append(a); p24.append(b)
    print(f'  [{i:2d}/20] seed {s:>8}  2023 {skill(a, y23):8.1f}   '
          f'2024 {skill(b, y24):8.1f}   ({time.time()-t0:.0f}s)', flush=True)

P23 = np.mean(p23, 0); P24 = np.mean(p24, 0)
np.save('armc_c2023_pred2023.npy', P23)
np.save('armc_c2023_pred2024.npy', P24)
print(f'\n배깅 결과:  2023 {skill(P23, y23):.2f}   2024 {skill(P24, y24):.2f}')
print(f'mean(p) 2023 {P23.mean():.4f} (r {y23.mean():.4f}) / '
      f'2024 {P24.mean():.4f} (r {y24.mean():.4f})')

# ---- 여기서 바로 핵심 판정까지 본다 (블렌드 없이 재보정 전이만) --------------
Xf = np.stack([np.ones_like(P23), P23 - 0.5], 1)
beta = np.linalg.lstsq(Xf, y23 - P23, rcond=None)[0]      # 2023 에서 적합
Xa = np.stack([np.ones_like(P24), P24 - 0.5], 1)
P24_cal = np.clip(P24 + Xa @ beta, 1e-6, 1 - 1e-6)        # 2024 에 적용
print(f'\n[전방검증] 2023 적합 아핀 shift={beta[0]:+.5f} scale_adj={beta[1]:+.4f}')
print(f'  2024 raw        {skill(P24, y24):9.2f}')
print(f'  2024 재보정후    {skill(P24_cal, y24):9.2f}   ← 이 값이 양수 대역이면 전이 성공')

# ---- [셀 3] 내려받기 --------------------------------------------------------
# from google.colab import files
# files.download('armc_c2023_pred2023.npy')
# files.download('armc_c2023_pred2024.npy')
