# =============================================================================
#  Arm C 배포용 재학습 — 전 시즌(2019~2024) 학습  [Google Colab / T4 GPU]
#
#  왜: 현재 출하된 C 는 `train<2024` 라 2025 예측 시 **격차 2년**이고,
#      2년 격차에서는 `w_C = 0`(폴드내 오라클조차) 임이 측정됐다 → v33 LB 1078.03 (v29 1082 대비 −3.97).
#      2024 까지 학습하면 격차가 **1년**이 되고, 1년 격차 기여는 이미 측정돼 있다(+33.80 @2024, W_C=0.10 에서 +22.74).
#
#  ⚠️ 전 시즌 학습이므로 **정직한 검증 폴드가 없다.** 가치 추정은 위 1년 격차 실측값을 쓰고,
#     이 스크립트로는 검증하지 않는다(할 수 없다). 여기서 하는 것은 배포물 생성뿐이다.
#
#  부수효과: EB 를 전 시즌으로 적합하므로 기존 `C/model/artifacts.json`(g=0.523766)과
#           학습이 **일치**한다 — 기존 train/serve 불일치가 해소된다.
#
#  런타임: 런타임 유형 → 하드웨어 가속기 → **T4 GPU**
# =============================================================================

# ---- [셀 1] ----------------------------------------------------------------
# !pip -q install catboost
# from google.colab import files
# files.upload()          # armc_slim.csv.gz

CSV = 'armc_slim.csv.gz'

# ---- [셀 2] ----------------------------------------------------------------
import numpy as np, pandas as pd, json, time, os, shutil
from catboost import CatBoostClassifier

SEEDS = [7,42,123,202,365,777,999,1024,2024,2025,
         3141,4096,5555,7890,8888,9999,12345,31415,65536,8675309]
# 원본과 동일 (변경 금지 — 1년 격차 실측값이 이 설정에서 나왔다)
PARAMS = dict(iterations=1300, learning_rate=0.03, depth=7, l2_leaf_reg=64,
              loss_function='Logloss', task_type='GPU', devices='0', verbose=False)

FN = ['inning','outs','balls','strikes','count_diff','score_diff','top_bottom','is_futures',
      'pitcher_eb','batter_eb','eb_diff','asof_pitcher_n','asof_pitcher_success_rate',
      'asof_pitcher_middle_rate','asof_pitcher_prev1_game_success_rate',
      'asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate',
      'asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate']
ASOF = FN[11:]

df = pd.read_csv(CSV)
print(f'loaded {len(df):,} rows, seasons {sorted(df.season.unique())}')

# ---- 전 시즌으로 EB 적합 (= 기존 artifacts.json 과 같은 방식) ---------------
C_SMOOTH = 50.0
g_mean = float(df.control_success.mean())
pa = df.groupby('pitcher_id')['control_success'].agg(['count','mean'])
p_eb = ((pa['count']*pa['mean'] + C_SMOOTH*g_mean)/(pa['count'] + C_SMOOTH))
ba = df.groupby('batter_id')['control_success'].agg(['count','mean'])
b_eb = ((ba['count']*ba['mean'] + C_SMOOTH*g_mean)/(ba['count'] + C_SMOOTH))
medians = {c: float(df[c].median()) for c in ASOF}
print(f'g_mean={g_mean:.6f}  pitcher={len(p_eb)}  batter={len(b_eb)}')

def build(t):
    f = pd.DataFrame(index=t.index)
    f['inning']   = t['inning'].fillna(1).astype(np.float32)
    f['outs']     = t['outs_before'].fillna(0).astype(np.float32)
    f['balls']    = t['balls_before'].fillna(0).astype(np.float32)
    f['strikes']  = t['strikes_before'].fillna(0).astype(np.float32)
    f['count_diff'] = f['strikes'] - f['balls']
    f['score_diff'] = t['score_diff'].fillna(0).astype(np.float32) \
                      if 'score_diff' in t.columns else np.float32(0.0)
    f['top_bottom'] = (t['top_bottom'] == 'T').astype(np.float32)
    f['is_futures'] = (t['game_type'] == 'F').astype(np.float32)
    f['pitcher_eb'] = t['pitcher_id'].map(p_eb).fillna(g_mean).astype(np.float32)
    f['batter_eb']  = t['batter_id'].map(b_eb).fillna(g_mean).astype(np.float32)
    f['eb_diff']    = f['pitcher_eb'] - f['batter_eb']
    for c in ASOF:
        f[c] = t[c].fillna(medians[c]).astype(np.float32)
    return f[FN].fillna(0.0)

X = build(df)
y = df.control_success.values
assert list(X.columns) == FN, '피처 순서 불일치'
print(f'X {X.shape}  (전 시즌 학습)')

os.makedirs('out_model', exist_ok=True)
t0 = time.time()
for i, s in enumerate(SEEDS, 1):
    m = CatBoostClassifier(random_seed=s, **PARAMS)
    m.fit(X, y)
    m.save_model(f'out_model/cb_gpu_seed{s}.cbm')
    print(f'  [{i:2d}/20] seed {s:>8} done ({time.time()-t0:.0f}s)', flush=True)

# 추론 스크립트가 읽는 아티팩트도 같은 적합으로 새로 굽는다 (일관성 보장)
json.dump({'g_mean': g_mean,
           'pitcher_eb': {str(k): float(v) for k, v in p_eb.items()},
           'batter_eb':  {str(k): float(v) for k, v in b_eb.items()},
           'medians': medians},
          open('out_model/artifacts.json', 'w'))

# 새너티: 피처명·개수가 기존과 같은지
mm = CatBoostClassifier(); mm.load_model(f'out_model/cb_gpu_seed{SEEDS[0]}.cbm')
print('\n피처명 일치:', list(mm.feature_names_) == FN, f'({len(FN)}개)')
print('트리 수:', mm.tree_count_)
built = ['inning','outs','balls','strikes','count_diff','score_diff','top_bottom','is_futures',
         'pitcher_eb','batter_eb','eb_diff'] + list(medians.keys())
print('artifacts medians 순서 일치:', built == FN)

shutil.make_archive('armc_full', 'zip', 'out_model')
print('\n->', os.path.getsize('armc_full.zip')/1e6, 'MB')

# ---- [셀 3] ----------------------------------------------------------------
# from google.colab import files
# files.download('armc_full.zip')
