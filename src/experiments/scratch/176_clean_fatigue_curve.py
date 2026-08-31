"""
176_clean_fatigue_curve.py
"완전히 새로운 판" 2번: 166/170번에서 실패한 "경기 내 최근 투구 시퀀스" 피처 중,
노이즈가 컸던 "직전 투구 성공여부" 계열은 다 빼고, 깨끗한 신호인
"이번 경기 몇 번째 투구인지"(itg_pitch_count_so_far)만 단독으로 사용.
170번에서 검증된 row_num-gap 기반 정확한 경기 경계 탐지 재사용.
GBDT가 이 하나의 연속형 피처에서 비선형(피로도 곡선) 패턴을 스스로 찾을 수 있음.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

GAP_THRESH = 250


def add_clean_fatigue_feature(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        d = df_src.copy()
        d['row_num'] = d['row_id'].str.extract(r'(\d+)').astype(int)
        d = d.sort_values('row_num')
        gap = d.groupby('pitcher_id')['row_num'].diff()
        new_session = gap.isna() | (gap > GAP_THRESH)
        d['session_id'] = new_session.groupby(d['pitcher_id']).cumsum()
        d['game_key'] = d['pitcher_id'].astype(str) + '_' + d['session_id'].astype(str)
        d['itg_pitch_count_so_far'] = d.groupby('game_key').cumcount()
        X_dst['itg_pitch_count_so_far'] = d['itg_pitch_count_so_far'].reindex(X_dst.index).astype(np.float32).values
    return X_tr_f, X_val_f


df_train = pd.read_csv(config.TRAIN_PATH)
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

log("=== 176: 정제된 경기내 피로도 곡선(투구 수만, 결과이력 없음) 2-seed 스크리닝 ===")
t0 = time.time()
r_seq = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                      extra_feature_fn=add_clean_fatigue_feature)
log(f"[+clean_fatigue] 2-seed skill={r_seq['mean_fold_skill']:.2f} "
    f"(delta vs {BASELINE_REF}={r_seq['mean_fold_skill']-BASELINE_REF:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r_seq['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

result = {'screen_skill': r_seq['mean_fold_skill'], 'delta': r_seq['mean_fold_skill'] - BASELINE_REF,
          'fold_details': r_seq['fold_details']}

if result['delta'] > -5.0:
    log("\n노이즈 바닥 근접/양수 -> 5-seed 정식 확인")
    t0 = time.time()
    r_full = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_clean_fatigue_feature)
    log(f"[FULL 5-seed +clean_fatigue] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\nDelta ({result['delta']:+.2f}) 너무 나쁨, 5-seed 생략")

with open('/tmp/176_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 176 DONE ===")
