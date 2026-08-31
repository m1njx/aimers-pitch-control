"""
181_rule_change_era_feature.py
174번(recency decay=0.7)이 크게 통한 이유의 실제 도메인 원인 가설: MLB는
2019~2024 사이 투수 제구에 직접 영향을 주는 실제 규칙 변화가 있었음
(2021년 중반 이물질/끈적이 단속, 2023년 시즌 시작 피치클락 도입 등).
매끄러운 decay 대신, 이 규칙변화 시점을 범주형 "era" 피처로 명시적으로 넣어
GBDT가 era별 기저 패턴을 직접 학습하게 함 - decay보다 더 정교한 신호일 수 있음.
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


def era_of_vectorized(df_src):
    s, m = df_src['season'], df_src['game_month']
    era = pd.Series('2023plus_pitchclock', index=df_src.index)
    era[s <= 2020] = 'pre2021'
    era[(s == 2021) & (m < 6)] = '2021_pre_crackdown'
    era[(s == 2021) & (m >= 6)] = '2021_post_crackdown'
    era[s == 2022] = '2022'
    return era


def add_era_feature(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        X_dst['rule_era_raw'] = era_of_vectorized(df_src)

    era_map = {v: i for i, v in enumerate(X_tr_f['rule_era_raw'].unique())}
    X_tr_f['rule_era'] = X_tr_f['rule_era_raw'].map(era_map).fillna(-1).astype(int)
    X_val_f['rule_era'] = X_val_f['rule_era_raw'].map(era_map).fillna(-1).astype(int)
    X_tr_f.drop(columns=['rule_era_raw'], inplace=True)
    X_val_f.drop(columns=['rule_era_raw'], inplace=True)
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

log("=== 181: MLB 규칙변화 era 범주형 피처, 2-seed 스크리닝 (classification baseline 기준) ===")
t0 = time.time()
r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                  weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                  extra_feature_fn=add_era_feature)
dt = (time.time() - t0) / 60
log(f"[+rule_era] 2-seed skill={r['mean_fold_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")

result = {'screen_skill': r['mean_fold_skill'], 'delta': r['mean_fold_skill'] - BASELINE_REF,
          'fold_details': r['fold_details']}

if result['delta'] > -10.0:
    log("\n노이즈 바닥 근접/양수 -> 5-seed 정식 확인")
    t0 = time.time()
    r_full = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_era_feature)
    log(f"[FULL 5-seed +rule_era] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\nDelta ({result['delta']:+.2f}) 너무 나쁨, 5-seed 생략")

with open('/tmp/181_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 181 DONE ===")
