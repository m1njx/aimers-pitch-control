"""
166_ingame_sequence_features.py
아이디어 5(변형): "이 경기 내에서 같은 투수가 방금 던진 직전 공들"의 결과 시퀀스를
피처화(itg_last1/3/5_success_rate, itg_pitch_count_so_far)해 GBDT에 추가.
기존 asof_* 피처는 전부 "이전 경기들" 집계이고, "이번 경기 내 최근 투구 흐름"은
지금까지 이 프로젝트에서 한 번도 시도되지 않은 각도.

리키지 방지: game 근사 키 = (pitcher_id, season, game_month, game_dayofweek).
같은 fold(df_tr_f 또는 df_val_f) 안에서 row_num(=row_id 숫자 부분) 기준으로
자기보다 이전 행만 사용 (shift 기반, strictly prior).
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


def add_ingame_sequence_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        d = df_src.copy()
        d['row_num'] = d['row_id'].str.extract(r'(\d+)').astype(int)
        d['game_key'] = (d['pitcher_id'].astype(str) + '_' + d['season'].astype(str) + '_' +
                         d['game_month'].astype(str) + '_' + d['game_dayofweek'].astype(str))
        d = d.sort_values('row_num')
        g = d.groupby('game_key')['control_success']
        shifted = g.shift(1)  # strictly prior pitch, current row excluded
        d['itg_last1_success'] = shifted.fillna(0.5)
        d['itg_last3_success_rate'] = g.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
        d['itg_last5_success_rate'] = g.apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
        d['itg_last3_success_rate'] = d['itg_last3_success_rate'].fillna(0.5)
        d['itg_last5_success_rate'] = d['itg_last5_success_rate'].fillna(0.5)
        d['itg_pitch_count_so_far'] = g.cumcount()

        feat_cols = ['itg_last1_success', 'itg_last3_success_rate', 'itg_last5_success_rate', 'itg_pitch_count_so_far']
        # d retains df_src's original index labels through sort_values/groupby, so a plain reindex realigns correctly
        for c in feat_cols:
            X_dst[c] = d[c].reindex(X_dst.index).astype(np.float32).values
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

log("=== 166: In-game recent-pitch-sequence features (2-seed screen) ===")
t0 = time.time()
r_base = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                       weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
log(f"[baseline] 2-seed skill={r_base['mean_fold_skill']:.2f} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_seq = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                      extra_feature_fn=add_ingame_sequence_features)
log(f"[+ingame_seq] 2-seed skill={r_seq['mean_fold_skill']:.2f} "
    f"(delta vs baseline_screen={r_seq['mean_fold_skill']-r_base['mean_fold_skill']:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r_seq['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

result = {
    'baseline_screen': r_base['mean_fold_skill'],
    'ingame_seq_screen': r_seq['mean_fold_skill'],
    'delta': r_seq['mean_fold_skill'] - r_base['mean_fold_skill'],
    'fold_details_seq': r_seq['fold_details'],
}

if result['delta'] > 10.0:
    log("\nDelta > +10, promoting to full 5-seed confirm ...")
    t0 = time.time()
    r_full = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_ingame_sequence_features)
    log(f"[FULL 5-seed +ingame_seq] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\nDelta ({result['delta']:+.2f}) not promising enough (<=+10), skipping full confirm.")

with open('/tmp/166_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 166 DONE ===")
