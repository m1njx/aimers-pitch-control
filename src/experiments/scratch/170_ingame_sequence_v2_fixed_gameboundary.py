"""
170_ingame_sequence_v2_fixed_gameboundary.py
166번(경기 내 최근 투구 시퀀스 피처)의 재시도. 166번은 game 근사 키가
(투수+시즌+월+요일)라서 실제로 200개 넘는 투구가 한 그룹에 섞이는 경우가
317개나 있었음(한 경기치고 너무 많음 = 서로 다른 날 경기가 뒤섞임) -> -139.55점 REJECT.

수정: 같은 투수의 연속 투구를 row_num(=row_id 숫자부분) 기준 정렬 후,
간격(gap) > 250이면 새 세션(진짜 경기)으로 판단. 검증 결과 이 방식은
세션 크기 분포가 훨씬 현실적(평균 32개, 200개 넘는 세션 1개뿐, inning 역행
발생률 0.2%)이라 실제 "이 경기 내" 정의에 훨씬 가깝다.
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


def add_ingame_sequence_features_v2(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        d = df_src.copy()
        d['row_num'] = d['row_id'].str.extract(r'(\d+)').astype(int)
        d = d.sort_values('row_num')
        gap = d.groupby('pitcher_id')['row_num'].diff()
        new_session = gap.isna() | (gap > GAP_THRESH)
        d['session_id'] = new_session.groupby(d['pitcher_id']).cumsum()
        d['game_key'] = d['pitcher_id'].astype(str) + '_' + d['session_id'].astype(str)

        g = d.groupby('game_key')['control_success']
        shifted = g.shift(1)
        d['itg_last1_success'] = shifted.fillna(0.5)
        d['itg_last3_success_rate'] = g.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
        d['itg_last5_success_rate'] = g.apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
        d['itg_last3_success_rate'] = d['itg_last3_success_rate'].fillna(0.5)
        d['itg_last5_success_rate'] = d['itg_last5_success_rate'].fillna(0.5)
        d['itg_pitch_count_so_far'] = g.cumcount()

        feat_cols = ['itg_last1_success', 'itg_last3_success_rate', 'itg_last5_success_rate', 'itg_pitch_count_so_far']
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

log("=== 170: In-game sequence features v2 (row_num-gap 기반 세션 경계 수정판, 2-seed 스크리닝) ===")
t0 = time.time()
r_base = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                       weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
log(f"[baseline] 2-seed skill={r_base['mean_fold_skill']:.2f} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_seq = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                      extra_feature_fn=add_ingame_sequence_features_v2)
log(f"[+ingame_seq_v2] 2-seed skill={r_seq['mean_fold_skill']:.2f} "
    f"(delta vs baseline_screen={r_seq['mean_fold_skill']-r_base['mean_fold_skill']:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r_seq['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

result = {
    'baseline_screen': r_base['mean_fold_skill'],
    'ingame_seq_v2_screen': r_seq['mean_fold_skill'],
    'delta': r_seq['mean_fold_skill'] - r_base['mean_fold_skill'],
    'fold_details_seq': r_seq['fold_details'],
}

if result['delta'] > 5.0:
    log("\nDelta > +5, promoting to full 5-seed confirm ...")
    t0 = time.time()
    r_full = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_ingame_sequence_features_v2)
    log(f"[FULL 5-seed +ingame_seq_v2] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\nDelta ({result['delta']:+.2f}) not promising enough (<=+5), skipping full confirm.")

with open('/tmp/170_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 170 DONE ===")
