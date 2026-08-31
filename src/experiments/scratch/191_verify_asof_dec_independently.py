"""
191_verify_asof_dec_independently.py
agent2가 자체 커스텀 파이프라인으로 주장한 asof_dec 피처셋의 +360.97점 개선을,
이 프로젝트의 공식/검증된 평가 하네스(core/eval_utils.py의 run_standard_sota_evaluation)
로 완전히 독립적으로 재현. agent2의 AsofDecomposer2를 extra_feature_fn으로 감싸서
투입 — agent2 자신의 코드와 다른 평가 경로를 타므로 구현 버그를 걸러낼 수 있음.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation
from agent2_asof_decomp2 import AsofDecomposer2

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1  # 프로젝트 컨벤션: fold_max_season=train마지막해, val=그 다음해
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f)
    val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index
    val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


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

log("=== 191: agent2의 asof_dec 피처셋, 공식 하네스로 독립 재현 (2-seed 스크리닝) ===")
t0 = time.time()
r_base = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                       weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS)
log(f"[baseline(공식하네스)] 2-seed skill={r_base['mean_fold_skill']:.2f} "
    f"folds={[round(fd['skill_k'],2) for fd in r_base['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

t0 = time.time()
r_dec = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                      extra_feature_fn=add_asof_dec_features)
log(f"[+asof_dec(공식하네스)] 2-seed skill={r_dec['mean_fold_skill']:.2f} "
    f"(delta vs baseline={r_dec['mean_fold_skill']-r_base['mean_fold_skill']:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r_dec['fold_details']]} ({(time.time()-t0)/60:.1f}min)")

log(f"\nouter(fold3=2024) 단독: baseline={r_base['fold_details'][2]['skill_k']:.2f} "
    f"asof_dec={r_dec['fold_details'][2]['skill_k']:.2f} "
    f"delta={r_dec['fold_details'][2]['skill_k']-r_base['fold_details'][2]['skill_k']:+.2f}")

result = {
    'baseline_screen': r_base['mean_fold_skill'], 'baseline_fold_details': r_base['fold_details'],
    'asof_dec_screen': r_dec['mean_fold_skill'], 'asof_dec_fold_details': r_dec['fold_details'],
    'delta_screen': r_dec['mean_fold_skill'] - r_base['mean_fold_skill'],
}

if result['delta_screen'] > 30.0:
    log("\n2-seed에서도 큰 개선 확인됨 -> 5-seed 정식 확인")
    t0 = time.time()
    r_dec5 = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_asof_dec_features)
    log(f"[FULL 5-seed +asof_dec] skill={r_dec5['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_dec5['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"outer(fold3)={r_dec5['fold_details'][2]['skill_k']:.2f} "
        f"folds={[round(fd['skill_k'],2) for fd in r_dec5['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_dec5['mean_fold_skill']
    result['full_fold_details'] = r_dec5['fold_details']
else:
    log(f"\n2-seed에서 재현 안 됨(delta={result['delta_screen']:+.2f}) — agent2 자체 파이프라인에 뭔가 다른 점이 있는 것으로 의심됨")

with open('/tmp/191_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 191 DONE ===")
