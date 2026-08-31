"""
precompute_gbdt_reference_oof.py
4개 DL 트랙이 공통으로 쓸 GBDT 3종 앙상블 OOF 예측(참조용, 상관관계/앙상블가치 분석에만 사용)을
한 번만 계산해서 저장. 공식 SSOT(843.69점) 자체를 재계산하는 게 목적이 아니므로 단일 시드(seed=7,
42-제외 5-seed 중 첫번째)로 빠르게 계산.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd

import config
from core.eval_utils import run_standard_sota_evaluation

df_train = pd.read_csv(config.TRAIN_PATH)
sota_mp = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
sota_weights = (0.15, 0.75, 0.10)
sota_shifts = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}

t0 = time.time()
r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                  weights=sota_weights, shifts=sota_shifts, random_seeds=[7])
val_idx = np.array(r['val_indices'])
p_lgb = np.clip(r['oof_preds_lgb'][val_idx] + sota_shifts['lgb'], 1e-6, 1 - 1e-6)
p_cb = np.clip(r['oof_preds_cb'][val_idx] + sota_shifts['cb'], 1e-6, 1 - 1e-6)
p_xgb = np.clip(r['oof_preds_xgb'][val_idx] + sota_shifts['xgb'], 1e-6, 1 - 1e-6)
p_ens = np.clip(0.15 * p_lgb + 0.75 * p_cb + 0.10 * p_xgb, 1e-6, 1 - 1e-6)

np.savez('/tmp/gbdt_reference_oof.npz', val_idx=val_idx, p_ens=p_ens, p_lgb=p_lgb, p_cb=p_cb, p_xgb=p_xgb,
         skill=r['mean_fold_skill'], brier=r['overall_raw_brier'])
print(f"Saved GBDT reference OOF (seed=7 single-seed, skill={r['mean_fold_skill']:.2f}, "
      f"brier={r['overall_raw_brier']:.6f}) in {(time.time()-t0)/60:.1f}min")
