"""
compare_v7_v9_gbdt_predictions.py
v7(메인환경, numpy2.x, xgboost3.4.0)와 v9(venv311, numpy1.26.4, xgboost3.2.0)의
GBDT 모델이 같은 입력에 대해 얼마나 다른 예측을 내는지 직접 비교.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config

df_sample = pd.read_csv(config.TRAIN_PATH).sample(n=2000, random_state=0)

for version, model_dir in [('v7 (main env, numpy2.x)', 'work/submit_v7/model'),
                            ('v9 (venv311, numpy1.26.4)', 'work/submit_v9/model')]:
    print(f"\n=== {version} ===")
    prep = joblib.load(f'{model_dir}/preprocessor_artifacts.pkl')
    prep.trackman_builder = joblib.load(f'{model_dir}/trackman_artifacts.pkl')
    X = prep.transform(df_sample)

    base_str = ((df_sample['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_sample['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_sample['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (df_sample['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_sample['strikes_before'].fillna(0).astype(int).astype(str))
    count_x_base_raw = (cc_str + '_' + base_str)
    cat_map = getattr(prep, 'count_x_base_map', {})
    X['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

    cat_cols = [c for c in X.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand',
                'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]

    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].fillna(-1).astype(int).astype(str)
    for c in [col for col in X_cb.columns if col not in cat_cols]:
        X_cb[c] = X_cb[c].astype(np.float32)

    X_xgb = X.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].astype('category').cat.codes.astype(np.float32)
    X_xgb = X_xgb.astype(np.float32)

    SEEDS = [7, 123, 2025, 31415, 8675309]
    S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006
    p_lgb_sum = np.zeros(len(df_sample))
    p_cb_sum = np.zeros(len(df_sample))
    p_xgb_sum = np.zeros(len(df_sample))
    for seed in SEEDS:
        m_lgb = lgb.Booster(model_file=f'{model_dir}/lgbm_model_seed{seed}.txt')
        p_lgb_sum += m_lgb.predict(X)
        m_cb = CatBoostClassifier()
        m_cb.load_model(f'{model_dir}/catboost_model_seed{seed}.cbm')
        p_cb_sum += m_cb.predict_proba(X_cb)[:, 1]
        m_xgb = xgb.XGBClassifier()
        m_xgb.load_model(f'{model_dir}/xgb_model_seed{seed}.json')
        p_xgb_sum += m_xgb.predict_proba(X_xgb)[:, 1]

    n_seeds = len(SEEDS)
    p_lgb = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
    p_cb = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
    p_xgb = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)
    p_ens = np.clip(0.15 * p_lgb + 0.75 * p_cb + 0.10 * p_xgb, 1e-6, 1 - 1e-6)

    globals()[f'p_ens_{version[:2]}'] = p_ens
    globals()[f'p_lgb_{version[:2]}'] = p_lgb
    globals()[f'p_cb_{version[:2]}'] = p_cb
    globals()[f'p_xgb_{version[:2]}'] = p_xgb
    print(f"  Ensemble pred: mean={p_ens.mean():.6f} std={p_ens.std():.6f}")
    print(f"  LGBM: mean={p_lgb.mean():.6f}  CatBoost: mean={p_cb.mean():.6f}  XGBoost: mean={p_xgb.mean():.6f}")

print("\n=== DIRECT COMPARISON (same 2000-row sample) ===")
diff_ens = np.abs(p_ens_v7 - p_ens_v9)
diff_lgb = np.abs(p_lgb_v7 - p_lgb_v9)
diff_cb = np.abs(p_cb_v7 - p_cb_v9)
diff_xgb = np.abs(p_xgb_v7 - p_xgb_v9)
print(f"Ensemble abs diff:  mean={diff_ens.mean():.6f} max={diff_ens.max():.6f} corr={np.corrcoef(p_ens_v7, p_ens_v9)[0,1]:.6f}")
print(f"LGBM abs diff:      mean={diff_lgb.mean():.6f} max={diff_lgb.max():.6f} corr={np.corrcoef(p_lgb_v7, p_lgb_v9)[0,1]:.6f}")
print(f"CatBoost abs diff:  mean={diff_cb.mean():.6f} max={diff_cb.max():.6f} corr={np.corrcoef(p_cb_v7, p_cb_v9)[0,1]:.6f}")
print(f"XGBoost abs diff:   mean={diff_xgb.mean():.6f} max={diff_xgb.max():.6f} corr={np.corrcoef(p_xgb_v7, p_xgb_v9)[0,1]:.6f}")
