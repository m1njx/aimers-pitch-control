"""
185_outer_only_rigorous_verification.py
사용자 지시(작업1~3)의 핵심 검증. 세 가지를 5-seed로 정식 확인:
(A) recency: 진짜 inner-only 기준으로 재선정된 decay=0.95를 5-seed로 확인
(B) 노이즈 프로브: classification baseline(843.69 구성)을 전혀 다른 5-seed 세트로
    재실행해서 outer(fold3) 단독 지표가 시드 선택만으로 얼마나 흔들리는지 추정
    (noise floor(±15.10)는 3-fold 평균 기준이라 단일 fold는 이보다 훨씬 노이즈가 클 수 있음)
(C) L2 + recency(진짜 inner-선택 decay=0.95) 결합이 outer에서 어떤지 확인
    (기존 179번은 decay=0.7이라는, outer로 오염된 값을 썼음)
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
NOISE_PROBE_SEEDS = [11, 222, 3333, 44444, 555555]


def run_classification(df_train, seeds, decay=None):
    mp_lgb = {'colsample_bytree': 0.7, 'subsample': 0.7}
    mp_xgb = {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
    folds = get_cv_folds(df_train)
    fold_details = []

    for k, fold in enumerate(folds):
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
        as_of = fold.fold_max_season
        prep = PitchPreprocessor()
        prep.fit(df_tr_f, as_of_season=as_of, is_final=False)
        X_tr_f = prep.transform(df_tr_f)
        X_val_f = prep.transform(df_val_f)
        for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
            b_str = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                     (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                     (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
            c_str = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                     df_src['strikes_before'].fillna(0).astype(int).astype(str))
            X_dst['count_x_base'] = (c_str + '_' + b_str)
        cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
        X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
        X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
        y_tr_f = df_tr_f[config.TARGET_COL].values
        y_val_f = df_val_f[config.TARGET_COL].values

        if decay is not None:
            season_gap = (as_of - df_tr_f['season']).clip(lower=0).values
            sw = np.power(decay, season_gap).astype(np.float64)
            sw = sw / sw.mean()
        else:
            sw = None

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        bag_p_lgb = np.zeros(len(fold.val_idx))
        bag_p_cb = np.zeros(len(fold.val_idx))
        bag_p_xgb = np.zeros(len(fold.val_idx))
        for seed in seeds:
            lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                               random_state=seed, verbosity=-1, n_jobs=-1)
            lgb_params.update(mp_lgb)
            m_lgb = lgb.LGBMClassifier(**lgb_params)
            m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx, sample_weight=sw)
            bag_p_lgb += np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

            X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
                X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
            for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
                X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
                X_val_cb[c] = X_val_cb[c].astype(np.float32)
            m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                       random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
            m_cb.fit(X_tr_cb, y_tr_f, sample_weight=sw)
            bag_p_cb += np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            xgb_params = dict(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=seed,
                               n_jobs=-1, eval_metric='logloss')
            xgb_params.update(mp_xgb)
            m_xgb = xgb.XGBClassifier(**xgb_params)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f, sample_weight=sw)
            bag_p_xgb += np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

        n_seeds = len(seeds)
        p_ens = np.clip(w_lgb * (bag_p_lgb / n_seeds) + w_cb * (bag_p_cb / n_seeds) + w_xgb * (bag_p_xgb / n_seeds), 1e-6, 1 - 1e-6)
        sk, _, _, _ = calc_brier_skill_score(y_val_f, p_ens)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk})
    return {'mean_fold_skill': evaluate_fold_skills(fold_details), 'fold_details': fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)

log("=== 185(A): recency decay=0.95 (진짜 inner-only 재선정값) 5-seed 확인 ===")
t0 = time.time()
r_a = run_classification(df_train, FULL_SEEDS, decay=0.95)
log(f"[decay=0.95, 5-seed] skill={r_a['mean_fold_skill']:.2f} folds={[round(fd['skill_k'],2) for fd in r_a['fold_details']]} "
    f"({(time.time()-t0)/60:.1f}min)")
outer_decay095 = r_a['fold_details'][2]['skill_k']

log("\n=== 185(B): 노이즈 프로브 - classification baseline을 다른 5-seed 세트로 재실행 ===")
t0 = time.time()
r_b = run_classification(df_train, NOISE_PROBE_SEEDS, decay=None)
log(f"[baseline, 대체 5-seed {NOISE_PROBE_SEEDS}] skill={r_b['mean_fold_skill']:.2f} "
    f"folds={[round(fd['skill_k'],2) for fd in r_b['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
log(f"기존(공식) baseline outer(fold3, seeds={FULL_SEEDS}): 616.92")
log(f"대체 시드 baseline outer(fold3): {r_b['fold_details'][2]['skill_k']:.2f}")
log(f"-> 단순히 시드셋을 바꾸는 것만으로도 outer(fold3) 단일 지표가 이 정도 흔들림 = 단일폴드 노이즈 추정치")
noise_probe_delta = r_b['fold_details'][2]['skill_k'] - 616.92

result = {
    'decay_0.95_5seed': r_a['mean_fold_skill'],
    'decay_0.95_fold_details': r_a['fold_details'],
    'decay_0.95_outer': outer_decay095,
    'baseline_outer_official': 616.92,
    'noise_probe_alt_seed_baseline_fold_details': r_b['fold_details'],
    'noise_probe_outer_delta': noise_probe_delta,
}
with open('/tmp/185_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log(f"\n=== 185 DONE ===")
log(f"요약: decay=0.95(진짜 inner-선택) outer={outer_decay095:.2f} vs baseline outer=616.92, delta={outer_decay095-616.92:+.2f}")
log(f"참고: 단일 시드셋 교체만으로도 노이즈가 {noise_probe_delta:+.2f}점 발생 — 단일폴드 비교의 노이즈 폭이 매우 클 수 있음")
