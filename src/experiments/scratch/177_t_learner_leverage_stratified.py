"""
177_t_learner_leverage_stratified.py
"완전히 새로운 판" 3번: 인과추론의 T-learner(이질적 처치효과 추정) 구조를 차용.
"처치"를 고-레버리지 상황(li > 중앙값)으로 정의하고, 저/고 레버리지 그룹마다
완전히 별도의 GBDT 앙상블을 각각 학습(공유 파라미터 없음). 예측 시 각 행의
실제 li 값으로 해당 stratum 모델을 사용. 163번의 "블렌딩 가중치만 구간별 조정"과
달리 이번엔 모델 자체를 stratum별로 통째로 다시 학습 -> 저/고 레버리지 상황의
투수 행동 패턴이 근본적으로 다르다면(예: 클러치 상황에서의 접근법 변화) 단일
모델보다 유리할 수 있음. inner(2022/23)에서 분할 기준 선택 -> outer(2024) 적용.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_ensemble(X_tr, y_tr, X_val, cat_cols, seeds, mp):
    w_lgb, w_cb, w_xgb = 0.15, 0.75, 0.10
    s_lgb, s_cb, s_xgb = -0.007, -0.008, -0.006
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    bag_p_lgb = np.zeros(len(X_val))
    bag_p_cb = np.zeros(len(X_val))
    bag_p_xgb = np.zeros(len(X_val))
    for seed in seeds:
        lgb_params = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                           random_state=seed, verbosity=-1, n_jobs=-1)
        lgb_params.update(mp.get('lgb', {}))
        lgb_params['random_state'] = seed
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
        bag_p_lgb += np.clip(m_lgb.predict_proba(X_val)[:, 1] + s_lgb, 1e-6, 1 - 1e-6)

        X_tr_cb, X_val_cb = X_tr.copy(), X_val.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
        for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_val_cb[c] = X_val_cb[c].astype(np.float32)
        cb_params = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                          random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
        cb_params.update(mp.get('cb', {}))
        cb_params['random_seed'] = seed
        cb_params['cat_features'] = cat_cols
        m_cb = CatBoostClassifier(**cb_params)
        m_cb.fit(X_tr_cb, y_tr)
        bag_p_cb += np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + s_cb, 1e-6, 1 - 1e-6)

        X_tr_xgb, X_val_xgb = X_tr.copy(), X_val.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
        xgb_params = dict(n_estimators=250, max_depth=5, learning_rate=0.05, random_state=seed,
                           n_jobs=-1, eval_metric='logloss')
        xgb_params.update(mp.get('xgb', {}))
        xgb_params['random_state'] = seed
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr)
        bag_p_xgb += np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] + s_xgb, 1e-6, 1 - 1e-6)

    n_seeds = len(seeds)
    p_ens = np.clip(w_lgb * (bag_p_lgb / n_seeds) + w_cb * (bag_p_cb / n_seeds) + w_xgb * (bag_p_xgb / n_seeds),
                     1e-6, 1 - 1e-6)
    return p_ens


def run_t_learner(df_train, seeds):
    mp = {
        'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
        'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
        'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
    }
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
        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']

        li_median_tr = df_tr_f['li'].median()
        t_tr = (df_tr_f['li'].fillna(li_median_tr).values > li_median_tr)
        t_val = (df_val_f['li'].fillna(li_median_tr).values > li_median_tr)

        p_val_stratified = np.zeros(len(y_val_f))
        for stratum, mask_name in [(True, 'high-li'), (False, 'low-li')]:
            tr_mask = (t_tr == stratum)
            val_mask = (t_val == stratum)
            if val_mask.sum() == 0:
                continue
            p_stratum = train_ensemble(X_tr_f[tr_mask], y_tr_f[tr_mask], X_val_f[val_mask], cat_cols, seeds, mp)
            p_val_stratified[val_mask] = p_stratum
            log(f"  fold{k+1}({fold.val_season}) {mask_name} stratum: train_n={tr_mask.sum()}, val_n={val_mask.sum()}")

        sk, _, _, _ = calc_brier_skill_score(y_val_f, p_val_stratified)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk})
        log(f"[T-learner] Fold {k+1} ({fold.val_season}) skill={sk:.2f}")

    return {'mean_skill': evaluate_fold_skills(fold_details), 'fold_details': fold_details}


df_train = pd.read_csv(config.TRAIN_PATH)
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

log("=== 177: T-learner 레버리지 계층화 모델 (2-seed 스크리닝) ===")
t0 = time.time()
r = run_t_learner(df_train, SCREEN_SEEDS)
dt = (time.time() - t0) / 60
log(f"\n[T-learner] 2-seed skill={r['mean_skill']:.2f} (delta vs {BASELINE_REF}={r['mean_skill']-BASELINE_REF:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")

result = {'screen_skill': r['mean_skill'], 'fold_details': r['fold_details'], 'minutes': dt}

if r['mean_skill'] > BASELINE_REF - 15.0:
    log("\n노이즈 바닥 근접 -> 5-seed 정식 확인")
    t0 = time.time()
    r_full = run_t_learner(df_train, FULL_SEEDS)
    dt = (time.time() - t0) / 60
    log(f"[FULL 5-seed T-learner] skill={r_full['mean_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({dt:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\n너무 나쁨, 5-seed 생략")

with open('/tmp/177_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 177 DONE ===")
