"""
173_l2_objective_followups.py
168번(L2/Brier 직접 목적함수, 5-seed 정식검증 +9.70점=853.39) 후속 실험 일괄 처리:
1. L2-objective 모델별(lgb/cb/xgb) OOF 저장 (재사용을 위해)
2. classification-objective(캐시됨) vs L2-objective 모델별 단독 성능 비교 -> 어느 모델이
   L2 덕을 가장 많이 보는지 분리
3. L2-앙상블 vs classification-앙상블 블렌딩 (서로 다른 목적함수라 상관관계가 낮을 수 있음)
4. 하이브리드: CatBoost만 L2, 나머지는 classification (혹은 그 반대) 조합 탐색
5. L2-objective 앙상블 전용 가중치(w_lgb,w_cb,w_xgb) nested 재탐색
전부 nested-honest(inner 2022/23 선택 -> outer 2024 최초 적용) 원칙 준수.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BEST_SHIFT = (-0.025, 0.0, -0.05)  # 168번에서 nested-선택된 shift


def train_l2_and_save_oof(df_train, random_seeds):
    """168번의 run_eval_l2를 모델별 OOF 저장하도록 재작성."""
    folds = get_cv_folds(df_train)
    n = len(df_train)
    oof_lgb = np.full(n, np.nan)
    oof_cb = np.full(n, np.nan)
    oof_xgb = np.full(n, np.nan)

    for k, fold in enumerate(folds):
        idx_tr, idx_val = fold.train_idx, fold.val_idx
        df_tr_f = df_train.iloc[idx_tr].copy()
        df_val_f = df_train.iloc[idx_val].copy()
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

        y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']

        bag_p_lgb = np.zeros(len(idx_val))
        bag_p_cb = np.zeros(len(idx_val))
        bag_p_xgb = np.zeros(len(idx_val))

        for seed in random_seeds:
            cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]
            m_lgb = lgb.LGBMRegressor(objective='regression', n_estimators=250, num_leaves=45,
                                       learning_rate=0.05, min_child_samples=20,
                                       colsample_bytree=0.7, subsample=0.7,
                                       random_state=seed, verbosity=-1, n_jobs=-1)
            m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
            bag_p_lgb += np.clip(m_lgb.predict(X_val_f), 1e-6, 1 - 1e-6)

            X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
                X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
            for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
                X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
                X_val_cb[c] = X_val_cb[c].astype(np.float32)
            m_cb = CatBoostRegressor(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                                      loss_function='RMSE', random_seed=seed, verbose=0,
                                      cat_features=cat_cols, thread_count=-1)
            m_cb.fit(X_tr_cb, y_tr_f)
            bag_p_cb += np.clip(m_cb.predict(X_val_cb), 1e-6, 1 - 1e-6)

            X_tr_xgb, X_val_xgb = X_tr_f.copy(), X_val_f.copy()
            for c in cat_cols:
                X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
                X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
            m_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250, max_depth=5,
                                      learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
                                      random_state=seed, n_jobs=-1)
            m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr_f)
            bag_p_xgb += np.clip(m_xgb.predict(X_val_xgb.astype(np.float32)), 1e-6, 1 - 1e-6)

        n_seeds = len(random_seeds)
        oof_lgb[idx_val] = bag_p_lgb / n_seeds
        oof_cb[idx_val] = bag_p_cb / n_seeds
        oof_xgb[idx_val] = bag_p_xgb / n_seeds
        log(f"  fold{k+1}({fold.val_season}) L2 학습 완료")

    return oof_lgb, oof_cb, oof_xgb


df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train)
y_full = df_train[config.TARGET_COL].values
n = len(df_train)
inner_folds = [f for f in folds if f.val_season in (2022, 2023)]
outer_fold = [f for f in folds if f.val_season == 2024][0]


def fold_skill_mean(p_full, fold_list):
    return float(np.mean([calc_brier_skill_score(y_full[f.val_idx], np.clip(p_full[f.val_idx], 1e-6, 1-1e-6))[0]
                           for f in fold_list]))


log("=== 173: L2-objective 5-seed 학습 (모델별 OOF 저장) ===")
t0 = time.time()
oof_lgb_l2, oof_cb_l2, oof_xgb_l2 = train_l2_and_save_oof(df_train, FULL_SEEDS)
log(f"학습 완료 ({(time.time()-t0)/60:.1f}min)")
np.savez('/tmp/173_l2_oof.npz', oof_lgb=oof_lgb_l2, oof_cb=oof_cb_l2, oof_xgb=oof_xgb_l2)

s_lgb, s_cb, s_xgb = BEST_SHIFT
p_l2_ens = np.clip(0.15 * (oof_lgb_l2 + s_lgb) + 0.75 * (oof_cb_l2 + s_cb) + 0.10 * (oof_xgb_l2 + s_xgb), 1e-6, 1-1e-6)
log(f"[검증] L2 앙상블(기존 168번 가중치/shift 재사용) nested-full: {fold_skill_mean(p_l2_ens, folds):.2f} "
    f"(168번 기록값: 853.39)")

# --- classification objective 캐시 로드 ---
d_cls = np.load('/tmp/gbdt_reference_5seed_oof.npz')
cls_val_idx = d_cls['val_idx']
p_cls_lgb = np.full(n, np.nan); p_cls_lgb[cls_val_idx] = d_cls['p_lgb']
p_cls_cb = np.full(n, np.nan); p_cls_cb[cls_val_idx] = d_cls['p_cb']
p_cls_xgb = np.full(n, np.nan); p_cls_xgb[cls_val_idx] = d_cls['p_xgb']
p_cls_ens = np.full(n, np.nan); p_cls_ens[cls_val_idx] = d_cls['p_ens']

log(f"\n=== 아이디어 2: 모델별 classification vs L2 단독 비교 (동일 가중치 앙상블 없이 개별 모델) ===")
for name, p_cls, p_l2 in [('lgb', p_cls_lgb, oof_lgb_l2), ('cb', p_cls_cb, oof_cb_l2), ('xgb', p_cls_xgb, oof_xgb_l2)]:
    log(f"  {name}: classification={fold_skill_mean(p_cls, folds):.2f}  L2={fold_skill_mean(p_l2, folds):.2f}  "
        f"delta={fold_skill_mean(p_l2, folds)-fold_skill_mean(p_cls, folds):+.2f}")

log(f"\n=== 아이디어 1: L2-앙상블 vs classification-앙상블 블렌딩 (상관관계 먼저 확인) ===")
mask = ~np.isnan(p_cls_ens) & ~np.isnan(p_l2_ens)
corr = np.corrcoef(p_cls_ens[mask], p_l2_ens[mask])[0, 1]
log(f"  두 앙상블 간 상관계수: {corr:.4f}")

inner_idx = np.concatenate([f.val_idx for f in inner_folds])
best_w, best_inner = 0.0, -1e9
for w in np.linspace(0, 1, 41):
    p_blend = np.clip((1 - w) * p_cls_ens + w * p_l2_ens, 1e-6, 1 - 1e-6)
    s = fold_skill_mean(p_blend, inner_folds)
    if s > best_inner:
        best_inner, best_w = s, float(w)
p_blend_full = np.clip((1 - best_w) * p_cls_ens + best_w * p_l2_ens, 1e-6, 1 - 1e-6)
log(f"  best_w_l2(inner-selected)={best_w:.2f} nested-full skill={fold_skill_mean(p_blend_full, folds):.2f} "
    f"(classification단독=841.97~843.69, L2단독=853.39)")

log(f"\n=== 아이디어 4: 하이브리드 (모델별로 objective를 다르게 섞은 조합) ===")
hybrid_variants = {
    'cb_L2+lgb_cls+xgb_cls': (p_cls_lgb, oof_cb_l2, p_cls_xgb),
    'cb_cls+lgb_L2+xgb_L2': (oof_lgb_l2, p_cls_cb, oof_xgb_l2),
    'all_L2(기존168번)': (oof_lgb_l2, oof_cb_l2, oof_xgb_l2),
    'all_classification(기존SSOT)': (p_cls_lgb, p_cls_cb, p_cls_xgb),
}
for name, (p_l, p_c, p_x) in hybrid_variants.items():
    p_h = np.clip(0.15 * p_l + 0.75 * p_c + 0.10 * p_x, 1e-6, 1 - 1e-6)
    log(f"  [{name}] nested-full skill={fold_skill_mean(p_h, folds):.2f}")

log(f"\n=== 아이디어 5: L2-objective 전용 가중치 nested 재탐색 (w_lgb,w_cb,w_xgb) ===")
best_weights, best_w_inner = (0.15, 0.75, 0.10), -1e9
for w_cb in np.linspace(0.5, 0.9, 9):
    for w_lgb in np.linspace(0.0, 1 - w_cb, 6):
        w_xgb = 1 - w_cb - w_lgb
        if w_xgb < 0:
            continue
        p_h = np.clip(w_lgb * (oof_lgb_l2 + s_lgb) + w_cb * (oof_cb_l2 + s_cb) + w_xgb * (oof_xgb_l2 + s_xgb), 1e-6, 1 - 1e-6)
        s = fold_skill_mean(p_h, inner_folds)
        if s > best_w_inner:
            best_w_inner, best_weights = s, (w_lgb, w_cb, w_xgb)
p_reweighted = np.clip(best_weights[0] * (oof_lgb_l2 + s_lgb) + best_weights[1] * (oof_cb_l2 + s_cb) +
                        best_weights[2] * (oof_xgb_l2 + s_xgb), 1e-6, 1 - 1e-6)
log(f"  best_weights(inner-selected)={best_weights} nested-full skill={fold_skill_mean(p_reweighted, folds):.2f} "
    f"(기존 15/75/10 가중치 L2 결과: 853.39)")

summary = {
    'l2_ensemble_verify': fold_skill_mean(p_l2_ens, folds),
    'per_model_cls_vs_l2': {name: {'cls': fold_skill_mean(p_cls, folds), 'l2': fold_skill_mean(p_l2, folds)}
                             for name, p_cls, p_l2 in [('lgb', p_cls_lgb, oof_lgb_l2), ('cb', p_cls_cb, oof_cb_l2), ('xgb', p_cls_xgb, oof_xgb_l2)]},
    'cls_l2_ensemble_corr': float(corr),
    'blend_best_w_l2': best_w,
    'blend_nested_full': fold_skill_mean(p_blend_full, folds),
    'hybrids': {name: fold_skill_mean(np.clip(0.15*p_l+0.75*p_c+0.10*p_x,1e-6,1-1e-6), folds)
                for name, (p_l, p_c, p_x) in hybrid_variants.items()},
    'reweighted_best_weights': best_weights,
    'reweighted_nested_full': fold_skill_mean(p_reweighted, folds),
}
with open('/tmp/173_result.json', 'w') as f:
    json.dump(summary, f, indent=2)
log("\n=== 173 DONE ===")
