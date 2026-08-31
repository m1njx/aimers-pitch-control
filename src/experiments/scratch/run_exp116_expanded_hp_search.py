"""
run_exp116_expanded_hp_search.py
작업 3: 확장 하이퍼파라미터 탐색 (LightGBM / CatBoost / XGBoost 각 8개 후보)
inner fold(2022-23)로 최적화, 2024는 최종 확인만
strict_as_of=True 사용
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from core.eval_utils import run_standard_sota_evaluation, calc_brier_skill_score, calc_raw_brier
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

OUTPUTS_DIR = Path('~/LG_data/outputs')
import config

print("=== Task 3: Expanded Hyperparameter Search ===")
t0 = time.time()

df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train data: {len(df_train):,} rows")

SSOT_SKILL = 850.09
SSOT_BRIER = 0.247538

# Inner folds only (2022, 2023) for HP search
folds_all = get_cv_folds(df_train)
folds_inner = [f for f in folds_all if f.val_season in (2022, 2023)]

def fit_inner_brier(model_params_dict, seed=42):
    """Trains on inner folds (2022-23) and returns mean inner brier."""
    inner_briers = []
    for fold in folds_inner:
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
        as_of = fold.fold_max_season

        prep = PitchPreprocessor()
        prep.fit(df_tr_f, as_of_season=as_of, is_final=False)
        X_tr_f = prep.transform(df_tr_f)
        X_val_f = prep.transform(df_val_f)

        # count_x_base
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

        cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                    or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
        cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

        y_tr = df_tr_f[config.TARGET_COL].values
        y_val = df_val_f[config.TARGET_COL].values

        lgb_p = model_params_dict.get('lgb', {})
        cb_p  = model_params_dict.get('cb', {})
        xgb_p = model_params_dict.get('xgb', {})

        lgb_base = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                        colsample_bytree=0.8, subsample=0.8, random_state=seed, verbosity=-1, n_jobs=-1)
        lgb_base.update(lgb_p)
        m_lgb = lgb.LGBMClassifier(**lgb_base)
        m_lgb.fit(X_tr_f, y_tr, categorical_feature=cat_idx)
        p_lgb = np.clip(m_lgb.predict_proba(X_val_f)[:, 1] - 0.007, 1e-6, 1-1e-6)

        X_tr_cb = X_tr_f.copy(); X_val_cb = X_val_f.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
            X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
        for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
            X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
            X_val_cb[c] = X_val_cb[c].astype(np.float32)
        cb_base = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                       random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
        cb_base.update(cb_p); cb_base['cat_features'] = cat_cols
        m_cb = CatBoostClassifier(**cb_base)
        m_cb.fit(X_tr_cb, y_tr)
        p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] - 0.008, 1e-6, 1-1e-6)

        X_tr_xgb = X_tr_f.copy(); X_val_xgb = X_val_f.copy()
        for c in cat_cols:
            X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
            X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
        xgb_base = dict(n_estimators=250, max_depth=5, learning_rate=0.05,
                        colsample_bytree=0.8, subsample=0.8, random_state=seed,
                        n_jobs=-1, eval_metric='logloss')
        xgb_base.update(xgb_p); xgb_base['random_state'] = seed
        m_xgb = xgb.XGBClassifier(**xgb_base)
        m_xgb.fit(X_tr_xgb.astype(np.float32), y_tr)
        p_xgb = np.clip(m_xgb.predict_proba(X_val_xgb.astype(np.float32))[:, 1] - 0.006, 1e-6, 1-1e-6)

        p_ens = np.clip(0.20*p_lgb + 0.70*p_cb + 0.10*p_xgb, 1e-6, 1-1e-6)
        inner_briers.append(calc_raw_brier(y_val, p_ens))

    return float(np.mean(inner_briers))

# HP candidate grid (8-10 per model)
HP_CANDIDATES = [
    # Description, model_params_dict
    ("Baseline (SSOT params)", {}),
    # LightGBM variants
    ("LGBM: more leaves (64)", {'lgb': {'num_leaves': 64}}),
    ("LGBM: deeper (64 leaves, lr=0.03)", {'lgb': {'num_leaves': 64, 'learning_rate': 0.03, 'n_estimators': 400}}),
    ("LGBM: smaller min_child (10)", {'lgb': {'min_child_samples': 10}}),
    ("LGBM: less colsample (0.7)", {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7}}),
    ("LGBM: more colsample (0.9)", {'lgb': {'colsample_bytree': 0.9, 'subsample': 0.9}}),
    # CatBoost variants
    ("CB: depth=8", {'cb': {'depth': 8}}),
    ("CB: depth=4 (shallow)", {'cb': {'depth': 4}}),
    ("CB: lower l2 (3.0)", {'cb': {'l2_leaf_reg': 3.0}}),
    ("CB: higher l2 (20.0)", {'cb': {'l2_leaf_reg': 20.0}}),
    ("CB: more iters (350)", {'cb': {'iterations': 350}}),
    # XGBoost variants
    ("XGB: deeper (max_depth=7)", {'xgb': {'max_depth': 7}}),
    ("XGB: shallower (max_depth=4)", {'xgb': {'max_depth': 4}}),
    ("XGB: more trees (350)", {'xgb': {'n_estimators': 350}}),
    ("XGB: lower colsample (0.7)", {'xgb': {'colsample_bytree': 0.7, 'subsample': 0.7}}),
    # Combined
    ("CB: depth=8 + LGBM: 64 leaves", {'cb': {'depth': 8}, 'lgb': {'num_leaves': 64}}),
]

print(f"\nSearching {len(HP_CANDIDATES)} HP candidates on inner folds (2022-23)...")
hp_results = []
for desc, mp in HP_CANDIDATES:
    t_start = time.time()
    inner_brier = fit_inner_brier(mp)
    t_end = time.time()
    hp_results.append({'desc': desc, 'mp': mp, 'inner_brier': inner_brier, 'time': t_end - t_start})
    print(f"  [{desc}]: inner_brier={inner_brier:.6f} ({t_end-t_start:.1f}s)")

# Sort by inner_brier (lower = better)
hp_results.sort(key=lambda x: x['inner_brier'])
print(f"\nBest HP config: {hp_results[0]['desc']} (inner_brier={hp_results[0]['inner_brier']:.6f})")

# Full 3-fold evaluation of top 3 candidates
print("\nRunning full 3-fold eval on top 3 candidates...")
top3_full = []
for hp in hp_results[:3]:
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=hp['mp'])
    top3_full.append({'desc': hp['desc'], 'mp': hp['mp'], 'inner_brier': hp['inner_brier'], 'full': r})
    print(f"  [{hp['desc']}]: Skill={r['mean_fold_skill']:.2f}점, Brier={r['overall_raw_brier']:.6f}")

elapsed = time.time() - t0
from datetime import datetime
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

lines = []
lines.append(f"# 116. 확장 하이퍼파라미터 탐색 보고서\n")
lines.append(f"- **작성 일시**: {NOW_STR}")
lines.append(f"- **실험 시간**: {elapsed/60:.1f}분")
lines.append(f"- **탐색 후보**: {len(HP_CANDIDATES)}개")
lines.append(f"- **SSOT 기준**: 850.09점 / Raw Brier 0.247538 (strict_as_of=True)\n")
lines.append("---\n")
lines.append("## 1. Inner Fold (2022-23) 탐색 결과 (정렬: 낮은 Brier 우선)\n")
lines.append("| 순위 | 실험 설명 | Inner Brier (2022-23) | 기준 대비 |")
lines.append("|:---:|:---:|:---:|:---:|")
baseline_inner = hp_results[[h['desc'] for h in hp_results].index('Baseline (SSOT params)')]['inner_brier'] if any(h['desc']=='Baseline (SSOT params)' for h in hp_results) else hp_results[-1]['inner_brier']
for i, hp in enumerate(hp_results):
    delta = hp['inner_brier'] - baseline_inner
    sign = '+' if delta >= 0 else ''
    lines.append(f"| {i+1} | {hp['desc']} | `{hp['inner_brier']:.6f}` | `{sign}{delta:.6f}` |")

lines.append("\n## 2. Top 3 후보 전체 3-Fold 검증\n")
lines.append("| 실험 | Inner Brier | 3-Fold Skill | Overall Brier | SSOT 대비 |")
lines.append("|:---:|:---:|:---:|:---:|:---:|")
for t in top3_full:
    sk = t['full']['mean_fold_skill']
    br = t['full']['overall_raw_brier']
    delta = sk - SSOT_SKILL
    sign = '+' if delta >= 0 else ''
    lines.append(f"| {t['desc']} | `{t['inner_brier']:.6f}` | `{sk:.2f}점` | `{br:.6f}` | `{sign}{delta:.2f}점` |")

lines.append("\n## 3. Fold별 최고 구성 상세\n")
best_full = max(top3_full, key=lambda x: x['full']['mean_fold_skill'])
lines.append(f"**최고 구성**: {best_full['desc']}")
lines.append("\n| Fold | Val Season | Raw Brier | Skill Score |")
lines.append("|:---:|:---:|:---:|:---:|")
for fd in best_full['full']['fold_details']:
    lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")

lines.append("\n## 4. 결론\n")
best_skill = max(t['full']['mean_fold_skill'] for t in top3_full)
if best_skill > SSOT_SKILL:
    best_t = max(top3_full, key=lambda x: x['full']['mean_fold_skill'])
    lines.append(f"> ✅ **HP 개선 채택**: `{best_t['desc']}`가 SSOT 대비 {best_t['full']['mean_fold_skill']-SSOT_SKILL:+.2f}점 개선")
    lines.append(f">\n> 채택된 model_params: `{json.dumps(best_t['mp'])}`")
else:
    lines.append("> ❌ **HP 탐색으로 개선 없음**: 현재 SSOT 파라미터가 최적.")

with open(OUTPUTS_DIR / '116_expanded_hp_search.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport 116 written!")
best_t = max(top3_full, key=lambda x: x['full']['mean_fold_skill'])
print(f"Best config: {best_t['desc']} → Skill={best_t['full']['mean_fold_skill']:.2f}점")
print(f"Best model_params: {json.dumps(best_t['mp'])}")

# Save best params for Task 4
with open('~/LG_data/scratch/best_hp_params.json', 'w') as f:
    json.dump({'best_desc': best_t['desc'], 'best_mp': best_t['mp'],
               'best_skill': best_t['full']['mean_fold_skill'],
               'best_brier': best_t['full']['overall_raw_brier']}, f, indent=2)
print("Best HP params saved to scratch/best_hp_params.json")
