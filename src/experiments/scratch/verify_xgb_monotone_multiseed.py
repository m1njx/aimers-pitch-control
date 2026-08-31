"""
verify_xgb_monotone_multiseed.py
135번 보고서의 XGBoost 단조제약 +34.87점 결과가 단일 시드 노이즈인지,
5-seed 평균으로도 재현되는 진짜 신호인지 검증.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

df_train = pd.read_csv(config.TRAIN_PATH)
target_col = config.TARGET_COL
folds = get_cv_folds(df_train)

MONO_FEATURES = {
    'asof_pitcher_success_rate': 1,
    'asof_pitcher_reverse_rate': -1,
    'asof_batter_success_rate': 1,
    'asof_pitcher_prev5_game_success_rate': 1,
}
SEEDS = [42, 100, 2024, 777, 999]

results = {'free': {s: [] for s in SEEDS}, 'mono': {s: [] for s in SEEDS}}

t0 = time.time()
for k, fold in enumerate(folds):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()

    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)

    y_tr_f = df_tr_f[target_col].values
    y_val_f = df_val_f[target_col].values

    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL]

    mono_vec = [0] * len(X_tr_f.columns)
    for feat, direction in MONO_FEATURES.items():
        if feat in X_tr_f.columns:
            mono_vec[X_tr_f.columns.get_loc(feat)] = direction
    mono_tuple = '(' + ','.join(str(v) for v in mono_vec) + ')'

    X_tr_xgb = X_tr_f.copy()
    X_val_xgb = X_val_f.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
    X_tr_xgb = X_tr_xgb.astype(np.float32)
    X_val_xgb = X_val_xgb.astype(np.float32)

    for seed in SEEDS:
        mx_free = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                     colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                     n_jobs=-1, eval_metric='logloss')
        mx_free.fit(X_tr_xgb, y_tr_f)
        p_xf = np.clip(mx_free.predict_proba(X_val_xgb)[:, 1] - 0.006, 1e-6, 1 - 1e-6)
        sk, br, _, _ = calc_brier_skill_score(y_val_f, p_xf)
        results['free'][seed].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

        mx_mono = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                     colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                     n_jobs=-1, eval_metric='logloss', monotone_constraints=mono_tuple)
        mx_mono.fit(X_tr_xgb, y_tr_f)
        p_xm = np.clip(mx_mono.predict_proba(X_val_xgb)[:, 1] - 0.006, 1e-6, 1 - 1e-6)
        sk, br, _, _ = calc_brier_skill_score(y_val_f, p_xm)
        results['mono'][seed].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    print(f"Fold {k+1} ({fold.val_season}) done. elapsed={time.time()-t0:.1f}s")

print("\n=== PER-SEED RESULTS ===")
deltas = []
for seed in SEEDS:
    sk_free = evaluate_fold_skills(results['free'][seed])
    sk_mono = evaluate_fold_skills(results['mono'][seed])
    d = sk_mono - sk_free
    deltas.append(d)
    print(f"  seed={seed}: free={sk_free:.2f}점 mono={sk_mono:.2f}점 delta={d:+.2f}점")

mean_delta = float(np.mean(deltas))
std_delta = float(np.std(deltas))
print(f"\n5-seed mean delta: {mean_delta:+.2f}점 (std={std_delta:.2f}점)")
print(f"Individual deltas: {[f'{d:+.2f}' for d in deltas]}")

with open('~/LG_data/outputs/135b_xgb_monotone_multiseed_verify.md', 'w', encoding='utf-8') as f:
    f.write("# 135b. XGBoost 단조제약 +34.87점 결과 5-seed 재검증 보고서\n\n")
    f.write("- **목적**: 135번 보고서의 XGBoost 단일시드(42) 단조제약 결과(+34.87점)가 진짜 신호인지 시드 노이즈인지 검증\n\n---\n\n")
    f.write("## 결과\n\n| Seed | Free Skill | Mono Skill | Delta |\n|:---:|:---:|:---:|:---:|\n")
    for seed in SEEDS:
        sk_free = evaluate_fold_skills(results['free'][seed])
        sk_mono = evaluate_fold_skills(results['mono'][seed])
        f.write(f"| {seed} | `{sk_free:.2f}점` | `{sk_mono:.2f}점` | `{sk_mono-sk_free:+.2f}점` |\n")
    f.write(f"\n**5-seed 평균 delta**: `{mean_delta:+.2f}점` (표준편차 `{std_delta:.2f}점`)\n\n")
    f.write(f"**원래 135번 보고서 단일시드(42) delta**: `+34.87점`\n\n")
    if mean_delta > 1.70 and std_delta < mean_delta:
        f.write("## 결론: 신호 재현됨 — 심화 검토 대상\n")
    else:
        f.write("## 결론: 단일시드 노이즈로 재분류. 135번 보고서의 +34.87점 결론은 철회.\n")

print("\nReport 135b written!")
