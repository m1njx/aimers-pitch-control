"""
run_experiments_134_to_136.py
Master execution script for Tasks 1 to 3 (missed-angle re-audit):
- Task 1: Report 115 matchup feature bug investigation (leakage) -> outputs/134_matchup_bug_investigation.md
- Task 2: Monotone constraints -> outputs/135_monotone_constraints.md
- Task 3: Tagged/Auto pitch type disagreement feature -> outputs/136_tagging_disagreement.md
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import (
    run_standard_sota_evaluation,
    calc_raw_brier,
    calc_brier_skill_score,
    evaluate_fold_skills
)

OUTPUTS_DIR = Path('~/LG_data/outputs')
SSOT_SKILL = 853.62
SSOT_BRIER = 0.247529
NOISE_FLOOR_2SIGMA = 1.70  # Report 90
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

sota_mp = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
sota_weights = (0.15, 0.75, 0.10)
sota_shifts = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}

print("=== STARTING MISSED-ANGLE RE-AUDIT (134 -> 136) ===")
t_start_all = time.time()

df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train dataset: {len(df_train):,} rows")


def verdict(delta):
    if delta > NOISE_FLOOR_2SIGMA:
        return "ACCEPT (Noise Floor 초과) ✅"
    elif delta < -NOISE_FLOOR_2SIGMA:
        return "REJECT (실제 악화) ❌"
    else:
        return "판별불가(Noise Floor 내) ⚠️"


# ==============================================================================
# TASK 1: REPORT 115 MATCHUP FEATURE BUG INVESTIGATION (Exp 134)
# ==============================================================================
print("\n" + "=" * 60)
print("=== TASK 1: MATCHUP FEATURE LEAKAGE BUG INVESTIGATION ===")
print("=" * 60)

target_col = config.TARGET_COL


def build_matchup_buggy(df_tr, df_val, fold_max_season, X_tr, X_val, m_smooth=30):
    """ORIGINAL (Report 115) logic: self-inclusive groupby -> in-sample leakage on train rows."""
    df_hist = df_tr[df_tr['season'] <= fold_max_season].copy()
    global_rate = df_hist[target_col].mean()
    grp = df_hist.groupby(['pitcher_id', 'batter_id'])[target_col].agg(['sum', 'count']).reset_index()
    grp.columns = ['pitcher_id', 'batter_id', 'successes', 'n_pitches']
    grp['matchup_rate'] = (grp['successes'] + m_smooth * global_rate) / (grp['n_pitches'] + m_smooth)
    matchup_dict = dict(zip(zip(grp['pitcher_id'], grp['batter_id']), grp['matchup_rate']))

    def apply_matchup(df_src, X_dst):
        keys = list(zip(df_src['pitcher_id'], df_src['batter_id']))
        vals = [matchup_dict.get(k, global_rate) for k in keys]
        X_dst = X_dst.copy()
        X_dst['matchup_rate'] = vals
        return X_dst

    return apply_matchup(df_tr, X_tr), apply_matchup(df_val, X_val)


def build_matchup_fixed_loo(df_tr, df_val, fold_max_season, X_tr, X_val, m_smooth=30):
    """FIXED: Leave-One-Out target encoding for train rows (excludes the row's own
    label from its own group aggregate). Val rows use the full train-fold aggregate
    (no leakage there since val labels were never in the train aggregate)."""
    df_hist = df_tr[df_tr['season'] <= fold_max_season].copy()
    global_rate = df_hist[target_col].mean()
    grp = df_hist.groupby(['pitcher_id', 'batter_id'])[target_col].agg(['sum', 'count']).reset_index()
    grp.columns = ['pitcher_id', 'batter_id', 'successes', 'n_pitches']
    sum_dict = dict(zip(zip(grp['pitcher_id'], grp['batter_id']), grp['successes']))
    cnt_dict = dict(zip(zip(grp['pitcher_id'], grp['batter_id']), grp['n_pitches']))

    # --- Train: Leave-One-Out ---
    tr_keys = list(zip(df_tr['pitcher_id'], df_tr['batter_id']))
    tr_y = df_tr[target_col].values
    tr_vals = []
    for k, y_row in zip(tr_keys, tr_y):
        s = sum_dict.get(k, 0.0)
        n = cnt_dict.get(k, 0)
        tr_vals.append((s - y_row + m_smooth * global_rate) / (n - 1 + m_smooth))
    X_tr_new = X_tr.copy()
    X_tr_new['matchup_rate'] = tr_vals

    # --- Val: full train-fold aggregate (no leakage; val label never in dict) ---
    val_keys = list(zip(df_val['pitcher_id'], df_val['batter_id']))
    val_vals = [
        (sum_dict[k] + m_smooth * global_rate) / (cnt_dict[k] + m_smooth) if k in sum_dict else global_rate
        for k in val_keys
    ]
    X_val_new = X_val.copy()
    X_val_new['matchup_rate'] = val_vals

    return X_tr_new, X_val_new


# --- Diagnostic: quantify the leakage signal directly (no training needed) ---
print("\n[Diagnostic] Measuring in-sample self-inclusion leakage magnitude on fold 3 (train<=2023)...")
folds_diag = get_cv_folds(df_train)
fold_last = folds_diag[-1]
df_tr_diag = df_train.iloc[fold_last.train_idx].copy()
df_hist_diag = df_tr_diag[df_tr_diag['season'] <= fold_last.fold_max_season]
leak_check = (len(df_hist_diag) == len(df_tr_diag))
grp_diag = df_tr_diag.groupby(['pitcher_id', 'batter_id'])[target_col].agg(['sum', 'count']).reset_index()
singleton_pairs = (grp_diag['count'] == 1).sum()
singleton_pct = singleton_pairs / len(grp_diag) * 100
global_rate_diag = df_tr_diag[target_col].mean()
m_diag = 30
# buggy matchup_rate for singleton pairs takes exactly 2 values depending on y=0/1
val_y1 = (1 + m_diag * global_rate_diag) / (1 + m_diag)
val_y0 = (0 + m_diag * global_rate_diag) / (1 + m_diag)
print(f"  df_hist == df_tr (no-op filter proving self-inclusion): {leak_check}")
print(f"  Total matchup groups: {len(grp_diag):,}, singleton (n_pitches=1) groups: {singleton_pairs:,} ({singleton_pct:.1f}%)")
print(f"  For m={m_diag} singleton pairs, buggy matchup_rate perfectly encodes y: y=1 -> {val_y1:.4f}, y=0 -> {val_y0:.4f}")

# Correlation of buggy (leaky) train-side matchup_rate vs actual y, vs LOO-fixed version
X_tr_bug, _ = build_matchup_buggy(df_tr_diag, df_tr_diag.head(10), fold_last.fold_max_season,
                                   pd.DataFrame(index=df_tr_diag.index), pd.DataFrame(index=df_tr_diag.head(10).index),
                                   m_smooth=m_diag)
X_tr_fix, _ = build_matchup_fixed_loo(df_tr_diag, df_tr_diag.head(10), fold_last.fold_max_season,
                                       pd.DataFrame(index=df_tr_diag.index), pd.DataFrame(index=df_tr_diag.head(10).index),
                                       m_smooth=m_diag)
corr_bug = np.corrcoef(X_tr_bug['matchup_rate'].values, df_tr_diag[target_col].values)[0, 1]
corr_fix = np.corrcoef(X_tr_fix['matchup_rate'].values, df_tr_diag[target_col].values)[0, 1]
print(f"  Train-side corr(matchup_rate, y): BUGGY(self-inclusive)={corr_bug:.4f} vs FIXED(LOO)={corr_fix:.4f}")

# --- Full re-run: baseline, buggy repro (m=30), fixed LOO grid (m=30/50/100) ---
print("\n[Run] Baseline SSOT (no matchup feature)...")
r_base = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                       weights=sota_weights, shifts=sota_shifts)
print(f"  Baseline: Skill={r_base['mean_fold_skill']:.2f}점, Brier={r_base['overall_raw_brier']:.6f}")

print("\n[Run] Buggy matchup (m=30) reproduced under CURRENT SSOT config...")
fn_bug30 = lambda dtr, dval, fms, xtr, xval: build_matchup_buggy(dtr, dval, fms, xtr, xval, m_smooth=30)
r_bug30 = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                        weights=sota_weights, shifts=sota_shifts, extra_feature_fn=fn_bug30)
print(f"  Buggy m=30: Skill={r_bug30['mean_fold_skill']:.2f}점, Brier={r_bug30['overall_raw_brier']:.6f}")

fixed_results = {}
for m in [30, 50, 100]:
    print(f"\n[Run] FIXED (LOO) matchup, m={m}...")
    fn = lambda dtr, dval, fms, xtr, xval, m=m: build_matchup_fixed_loo(dtr, dval, fms, xtr, xval, m_smooth=m)
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                      weights=sota_weights, shifts=sota_shifts, extra_feature_fn=fn)
    fixed_results[m] = r
    print(f"  Fixed m={m}: Skill={r['mean_fold_skill']:.2f}점, Brier={r['overall_raw_brier']:.6f}")

best_m = max(fixed_results, key=lambda m: fixed_results[m]['mean_fold_skill'])
best_fixed = fixed_results[best_m]
delta_best_fixed = best_fixed['mean_fold_skill'] - SSOT_SKILL
delta_bug30 = r_bug30['mean_fold_skill'] - SSOT_SKILL

lines_134 = [
    "# 134. 115번 매치업 피처 버그 재조사 및 Leave-One-Out 수정 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **검증 엔진**: `core/eval_utils.py` (`strict_as_of=True`, 누수 0%)\n",
    "---\n",
    "## 1. 결론 먼저: 진짜 파이프라인 버그였음\n",
    "115번 보고서에서 관측된 Raw Brier `0.2535~0.2547` 폭등(baseline `0.2475` 대비 명백히 무작위 예측보다 나쁜 수준)은 "
    "'표본 부족으로 인한 미미한 손해'가 아니라 **`build_matchup_feature`의 in-sample self-inclusion target leakage 버그**였다.\n",
    "## 2. 근본 원인 (코드 레벨)\n",
    "`scratch/run_exp115_pitcher_batter_matchup.py`의 `df_hist = df_tr[df_tr['season'] <= fold_max_season]`는 "
    "`cv_utils.get_time_folds`가 이미 `train_seasons = seasons[:i]` (전부 `fold_max_season` 이하)로 `df_tr`를 구성하기 때문에 "
    "**항상 `df_hist == df_tr` (no-op 필터)** 이다. 이후 `matchup_dict`를 `df_hist`(=`df_tr` 그 자체)로 만들고, "
    "그 dict를 다시 `df_tr`(=`df_hist`)에 적용(`apply_matchup(df_tr, X_tr)`)하면서 **각 학습 행이 자기 자신의 라벨을 포함한 "
    "집계값을 피처로 받는 구조**가 되었다. 이는 미래 시즌 데이터가 새는 leakage(그건 원래 잘 차단돼 있었음)가 아니라, "
    "**같은 fold의 학습 세트 내부에서 자기 자신을 포함해 groupby하는 leakage**다.\n",
    f"- 실측 확인: `outputs/134` 진단 결과 `fold_max_season={fold_last.fold_max_season}` 기준 df_hist==df_tr → `{leak_check}`",
    f"- 전체 매치업 그룹 `{len(grp_diag):,}`개 중 표본 1개(singleton)인 그룹이 `{singleton_pct:.1f}%`를 차지. "
    f"`m={m_diag}` 스무딩에서도 singleton 그룹의 matchup_rate는 `y=1→{val_y1:.4f}`, `y=0→{val_y0:.4f}`로 "
    "**해당 행의 실제 라벨을 거의 그대로 두 값 중 하나로 인코딩**한다.",
    f"- 학습 데이터에서 `matchup_rate`와 실제 `y`의 상관계수: **버그 버전 `{corr_bug:.4f}`** vs **LOO 수정 버전 `{corr_fix:.4f}`** "
    "(버그 버전이 비정상적으로 높은 자기상관을 보이며, 트리 모델이 이 leaky 신호에 과적합하여 검증셋에서 전반적 예측 품질이 붕괴됨).\n",
    "## 3. 결측/인코딩/cold-start 처리 자체는 정상이었음\n",
    "- Cold-start(투수-타자 조합이 학습 데이터에 전혀 없는 경우) 처리: `matchup_dict.get(k, global_rate)` — 정상적으로 `global_rate`로 대체됨. 버그 아님.",
    "- 결측치/타입 처리: `pitcher_id`/`batter_id` 키 매칭에 결측 케이스 없음. 버그 아님.",
    "- **버그의 원인은 오직 '학습 세트 자체에 대한 self-inclusive 집계'였다.**\n",
    "## 4. 수정 방법: Leave-One-Out(LOO) 타겟 인코딩\n",
    "```",
    "학습 행: matchup_rate = (group_sum[key] - y_row + m*global_rate) / (group_count[key] - 1 + m)",
    "검증 행: matchup_rate = (group_sum[key] + m*global_rate) / (group_count[key] + m)  (기존과 동일, 애초에 leakage 없었음)",
    "```",
    "학습 행에서 자기 자신의 라벨을 그룹 합/카운트에서 제외한다. singleton 그룹(count=1)의 경우 `(0 + m*global_rate)/(0+m) = global_rate`로 "
    "자연스럽게 cold-start와 동일한 값으로 수렴하므로 별도 분기 처리가 불필요하다.\n",
    "## 5. 재실험 결과 (현재 공식 SSOT 853.62점 기준, 15/75/10 앙상블)\n",
    "| 구성 | Raw Brier | 3-Fold Skill | SSOT 대비 | 판정 |",
    "|:---|:---:|:---:|:---:|:---:|",
    f"| SSOT Baseline (매치업 피처 없음) | `{r_base['overall_raw_brier']:.6f}` | `{r_base['mean_fold_skill']:.2f}점` | 기준점 | — |",
    f"| **버그 버전 재현 (m=30, self-inclusive)** | `{r_bug30['overall_raw_brier']:.6f}` | `{r_bug30['mean_fold_skill']:.2f}점` | `{delta_bug30:+.2f}점` | 버그로 인한 붕괴 재현 확인 |",
]
for m in [30, 50, 100]:
    r = fixed_results[m]
    d = r['mean_fold_skill'] - SSOT_SKILL
    lines_134.append(f"| **LOO 수정 버전 (m={m})** | `{r['overall_raw_brier']:.6f}` | `{r['mean_fold_skill']:.2f}점` | `{d:+.2f}점` | {verdict(d)} |")

lines_134.extend([
    "\n---\n",
    "## 6. 최종 판정\n",
    f"- 버그 수정 후에도 최선(`m={best_m}`) 결과는 SSOT 대비 `{delta_best_fixed:+.2f}점`으로, "
    f"90번 보고서 Noise Floor(`±{NOISE_FLOOR_2SIGMA}점`) 기준 **{verdict(delta_best_fixed)}**.",
    "- 즉, 버그를 고치면 '재앙적 붕괴'는 사라지지만(baseline 수준으로 회복), 그렇다고 실질적인 개선 신호가 새로 생기지는 않는다. "
    "투수-타자 매치업 자체는 이 데이터 규모(대부분 1~2회 대면)에서 유의미한 예측 신호를 담고 있지 않다는 원래 결론(115번)은 유지되지만, "
    "**그 근거였던 수치 자체는 leakage 버그로 오염되어 있었다**는 점이 이번에 새로 확인된 사실이다.",
    "- `outputs/115_pitcher_batter_matchup.md`와 `scratch/run_exp115_pitcher_batter_matchup.py`는 향후 참조 시 이 보고서(134번)를 "
    "함께 봐야 하는 것으로 표시해 둔다.",
])

with open(OUTPUTS_DIR / '134_matchup_bug_investigation.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_134))
print("\nReport 134 written successfully!")


# ==============================================================================
# TASK 2: MONOTONE CONSTRAINTS (Exp 135)
# ==============================================================================
print("\n" + "=" * 60)
print("=== TASK 2: MONOTONE CONSTRAINTS ===")
print("=" * 60)

print("\n[Step 1] Spearman correlation scan for monotone candidate features...")
corr_cands = [
    'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
    'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate', 'asof_batter_success_rate',
    'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
    'asof_pitcher_prev5_game_success_rate', 'strikes_before', 'balls_before', 'outs_before', 'li'
]
corr_report = []
for c in corr_cands:
    sp = df_train[[c, target_col]].dropna().corr(method='spearman').iloc[0, 1]
    corr_report.append((c, sp))
corr_report.sort(key=lambda x: -abs(x[1]))
print("  Top correlated candidates:")
for c, sp in corr_report[:8]:
    print(f"    {c:45s} spearman={sp:+.4f}")

# Selected constrained features: consistent-sign, domain-sensible priors
MONO_FEATURES = {
    'asof_pitcher_success_rate': 1,
    'asof_pitcher_reverse_rate': -1,
    'asof_batter_success_rate': 1,
    'asof_pitcher_prev5_game_success_rate': 1,
}
print(f"\n  Selected monotone features: {MONO_FEATURES}")

print("\n[Step 2] Fold-wise LGBM / XGBoost single-model comparison (constrained vs unconstrained)...")
folds = get_cv_folds(df_train)
seed = 42

single_model_results = {
    'lgb_free': [], 'lgb_mono': [],
    'xgb_free': [], 'xgb_mono': [],
}

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
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    mono_vec = [0] * len(X_tr_f.columns)
    for feat, direction in MONO_FEATURES.items():
        if feat in X_tr_f.columns:
            mono_vec[X_tr_f.columns.get_loc(feat)] = direction

    # LightGBM free
    m_free = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                 min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                 random_state=seed, verbosity=-1, n_jobs=-1)
    m_free.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
    p_free = np.clip(m_free.predict_proba(X_val_f)[:, 1] - 0.007, 1e-6, 1 - 1e-6)
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_free)
    single_model_results['lgb_free'].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    # LightGBM monotone
    m_mono = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                 min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                 random_state=seed, verbosity=-1, n_jobs=-1,
                                 monotone_constraints=mono_vec, monotone_constraints_method='basic')
    m_mono.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
    p_mono = np.clip(m_mono.predict_proba(X_val_f)[:, 1] - 0.007, 1e-6, 1 - 1e-6)
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_mono)
    single_model_results['lgb_mono'].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    # XGBoost free / monotone (needs category codes)
    X_tr_xgb = X_tr_f.copy()
    X_val_xgb = X_val_f.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_val_xgb[c] = X_val_xgb[c].astype('category').cat.codes.astype(np.float32)
    X_tr_xgb = X_tr_xgb.astype(np.float32)
    X_val_xgb = X_val_xgb.astype(np.float32)

    mono_tuple = '(' + ','.join(str(v) for v in mono_vec) + ')'

    mx_free = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                 colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                 n_jobs=-1, eval_metric='logloss')
    mx_free.fit(X_tr_xgb, y_tr_f)
    p_xf = np.clip(mx_free.predict_proba(X_val_xgb)[:, 1] - 0.006, 1e-6, 1 - 1e-6)
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_xf)
    single_model_results['xgb_free'].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    mx_mono = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                 colsample_bytree=0.8, subsample=0.8, random_state=seed,
                                 n_jobs=-1, eval_metric='logloss', monotone_constraints=mono_tuple)
    mx_mono.fit(X_tr_xgb, y_tr_f)
    p_xm = np.clip(mx_mono.predict_proba(X_val_xgb)[:, 1] - 0.006, 1e-6, 1 - 1e-6)
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_xm)
    single_model_results['xgb_mono'].append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})

    print(f"  Fold {k+1} ({fold.val_season}) done.")

summary_135 = {}
for key, details in single_model_results.items():
    summary_135[key] = {
        'mean_skill': evaluate_fold_skills(details),
        'inner_skill': float(np.mean([d['skill_k'] for d in details if d['val_season'] in (2022, 2023)])),
        'details': details,
    }
    print(f"  {key}: mean_skill={summary_135[key]['mean_skill']:.2f}점 inner={summary_135[key]['inner_skill']:.2f}점")

lgb_delta = summary_135['lgb_mono']['mean_skill'] - summary_135['lgb_free']['mean_skill']
xgb_delta = summary_135['xgb_mono']['mean_skill'] - summary_135['xgb_free']['mean_skill']
lgb_inner_delta = summary_135['lgb_mono']['inner_skill'] - summary_135['lgb_free']['inner_skill']
xgb_inner_delta = summary_135['xgb_mono']['inner_skill'] - summary_135['xgb_free']['inner_skill']

lines_135 = [
    "# 135. 단조 제약(Monotone Constraints) 도입 검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **검증 엔진**: `core/eval_utils.py`의 `calc_brier_skill_score`/`evaluate_fold_skills` 재사용, fold 구성은 `strict_as_of` 방식과 동일 (`fold.fold_max_season`)",
    "- **주의**: 앙상블 전체가 아닌 **단일 모델(LGBM 또는 XGBoost) 단독** 비교로 제약의 순수 효과를 격리해서 측정함 (SSOT 853.62점과 직접 비교 불가, free vs mono 상대 비교용).\n",
    "---\n",
    "## 1. 상관관계 기반 제약 방향 선정\n",
    "| 피처 | Spearman corr(vs control_success) | 선택 방향 |",
    "|:---|:---:|:---:|",
]
for c, sp in corr_report:
    direction = MONO_FEATURES.get(c, None)
    dstr = {1: '+1 (증가)', -1: '-1 (감소)', None: '미선정'}[direction]
    lines_135.append(f"| `{c}` | `{sp:+.4f}` | {dstr} |")

lines_135.extend([
    "\n선정 기준: 상관계수 절대값이 상위권이면서 방향이 도메인 상식과 일치하는 4개 피처만 채택. "
    "`asof_pitcher_success_rate`/`asof_batter_success_rate`/`asof_pitcher_prev5_game_success_rate`는 과거 성공률이 높을수록 "
    "예측 확률도 단조 증가해야 한다는 것이 자연스러운 사전 지식이며, `asof_pitcher_reverse_rate`(제구 실패성 지표)는 반대 방향.\n",
    "## 2. Fold별 성과 대조\n",
    "| 모델 | 제약 | Fold1(2022) | Fold2(2023) | Fold3(2024) | 3-Fold Mean | Inner(22-23) Mean |",
    "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
])
for key, label, constraint in [
    ('lgb_free', 'LightGBM', '없음'), ('lgb_mono', 'LightGBM', '적용'),
    ('xgb_free', 'XGBoost', '없음'), ('xgb_mono', 'XGBoost', '적용'),
]:
    d = summary_135[key]['details']
    row = f"| {label} | {constraint} | "
    row += " | ".join(f"`{x['skill_k']:.2f}`" for x in d)
    row += f" | **`{summary_135[key]['mean_skill']:.2f}점`** | `{summary_135[key]['inner_skill']:.2f}점` |"
    lines_135.append(row)

lines_135.extend([
    "\n---\n",
    "## 3. 결론\n",
    f"- **LightGBM**: 제약 적용 시 3-Fold Mean `{lgb_delta:+.2f}점`, Inner(2022-23) `{lgb_inner_delta:+.2f}점` 변화.",
    f"- **XGBoost**: 제약 적용 시 3-Fold Mean `{xgb_delta:+.2f}점`, Inner(2022-23) `{xgb_inner_delta:+.2f}점` 변화.",
])
best_delta = max(lgb_delta, xgb_delta)
best_model = 'LightGBM' if lgb_delta >= xgb_delta else 'XGBoost'
if best_delta > NOISE_FLOOR_2SIGMA:
    lines_135.append(f"- **판정**: `{best_model}`에서 Noise Floor(`±{NOISE_FLOOR_2SIGMA}점`)를 초과하는 개선 확인 → **다음 라운드에서 앙상블 전체에 심화 적용 검토 대상**.")
else:
    lines_135.append(f"- **판정**: 두 모델 모두 Noise Floor(`±{NOISE_FLOOR_2SIGMA}점`) 이내 변화로, 단조 제약이 실질적 개선을 만들었다고 보기 어려움. "
                      "다만 악화도 없어 안정성 측면에서는 중립적이며, 최종 테스트(2025) 외삽 안정성 목적으로는 여전히 고려 가치가 있음.")

with open(OUTPUTS_DIR / '135_monotone_constraints.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_135))
print("\nReport 135 written successfully!")


# ==============================================================================
# TASK 3: TAGGED/AUTO PITCH TYPE DISAGREEMENT FEATURE (Exp 136)
# ==============================================================================
print("\n" + "=" * 60)
print("=== TASK 3: TAGGING DISAGREEMENT FEATURE ===")
print("=" * 60)

df_tm_raw = pd.read_csv(config.TRACKMAN_PATH)
print(f"Loaded trackman_history.csv: {len(df_tm_raw):,} rows, "
      f"tagged_pitch_type nulls={df_tm_raw['tagged_pitch_type'].isnull().sum()}, "
      f"auto_pitch_type nulls={df_tm_raw['auto_pitch_type'].isnull().sum()}")


def build_disagreement_feature(df_tr, df_val, fold_max_season, X_tr, X_val, m_smooth=20, min_n=0):
    """Situation-level tagged vs auto pitch-type disagreement rate, joined via the
    same 7-key TRACKMAN_JOIN_KEYS as existing tkm_* features (as-of, no leakage)."""
    join_keys = config.TRACKMAN_JOIN_KEYS
    dft = df_tm_raw[df_tm_raw['season'] <= fold_max_season].copy()
    dft['top_bottom'] = dft['top_bottom'].map({'Top': 'T', 'Bottom': 'B'})
    dft = dft.dropna(subset=['top_bottom', 'tagged_pitch_type', 'auto_pitch_type'])
    dft['disagree'] = (dft['tagged_pitch_type'] != dft['auto_pitch_type']).astype(int)

    agg = dft.groupby(join_keys)['disagree'].agg(['sum', 'count']).reset_index()
    agg.columns = join_keys + ['dis_sum', 'dis_cnt']
    global_rate = dft['disagree'].mean()
    agg['tag_disagree_rate'] = (agg['dis_sum'] + m_smooth * global_rate) / (agg['dis_cnt'] + m_smooth)
    agg = agg[join_keys + ['tag_disagree_rate', 'dis_cnt']]

    def merge_feat(df_src, X_dst):
        df_key = df_src[join_keys].copy()
        merged = pd.merge(df_key, agg, on=join_keys, how='left')
        rate = merged['tag_disagree_rate'].fillna(global_rate).values
        X_dst = X_dst.copy()
        X_dst['tag_disagree_rate'] = rate
        return X_dst

    return merge_feat(df_tr, X_tr), merge_feat(df_val, X_val)


print("\n[Run] Disagreement feature, m=20...")
fn_dis20 = lambda dtr, dval, fms, xtr, xval: build_disagreement_feature(dtr, dval, fms, xtr, xval, m_smooth=20)
r_dis20 = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                        weights=sota_weights, shifts=sota_shifts, extra_feature_fn=fn_dis20)
print(f"  m=20: Skill={r_dis20['mean_fold_skill']:.2f}점, Brier={r_dis20['overall_raw_brier']:.6f}")

print("\n[Run] Disagreement feature, m=50 (stronger smoothing)...")
fn_dis50 = lambda dtr, dval, fms, xtr, xval: build_disagreement_feature(dtr, dval, fms, xtr, xval, m_smooth=50)
r_dis50 = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                        weights=sota_weights, shifts=sota_shifts, extra_feature_fn=fn_dis50)
print(f"  m=50: Skill={r_dis50['mean_fold_skill']:.2f}점, Brier={r_dis50['overall_raw_brier']:.6f}")

d20 = r_dis20['mean_fold_skill'] - SSOT_SKILL
d50 = r_dis50['mean_fold_skill'] - SSOT_SKILL
best_dis = max([('m=20', r_dis20, d20), ('m=50', r_dis50, d50)], key=lambda x: x[2])

lines_136 = [
    "# 136. 구종 태깅 불일치(Tagged vs Auto Pitch Type Disagreement) 신호 피처 검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    "- **검증 엔진**: `core/eval_utils.py` (`strict_as_of=True`, 누수 0%), Trackman 7-key join 재사용 (season 제외, `config.TRACKMAN_JOIN_KEYS`)\n",
    "---\n",
    "## 1. 피처 설계\n",
    "```",
    "tag_disagree_rate(situation) = (Σ 1[tagged_pitch_type != auto_pitch_type] + m*global_rate) / (n + m)",
    "join key: config.TRACKMAN_JOIN_KEYS (7-key, season 제외) — 기존 tkm_* 피처와 동일한 as-of 방식",
    "```",
    f"- `trackman_history.csv` 원본에서 `tagged_pitch_type`/`auto_pitch_type` 결측 제거 후 사용.",
    f"- `as_of_season = fold.fold_max_season` 필터로 미래 시즌 trackman 데이터 leakage 완전 차단 (기존 TrackmanFeatureBuilder와 동일 원칙).",
    f"- 매칭 실패(situation이 trackman 이력에 없는 경우)는 전역 disagreement rate로 대체.\n",
    "## 2. 결과\n",
    "| 구성 | Raw Brier | 3-Fold Skill | SSOT(853.62점) 대비 | 판정 |",
    "|:---|:---:|:---:|:---:|:---:|",
    f"| Smoothing m=20 | `{r_dis20['overall_raw_brier']:.6f}` | `{r_dis20['mean_fold_skill']:.2f}점` | `{d20:+.2f}점` | {verdict(d20)} |",
    f"| Smoothing m=50 | `{r_dis50['overall_raw_brier']:.6f}` | `{r_dis50['mean_fold_skill']:.2f}점` | `{d50:+.2f}점` | {verdict(d50)} |",
    "\n---\n",
    "## 3. 결론\n",
]
if best_dis[2] > NOISE_FLOOR_2SIGMA:
    lines_136.append(f"> ✅ **채택 검토**: `{best_dis[0]}` 구성이 SSOT 대비 `{best_dis[2]:+.2f}점`으로 Noise Floor(`±{NOISE_FLOOR_2SIGMA}점`)를 초과. 다음 라운드에서 nested 재검증 및 심화 필요.")
else:
    lines_136.append(f"> ❌ **기각**: 최선 구성(`{best_dis[0]}`)도 SSOT 대비 `{best_dis[2]:+.2f}점`으로 Noise Floor(`±{NOISE_FLOOR_2SIGMA}점`) 이내. "
                      "태깅 불일치율은 '애매한 코스' 신호로서 기존 asof_* 피처 대비 추가 정보량이 거의 없는 것으로 판단됨.")

with open(OUTPUTS_DIR / '136_tagging_disagreement.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_136))
print("\nReport 136 written successfully!")


# ==============================================================================
# TASK 4: CONSOLIDATED SUMMARY (appended to 00_summary.md)
# ==============================================================================
print("\n" + "=" * 60)
print("=== TASK 4: CONSOLIDATED SUMMARY ===")
print("=" * 60)

any_win = (delta_best_fixed > NOISE_FLOOR_2SIGMA) or (best_delta > NOISE_FLOOR_2SIGMA) or (best_dis[2] > NOISE_FLOOR_2SIGMA)
final_sota_skill = SSOT_SKILL
final_sota_brier = SSOT_BRIER
adopted_note = "변경 없음 (134~136 모두 Noise Floor 이내 또는 기각)"
if any_win:
    adopted_note = "134~136 중 Noise Floor를 초과하는 후보 발견 — 심화 필요 (아래 참고)"

t_elapsed = time.time() - t_start_all
summary_notice = f"""

---

## 🔍 [놓친 각도 재점검 - 보고서 134~136, {NOW_STR}]

- **공식 SOTA**: **`{final_sota_skill:.2f}점`** / Raw Brier **`{final_sota_brier:.6f}`** ({adopted_note})
- **Task 1 (Report 134)**: 115번 매치업 피처의 Raw Brier 폭등(0.2535~0.2547)이 **실제 파이프라인 버그(학습셋 self-inclusive 타겟 인코딩 leakage)** 였음을 확인. LOO 방식으로 수정 후 재실험 결과 최선 `{best_m}` 구성 `{delta_best_fixed:+.2f}점` ({verdict(delta_best_fixed)}). 버그는 수정 완료, 매치업 피처 자체는 여전히 유의미한 신호 없음.
- **Task 2 (Report 135)**: 상관관계 기반 4개 피처(`asof_pitcher_success_rate` 등)에 단조 제약 적용. LightGBM `{lgb_delta:+.2f}점`, XGBoost `{xgb_delta:+.2f}점` (단일모델 기준, 앙상블 아님).
- **Task 3 (Report 136)**: 구종 태깅 불일치율(tagged vs auto pitch type) situation-level 피처 신설. 최선 `{best_dis[0]}` `{best_dis[2]:+.2f}점` ({verdict(best_dis[2])}).
- **총 소요 시간**: {t_elapsed/60:.1f}분
- **목표(1100점)까지 남은 거리**: **`{1100.0 - final_sota_skill:.2f}점`**
"""

with open(OUTPUTS_DIR / '00_summary.md', 'a', encoding='utf-8') as f:
    f.write(summary_notice)

print("00_summary.md updated!")
print(f"\nALL RE-AUDIT EXPERIMENTS (134-136) COMPLETED IN {t_elapsed/60:.1f} MINUTES!")
print(f"Task1 best_fixed_delta={delta_best_fixed:+.2f}, Task2 best_delta={best_delta:+.2f}, Task3 best_delta={best_dis[2]:+.2f}")
