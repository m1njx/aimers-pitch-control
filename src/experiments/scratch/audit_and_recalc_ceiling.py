import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/LG_data'))
import config
from cv_utils import get_cv_folds

warnings.filterwarnings('ignore')

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
RAW_DIR = OUTPUTS_DIR / 'raw'

NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def calc_raw_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def calc_fold_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier

print("Loading dataset for Predictability Ceiling Audit & Recalculation...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Prep grouping columns
df_all = df_train.copy()
df_all['base_state_str'] = ((df_all['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                            (df_all['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                            (df_all['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
df_all['count_code_str'] = (df_all['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                            df_all['strikes_before'].fillna(0).astype(int).astype(str))
df_all['outs_str'] = df_all['outs_before'].fillna(0).astype(int).astype(str)

# =========================================================================
# WORK 1: Audit 79th Ceiling Calculation Method (In-Sample Overfitting Audit)
# =========================================================================
print("\n" + "="*70)
print("[Task 1] 79번 이론적 한계 계산 방식 감사")
print("="*70)

# Re-examine Group: [pitcher_id, count_code_str, base_state_str]
grp_pitcher_sit = df_all.groupby(['pitcher_id', 'count_code_str', 'base_state_str'])[config.TARGET_COL].agg(['count', 'mean', 'std']).reset_index()

total_groups = len(grp_pitcher_sit)
sample_counts = grp_pitcher_sit['count'].values
min_samples = int(np.min(sample_counts))
max_samples = int(np.max(sample_counts))
mean_samples = float(np.mean(sample_counts))
median_samples = float(np.median(sample_counts))

pct_less_than_5 = float(np.mean(sample_counts < 5) * 100)
pct_less_than_3 = float(np.mean(sample_counts < 3) * 100)
pct_equal_1 = float(np.mean(sample_counts == 1) * 100)

print(f"Total Pitcher x Situation Groups: {total_groups}개")
print(f"Sample Count Distribution -> Min: {min_samples}, Max: {max_samples}, Mean: {mean_samples:.2f}, Median: {median_samples:.2f}")
print(f"  - Sample Count == 1 (Direct 0 Brier Error in-sample!): {pct_equal_1:.2f}% ({np.sum(sample_counts == 1)}개 그룹)")
print(f"  - Sample Count < 3: {pct_less_than_3:.2f}%")
print(f"  - Sample Count < 5: {pct_less_than_5:.2f}%")

audit_results = {
    "total_groups": total_groups,
    "min_samples": min_samples,
    "max_samples": max_samples,
    "mean_samples": mean_samples,
    "median_samples": median_samples,
    "pct_equal_1": pct_equal_1,
    "pct_less_than_3": pct_less_than_3,
    "pct_less_than_5": pct_less_than_5,
    "verdict": "In-Sample Overfitting (학습 데이터 전체의 관측 평균을 대입하여 표본 1~2개 그룹에서 0 오차가 발생하는 착시 현상)"
}

with open(RAW_DIR / 'task1_ceiling_audit_summary.json', 'w', encoding='utf-8') as f:
    json.dump(audit_results, f, indent=2, ensure_ascii=False)

# =========================================================================
# WORK 2: Recalculate Ceiling using Strict Nested Validation (Held-Out)
# =========================================================================
print("\n" + "="*70)
print("[Task 2] Strict Out-of-Sample (Held-Out 2024년) 이론적 한계 재계산")
print("="*70)

# Outer Fold: Train (2019-2023, season <= 2023) -> Test/Holdout (2024, season == 2024)
df_tr_outer = df_all[df_all['season'] <= 2023].copy()
df_va_outer = df_all[df_all['season'] == 2024].copy()

global_prior = float(df_tr_outer[config.TARGET_COL].mean())

levels = [
    ("L1: count_code (12개 그룹)", ['count_code_str']),
    ("L2: count_code x base_state (96개 그룹)", ['count_code_str', 'base_state_str']),
    ("L3: count_code x base_state x outs (288개 그룹)", ['count_code_str', 'base_state_str', 'outs_str']),
    ("L4: pitcher_id x count_code", ['pitcher_id', 'count_code_str']),
    ("L5: pitcher_id x count_code x base_state", ['pitcher_id', 'count_code_str', 'base_state_str']),
    ("L6: pitcher_id x count_code x base_state x outs", ['pitcher_id', 'count_code_str', 'base_state_str', 'outs_str']),
]

recalc_results = []

for lname, group_cols in levels:
    # Calculate empirical means with Bayesian m-estimate smoothing (m=10)
    # Target p_g_smoothed = (count * mean + m * global_prior) / (count + m)
    m_smooth = 10.0
    grp_stats = df_tr_outer.groupby(group_cols)[config.TARGET_COL].agg(['count', 'mean']).reset_index()
    grp_stats['p_smooth'] = (grp_stats['count'] * grp_stats['mean'] + m_smooth * global_prior) / (grp_stats['count'] + m_smooth)

    # Merge onto 2024 outer fold
    df_merged = df_va_outer.merge(grp_stats[group_cols + ['p_smooth']], on=group_cols, how='left')
    p_pred = df_merged['p_smooth'].fillna(global_prior).values
    y_true = df_va_outer[config.TARGET_COL].values

    # Evaluate 2024 Held-Out Brier & Skill
    sk_2024, br_2024, base_br_2024 = calc_fold_skill_score(y_true, p_pred)

    # Calculate 3-fold Out-of-Sample average
    fold_briers, fold_skills = [], []
    for fi, fold in enumerate(folds):
        df_tr_f = df_all.iloc[fold.train_idx].copy()
        df_va_f = df_all.iloc[fold.val_idx].copy()
        g_f = df_tr_f.groupby(group_cols)[config.TARGET_COL].agg(['count', 'mean']).reset_index()
        g_f['p_smooth'] = (g_f['count'] * g_f['mean'] + m_smooth * global_prior) / (g_f['count'] + m_smooth)

        df_m_f = df_va_f.merge(g_f[group_cols + ['p_smooth']], on=group_cols, how='left')
        p_pred_f = df_m_f['p_smooth'].fillna(global_prior).values
        y_val_f = df_va_f[config.TARGET_COL].values

        sk_f, br_f, _ = calc_fold_skill_score(y_val_f, p_pred_f)
        fold_briers.append(br_f)
        fold_skills.append(sk_f)

    mean_brier_3f = float(np.mean(fold_briers))
    mean_skill_3f = float(np.mean(fold_skills))
    inner_brier = float((fold_briers[0] + fold_briers[1]) / 2.0)

    print(f"[{lname}] Inner Brier={inner_brier:.6f} | 3-Fold Brier={mean_brier_3f:.6f} | 3-Fold Skill={mean_skill_3f:.2f}점 | 2024 Held-out Skill={sk_2024:.2f}점")

    recalc_results.append({
        "level_name": lname,
        "group_cols": group_cols,
        "num_groups_tr": len(grp_stats),
        "inner_brier": inner_brier,
        "mean_brier_3f": mean_brier_3f,
        "mean_skill_3f": mean_skill_3f,
        "heldout_2024_brier": br_2024,
        "heldout_2024_skill": sk_2024
    })

best_recalc = submission_checklist.safe_select_best_candidate(recalc_results, sort_key="inner_brier", exp_name="Recalculated Ceiling Levels")

with open(RAW_DIR / 'task2_recalculated_ceiling_summary.json', 'w', encoding='utf-8') as f:
    json.dump(recalc_results, f, indent=2, ensure_ascii=False)

print("\nCeiling Audit and Recalculation Finished Successfully!")
