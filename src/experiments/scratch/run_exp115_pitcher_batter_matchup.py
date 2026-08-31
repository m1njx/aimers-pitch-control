"""
run_exp115_pitcher_batter_matchup.py
작업 2: 투수-타자 개별 매치업 이력 피처 실험
Bayesian smoothing (m=30/50) 적용, strict_as_of=True
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from core.eval_utils import run_standard_sota_evaluation

OUTPUTS_DIR = Path('~/LG_data/outputs')
import config

print("=== Task 2: Pitcher-Batter Matchup Feature Experiment ===")
t0 = time.time()

df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train data: {len(df_train):,} rows")

SSOT_SKILL = 850.09
SSOT_BRIER = 0.247538

def build_matchup_feature(df_tr, df_val, fold_max_season, X_tr, X_val, m_smooth=30):
    """
    Computes pitcher_id x batter_id historical control_success rate.
    Applies Bayesian smoothing: estimate = (sum_successes + m * global_rate) / (n + m)
    Only uses data from seasons <= fold_max_season (no leakage).
    """
    target_col = config.TARGET_COL
    
    # Filter history strictly to <= fold_max_season
    df_hist = df_tr[df_tr['season'] <= fold_max_season].copy()
    
    global_rate = df_hist[target_col].mean()
    
    # Group by pitcher_id x batter_id
    grp = df_hist.groupby(['pitcher_id', 'batter_id'])[target_col].agg(['sum', 'count']).reset_index()
    grp.columns = ['pitcher_id', 'batter_id', 'successes', 'n_pitches']
    grp['matchup_rate'] = (grp['successes'] + m_smooth * global_rate) / (grp['n_pitches'] + m_smooth)
    
    # Sample distribution report
    print(f"  [m={m_smooth}] Matchup groups: {len(grp):,}, "
          f"mean n_pitches={grp['n_pitches'].mean():.1f}, "
          f"median={grp['n_pitches'].median():.0f}, "
          f"groups with <5 pitches: {(grp['n_pitches']<5).sum():,} ({(grp['n_pitches']<5).mean()*100:.1f}%)")
    
    matchup_dict = dict(zip(
        zip(grp['pitcher_id'], grp['batter_id']),
        grp['matchup_rate']
    ))
    
    def apply_matchup(df_src, X_dst):
        keys = list(zip(df_src['pitcher_id'], df_src['batter_id']))
        vals = [matchup_dict.get(k, global_rate) for k in keys]
        X_dst = X_dst.copy()
        X_dst['matchup_rate'] = vals
        return X_dst
    
    X_tr_new = apply_matchup(df_tr, X_tr)
    X_val_new = apply_matchup(df_val, X_val)
    return X_tr_new, X_val_new

# --- Experiments ---
results = {}

# Baseline (SSOT confirmation)
print("\n[Baseline] Confirming SSOT baseline...")
r_base = run_standard_sota_evaluation(df_train, strict_as_of=True)
results['baseline'] = r_base
print(f"  Baseline: Skill={r_base['mean_fold_skill']:.2f}점, Brier={r_base['overall_raw_brier']:.6f}")

# Matchup with m=30
print("\n[Exp A] Matchup feature, m=30...")
fn_m30 = lambda df_tr, df_val, fms, X_tr, X_val: build_matchup_feature(df_tr, df_val, fms, X_tr, X_val, m_smooth=30)
r_m30 = run_standard_sota_evaluation(df_train, strict_as_of=True, extra_feature_fn=fn_m30)
results['matchup_m30'] = r_m30
print(f"  m=30: Skill={r_m30['mean_fold_skill']:.2f}점, Brier={r_m30['overall_raw_brier']:.6f}")

# Matchup with m=50
print("\n[Exp B] Matchup feature, m=50...")
fn_m50 = lambda df_tr, df_val, fms, X_tr, X_val: build_matchup_feature(df_tr, df_val, fms, X_tr, X_val, m_smooth=50)
r_m50 = run_standard_sota_evaluation(df_train, strict_as_of=True, extra_feature_fn=fn_m50)
results['matchup_m50'] = r_m50
print(f"  m=50: Skill={r_m50['mean_fold_skill']:.2f}점, Brier={r_m50['overall_raw_brier']:.6f}")

# Matchup with m=100 (very strong smoothing)
print("\n[Exp C] Matchup feature, m=100...")
fn_m100 = lambda df_tr, df_val, fms, X_tr, X_val: build_matchup_feature(df_tr, df_val, fms, X_tr, X_val, m_smooth=100)
r_m100 = run_standard_sota_evaluation(df_train, strict_as_of=True, extra_feature_fn=fn_m100)
results['matchup_m100'] = r_m100
print(f"  m=100: Skill={r_m100['mean_fold_skill']:.2f}점, Brier={r_m100['overall_raw_brier']:.6f}")

elapsed = time.time() - t0
from datetime import datetime
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

lines = []
lines.append(f"# 115. 투수-타자 매치업 이력 피처 실험 보고서\n")
lines.append(f"- **작성 일시**: {NOW_STR}")
lines.append(f"- **실험 시간**: {elapsed/60:.1f}분\n")
lines.append(f"- **SSOT 기준**: 850.09점 / Raw Brier 0.247538 (strict_as_of=True)\n")
lines.append("---\n")
lines.append("## 1. 피처 설계\n")
lines.append("```")
lines.append("matchup_rate(pitcher, batter) = (Σ control_success + m × global_rate) / (n + m)")
lines.append("단, 학습 데이터 중 season <= fold_max_season만 사용 (leakage 완전 차단)")
lines.append("```\n")
lines.append("## 2. 실험 결과 요약\n")
lines.append("| 실험 | Smoothing m | 3-Fold Skill | Overall Raw Brier | SSOT 대비 |")
lines.append("|:---:|:---:|:---:|:---:|:---:|")
for key, label, r in [
    ('baseline', 'Baseline (no matchup)', results['baseline']),
    ('matchup_m30', 'Matchup m=30', results['matchup_m30']),
    ('matchup_m50', 'Matchup m=50', results['matchup_m50']),
    ('matchup_m100', 'Matchup m=100', results['matchup_m100']),
]:
    sk = r['mean_fold_skill']
    br = r['overall_raw_brier']
    delta = sk - SSOT_SKILL
    sign = '+' if delta >= 0 else ''
    m_val = label.split('m=')[-1] if 'm=' in label else '—'
    lines.append(f"| {label} | {m_val} | `{sk:.2f}점` | `{br:.6f}` | `{sign}{delta:.2f}점` |")

lines.append("\n## 3. Fold별 상세 결과\n")
for key, label, r in [
    ('baseline', 'Baseline', results['baseline']),
    ('matchup_m30', 'Matchup m=30', results['matchup_m30']),
    ('matchup_m50', 'Matchup m=50', results['matchup_m50']),
    ('matchup_m100', 'Matchup m=100', results['matchup_m100']),
]:
    lines.append(f"### {label}")
    lines.append("| Fold | Val Season | Raw Brier | Skill Score |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for fd in r['fold_details']:
        lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")
    lines.append("")

lines.append("## 4. 결론\n")
best_key = max(['baseline','matchup_m30','matchup_m50','matchup_m100'],
               key=lambda k: results[k]['mean_fold_skill'])
best_r = results[best_key]
if best_key == 'baseline':
    lines.append("> ❌ **매치업 피처 기각**: 어떤 smoothing 강도에서도 baseline 대비 개선 없음.")
else:
    delta = best_r['mean_fold_skill'] - SSOT_SKILL
    lines.append(f"> ✅ **매치업 피처 채택**: `{best_key}` 구성이 SSOT 대비 {delta:+.2f}점 개선.")

with open(OUTPUTS_DIR / '115_pitcher_batter_matchup.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport 115 written!")
print(f"Results: baseline={results['baseline']['mean_fold_skill']:.2f}, "
      f"m30={results['matchup_m30']['mean_fold_skill']:.2f}, "
      f"m50={results['matchup_m50']['mean_fold_skill']:.2f}, "
      f"m100={results['matchup_m100']['mean_fold_skill']:.2f}")
