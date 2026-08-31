"""
run_exp114_multiseed_bagging.py
작업 1: 멀티시드 배깅 실험
seed=3, 5, 10개로 배깅 앙상블 검증 (strict_as_of=True)
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from core.eval_utils import run_standard_sota_evaluation

OUTPUTS_DIR = Path('~/LG_data/outputs')
import config

print("=== Task 1: Multi-Seed Bagging Experiment ===")
t0 = time.time()

df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train data: {len(df_train):,} rows")

SSOT_SKILL = 850.09
SSOT_BRIER = 0.247538

results = {}

# Baseline: single seed 42
print("\n[Baseline] Single seed (42)...")
r0 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=[42])
results['seed_1'] = r0
print(f"  Skill={r0['mean_fold_skill']:.2f}점, Brier={r0['overall_raw_brier']:.6f}")

# 3 seeds
SEEDS_3 = [42, 123, 777]
print(f"\n[3 Seeds] {SEEDS_3}...")
r3 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=SEEDS_3)
results['seed_3'] = r3
print(f"  Skill={r3['mean_fold_skill']:.2f}점, Brier={r3['overall_raw_brier']:.6f}")

# 5 seeds
SEEDS_5 = [42, 123, 777, 2024, 314]
print(f"\n[5 Seeds] {SEEDS_5}...")
r5 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=SEEDS_5)
results['seed_5'] = r5
print(f"  Skill={r5['mean_fold_skill']:.2f}점, Brier={r5['overall_raw_brier']:.6f}")

# 10 seeds
SEEDS_10 = [42, 123, 777, 2024, 314, 999, 1234, 5678, 11, 73]
print(f"\n[10 Seeds] {SEEDS_10}...")
r10 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=SEEDS_10)
results['seed_10'] = r10
print(f"  Skill={r10['mean_fold_skill']:.2f}점, Brier={r10['overall_raw_brier']:.6f}")

elapsed = time.time() - t0

# Write report
from datetime import datetime
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

lines = []
lines.append(f"# 114. 멀티시드 배깅 실험 보고서\n")
lines.append(f"- **작성 일시**: {NOW_STR}")
lines.append(f"- **실험 시간**: {elapsed/60:.1f}분\n")
lines.append(f"- **SSOT 기준**: 850.09점 / Raw Brier 0.247538 (strict_as_of=True)\n")
lines.append("---\n")
lines.append("## 1. 실험 결과 요약\n")
lines.append("| 시드 수 | Seeds | 3-Fold Mean Skill | Overall Raw Brier | SSOT 대비 개선폭 |")
lines.append("|:---:|:---:|:---:|:---:|:---:|")

for key, r, seeds in [
    ('seed_1', results['seed_1'], [42]),
    ('seed_3', results['seed_3'], SEEDS_3),
    ('seed_5', results['seed_5'], SEEDS_5),
    ('seed_10', results['seed_10'], SEEDS_10),
]:
    skill = r['mean_fold_skill']
    brier = r['overall_raw_brier']
    delta = skill - SSOT_SKILL
    sign = '+' if delta >= 0 else ''
    lines.append(f"| {len(seeds)}개 | {seeds[:3]}{'...' if len(seeds)>3 else ''} | `{skill:.2f}점` | `{brier:.6f}` | `{sign}{delta:.2f}점` |")

lines.append("\n## 2. Fold별 상세 결과\n")
for key, r, seeds in [
    ('seed_1', results['seed_1'], [42]),
    ('seed_3', results['seed_3'], SEEDS_3),
    ('seed_5', results['seed_5'], SEEDS_5),
    ('seed_10', results['seed_10'], SEEDS_10),
]:
    lines.append(f"### {len(seeds)}개 시드")
    lines.append("| Fold | Val Season | Raw Brier | Skill Score |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for fd in r['fold_details']:
        lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")
    lines.append(f"| **Mean** | — | `{r['overall_raw_brier']:.6f}` | **`{r['mean_fold_skill']:.2f}점`** |")
    lines.append("")

lines.append("## 3. 결론 및 수확체감 분석\n")
s1 = results['seed_1']['mean_fold_skill']
s3 = results['seed_3']['mean_fold_skill']
s5 = results['seed_5']['mean_fold_skill']
s10 = results['seed_10']['mean_fold_skill']
best_skill = max(s1, s3, s5, s10)
best_k = [1, 3, 5, 10][[s1, s3, s5, s10].index(best_skill)]
lines.append(f"- **최고 성능**: {best_k}개 시드, Skill={best_skill:.2f}점 (SSOT 대비 {best_skill-SSOT_SKILL:+.2f}점)")
lines.append(f"- 1→3 개선폭: `{s3-s1:+.2f}점`")
lines.append(f"- 3→5 개선폭: `{s5-s3:+.2f}점`")
lines.append(f"- 5→10 개선폭: `{s10-s5:+.2f}점`")
lines.append(f"- 수확체감 확인: {'Yes — 시드 증가 효과 감소' if (s10-s5) < (s3-s1) else 'No — 계속 개선 중'}")
if best_skill > SSOT_SKILL:
    lines.append(f"\n> ✅ **멀티시드 배깅 채택**: {best_k}개 시드 구성이 SSOT 대비 {best_skill-SSOT_SKILL:+.2f}점 개선 확인")
else:
    lines.append(f"\n> ❌ **멀티시드 배깅 기각**: 단일 시드 대비 유의미한 개선 없음")

with open(OUTPUTS_DIR / '114_multiseed_bagging.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport 114 written: {OUTPUTS_DIR}/114_multiseed_bagging.md")
print(f"FINAL: seed=1:{s1:.2f}, seed=3:{s3:.2f}, seed=5:{s5:.2f}, seed=10:{s10:.2f}")
