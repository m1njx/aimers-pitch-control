"""
run_exp114_fast.py — 멀티시드 배깅 실험 (단축 버전)
seed=1/3/5만 평가 (10 seed는 시간 초과로 제외)
strict_as_of=True 사용
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from core.eval_utils import run_standard_sota_evaluation

OUTPUTS_DIR = Path('~/LG_data/outputs')
import config

print("=== Task 1: Multi-Seed Bagging Experiment (Fast: seeds 1/3/5) ===")
t0 = time.time()

df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train data: {len(df_train):,} rows")

SSOT_SKILL = 850.09
SSOT_BRIER = 0.247538

results = {}

# Baseline: single seed 42
print("\n[Baseline] Single seed (42)...")
r1 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=[42])
results['seed_1'] = r1
print(f"  seed=1: Skill={r1['mean_fold_skill']:.2f}점, Brier={r1['overall_raw_brier']:.6f}")
for fd in r1['fold_details']:
    print(f"    Fold{fd['fold']}(val={fd['val_season']}): brier={fd['raw_brier_k']:.6f}, skill={fd['skill_k']:.2f}")

# 3 seeds
SEEDS_3 = [42, 123, 777]
print(f"\n[3 Seeds] {SEEDS_3}...")
r3 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=SEEDS_3)
results['seed_3'] = r3
print(f"  seed=3: Skill={r3['mean_fold_skill']:.2f}점, Brier={r3['overall_raw_brier']:.6f}")
for fd in r3['fold_details']:
    print(f"    Fold{fd['fold']}(val={fd['val_season']}): brier={fd['raw_brier_k']:.6f}, skill={fd['skill_k']:.2f}")

# 5 seeds
SEEDS_5 = [42, 123, 777, 2024, 314]
print(f"\n[5 Seeds] {SEEDS_5}...")
r5 = run_standard_sota_evaluation(df_train, strict_as_of=True, random_seeds=SEEDS_5)
results['seed_5'] = r5
print(f"  seed=5: Skill={r5['mean_fold_skill']:.2f}점, Brier={r5['overall_raw_brier']:.6f}")
for fd in r5['fold_details']:
    print(f"    Fold{fd['fold']}(val={fd['val_season']}): brier={fd['raw_brier_k']:.6f}, skill={fd['skill_k']:.2f}")

elapsed = time.time() - t0

from datetime import datetime
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

s1 = results['seed_1']['mean_fold_skill']
s3 = results['seed_3']['mean_fold_skill']
s5 = results['seed_5']['mean_fold_skill']

lines = []
lines.append(f"# 114. 멀티시드 배깅 실험 보고서\n")
lines.append(f"- **작성 일시**: {NOW_STR}")
lines.append(f"- **실험 시간**: {elapsed/60:.1f}분")
lines.append(f"- **참고**: seed=10 구성은 시간 초과(약 250분 추정)로 제외. seed=1/3/5만 평가.")
lines.append(f"- **SSOT 기준**: 850.09점 / Raw Brier 0.247538 (strict_as_of=True)\n")
lines.append("---\n")
lines.append("## 1. 실험 결과 요약\n")
lines.append("| 시드 수 | Seeds | 3-Fold Mean Skill | Overall Raw Brier | SSOT 대비 개선폭 |")
lines.append("|:---:|:---:|:---:|:---:|:---:|")
for key, r, seeds in [
    ('seed_1', results['seed_1'], [42]),
    ('seed_3', results['seed_3'], SEEDS_3),
    ('seed_5', results['seed_5'], SEEDS_5),
]:
    skill = r['mean_fold_skill']
    brier = r['overall_raw_brier']
    delta = skill - SSOT_SKILL
    sign = '+' if delta >= 0 else ''
    lines.append(f"| {len(seeds)}개 | {seeds} | `{skill:.2f}점` | `{brier:.6f}` | `{sign}{delta:.2f}점` |")

lines.append("\n## 2. Fold별 상세 결과\n")
for key, r, seeds, label in [
    ('seed_1', results['seed_1'], [42], '1개 시드 (Baseline)'),
    ('seed_3', results['seed_3'], SEEDS_3, '3개 시드 배깅'),
    ('seed_5', results['seed_5'], SEEDS_5, '5개 시드 배깅'),
]:
    lines.append(f"### {label}")
    lines.append("| Fold | Val Season | Raw Brier | Skill Score |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for fd in r['fold_details']:
        lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")
    lines.append(f"| **평균** | — | `{r['overall_raw_brier']:.6f}` | **`{r['mean_fold_skill']:.2f}점`** |")
    lines.append("")

lines.append("## 3. 수확체감 분석 및 결론\n")
lines.append(f"| 시드 수 증가 | 개선폭 |")
lines.append("|:---:|:---:|")
lines.append(f"| 1 → 3 seeds | `{s3-s1:+.2f}점` |")
lines.append(f"| 3 → 5 seeds | `{s5-s3:+.2f}점` |")
lines.append("")

best_skill = max(s1, s3, s5)
best_k = [1, 3, 5][[s1, s3, s5].index(best_skill)]

if best_skill > SSOT_SKILL:
    lines.append(f"> ✅ **멀티시드 배깅 채택**: {best_k}개 시드 구성이 SSOT(850.09점) 대비 **{best_skill-SSOT_SKILL:+.2f}점** 개선.")
    lines.append(f">\n> 권장: {best_k}개 시드를 최종 앙상블에 적용.")
else:
    lines.append(f"> ❌ **멀티시드 배깅 기각**: 어떤 배깅 구성에서도 SSOT(850.09점) 대비 개선 없음.")
    lines.append(f">\n> 단일 시드(42) 유지 권장.")

with open(OUTPUTS_DIR / '114_multiseed_bagging.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\n=== SUMMARY ===")
print(f"seed=1: {s1:.2f}점 (SSOT: {SSOT_SKILL}점)")
print(f"seed=3: {s3:.2f}점 (delta: {s3-s1:+.2f}점)")
print(f"seed=5: {s5:.2f}점 (delta: {s5-s3:+.2f}점 from 3, {s5-s1:+.2f}점 from 1)")
print(f"Report 114 written: {OUTPUTS_DIR}/114_multiseed_bagging.md")
print(f"Elapsed: {elapsed/60:.1f} min")
