"""
run_task4_final_sota.py
작업 4: 종합 앙상블 재구성 및 최종 공식 SOTA 확정 보고서 (117_final_ensemble.md)
"""
import sys, os, time, json
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from core.eval_utils import run_standard_sota_evaluation
import config

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

df_train = pd.read_csv(config.TRAIN_PATH)

print("=== Task 4: Final Consolidated SOTA Verification ===")

# Old SSOT
OLD_SSOT_SKILL = 850.09
OLD_SSOT_BRIER = 0.247538
TARGET_SCORE = 1100.00

# Best model_params from Task 3
best_model_params = {'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7}}

print(f"Running strict_as_of=True evaluation with best model_params: {best_model_params}...")
res_final = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=best_model_params)

final_skill = res_final['mean_fold_skill']
final_brier = res_final['overall_raw_brier']

print(f"\nFinal Verified SOTA:")
print(f"  3-Fold Mean Skill Score: {final_skill:.2f}점")
print(f"  Overall Raw Brier Score: {final_brier:.6f}")
print(f"  Improvement over 850.09점: {final_skill - OLD_SSOT_SKILL:+.2f}점")
print(f"  Remaining Gap to 1100점: {TARGET_SCORE - final_skill:.2f}점")

lines = []
lines.append(f"# 117. 방향성 3종 시도 종합 및 최종 공식 SOTA 확정 보고서\n")
lines.append(f"- **작성 일시**: {NOW_STR}")
lines.append(f"- **검증 엔진**: `core/eval_utils.py` (`strict_as_of=True`, leakage 0%)")
lines.append(f"- **기존 공식 SSOT**: `850.09점` / Raw Brier `0.247538`\n")
lines.append("---\n")

lines.append("## 1. 세 가지 새로운 방향성 시도 결과 요약\n")
lines.append("| 방향 | 시도 내용 | 결과 | 판정 | 비고 |")
lines.append("|:---:|:---|:---:|:---:|:---|")
lines.append("| **작업 1** | 멀티시드 배깅 (seed=3, 5) | `846.36점` ~ `846.53점` | **REJECTED** | 시드 평균 시 확률 평탄화로 Brier 미세 악화 |")
lines.append("| **작업 2** | 투수-타자 매치업 이력 (m=30~100) | `0.00점` | **REJECTED** | 희소 매치업 피처의 심각한 과적합/일반화 실패 |")
lines.append("| **작업 3** | 확장 하이퍼파라미터 탐색 (16개 후보) | **`853.24점`** | **ACCEPTED ✅** | LightGBM `colsample_bytree=0.7, subsample=0.7` |")

lines.append("\n---\n")
lines.append("## 2. 최종 공식 SOTA (New SSOT) 수치\n")
lines.append(f"- **최종 3-Fold Mean Skill Score**: **`{final_skill:.2f}점`**")
lines.append(f"- **Overall Raw Brier**: **`{final_brier:.6f}`**")
lines.append(f"- **기존 850.09점 대비 개선폭**: **`{final_skill - OLD_SSOT_SKILL:+.2f}점`**")
lines.append(f"- **목표 점수 (1100.00점)까지 남은 거리**: **`{TARGET_SCORE - final_skill:.2f}점`**\n")

lines.append("### Fold별 상세 성능 대조표\n")
lines.append("| Fold | 검증 시즌 | $r_k$ (실제성공률) | Raw Brier (기존) | Raw Brier (최종) | Skill Score (기존) | **Skill Score (최종)** |")
lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

old_fds = [
    {"fold": 1, "val": 2022, "r": 0.528920, "brier": 0.244543, "skill": 1854.48},
    {"fold": 2, "val": 2023, "r": 0.499957, "brier": 0.249737, "skill": 105.12},
    {"fold": 3, "val": 2024, "r": 0.486105, "brier": 0.248331, "skill": 590.67},
]

for old_fd, new_fd in zip(old_fds, res_final['fold_details']):
    lines.append(f"| {new_fd['fold']} | {new_fd['val_season']}년 | `{new_fd['r_k']:.6f}` | `{old_fd['brier']:.6f}` | `{new_fd['raw_brier_k']:.6f}` | `{old_fd['skill']:.2f}점` | **`{new_fd['skill_k']:.2f}점`** |")

lines.append(f"| **평균** | — | — | `{OLD_SSOT_BRIER:.6f}` | **`{final_brier:.6f}`** | `{OLD_SSOT_SKILL:.2f}점` | **`{final_skill:.2f}점`** |")

lines.append("\n---\n")
lines.append("## 3. 최종 모델 스펙 및 핵심 구현\n")
lines.append("### 3.1 하이퍼파라미터 구성\n")
lines.append("```python")
lines.append("# LightGBM (최적 오버라이드)")
lines.append("lgb_params = {")
lines.append("    'n_estimators': 250, 'num_leaves': 45, 'learning_rate': 0.05,")
lines.append("    'min_child_samples': 20, 'colsample_bytree': 0.7, 'subsample': 0.7,")
lines.append("    'random_state': 42")
lines.append("}")
lines.append("")
lines.append("# CatBoost (SSOT baseline)")
lines.append("cb_params = {")
lines.append("    'iterations': 250, 'depth': 6, 'learning_rate': 0.05,")
lines.append("    'l2_leaf_reg': 10.0, 'random_seed': 42")
lines.append("}")
lines.append("")
lines.append("# XGBoost (SSOT baseline)")
lines.append("xgb_params = {")
lines.append("    'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05,")
lines.append("    'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42")
lines.append("}")
lines.append("```\n")

lines.append("### 3.2 앙상블 및 Shift 가중치\n")
lines.append("- **앙상블 가중치**: LightGBM `20%` + CatBoost `70%` + XGBoost `10%`")
lines.append("- **독립 Shift 오프셋**: LightGBM `-0.007`, CatBoost `-0.008`, XGBoost `-0.006`")
lines.append("- **핵심 교차 피처**: `count_x_base` (볼카운트 × 주자상황 카테고리 교차)\n")

lines.append("---\n")
lines.append("## 4. 로드맵 및 향후 1100점 달성 전략\n")
lines.append(f"1. **현황**: 정직한 strict CV 기준으로 **`{final_skill:.2f}점`** 달성 (목표 1100점까지 **`{TARGET_SCORE - final_skill:.2f}점`** 남음).")
lines.append("2. **시사점**: 피처 추가(매치업)나 단순 배깅(시드)보다는 **피처 서브샘플링(`colsample_bytree=0.7`)과 정밀 하이퍼파라미터 조절**이 오차 축소에 효과적임.")
lines.append("3. **차기 시도 방향**: 모델별 독립 weights/shift 조절, 타자 세그먼트별 조건부 캘리브레이션, 초구/풀카운트 전용 서브모델 앙상블 등.")

report_path = OUTPUTS_DIR / '117_final_ensemble.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nReport 117 successfully written to: {report_path}")

# Also update 00_summary.md with the new SOTA
summary_path = OUTPUTS_DIR / '00_summary.md'
summary_update = f"""

---

## 🏆 [최신 공식 SOTA 확정 - 보고서 117, {NOW_STR}]

- **공식 SOTA**: **`{final_skill:.2f}점`** / Raw Brier **`{final_brier:.6f}`** (`strict_as_of=True`, `core/eval_utils.py`)
- **개선 요인**: LightGBM 피처 서브샘플링 최적화 (`colsample_bytree=0.7, subsample=0.7`)
- **이전 SSOT(850.09점) 대비**: **`{final_skill - OLD_SSOT_SKILL:+.2f}점`** 상승
- **목표(1100점)까지 남은 거리**: **`{TARGET_SCORE - final_skill:.2f}점`**
"""

with open(summary_path, 'a', encoding='utf-8') as f:
    f.write(summary_update)

print("00_summary.md updated with new SOTA!")
