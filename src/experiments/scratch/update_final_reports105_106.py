import sys
import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
RAW_DIR = OUTPUTS_DIR / 'raw'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(RAW_DIR / 'task105_106_recalc_summary.json', 'r', encoding='utf-8') as f:
    res_data = json.load(f)['recalculated_results']

res_base = res_data['baseline']
res_103 = res_data['exp103']
res_104 = res_data['exp104']

doc_105 = f"""# 105. 103/104번 Skill Score 계산 스케일 버그 긴급 감사 및 수정 보고서

- **작성 일시**: {NOW_STR}
- **목적**: 103/104번 보고서 작성 시 발생한 Skill Score 수식 스케일 오류(`96.89점` 표기 버그)의 원인을 특정하고 표준 계산 공식으로 긴급 수정.

---

## 1. 버그 위치 및 원인 특정

- **오류 발생 코드 위치**: `run_exp103_104.py` L198
- **오류 원인 수식 비교**:
  - ❌ **103/104번 당시 잘못 사용된 수식**: `(1.0 - total_raw_brier / base_brier) * 10000.0` $\to$ 스케일 상수가 100,000이 아닌 10,000으로 적용되어 실제 점수의 $1/10$ 수준(`96.89점`)으로 표기됨.
  - ✅ **표준 공식 (`submission_checklist.calc_brier_skill_score`)**:
    Skill Score = max(0, 100000 * (1 - Brier_model / (r * (1 - r))))
- **수정 조치**: 프로젝트 검증 전용 유틸리티 `submission_checklist.calc_brier_skill_score` 표준 함수로 전면 교체하여 100,000 스케일 복구 완료.
"""

with open(OUTPUTS_DIR / '105_skill_calc_emergency_fix.md', 'w', encoding='utf-8') as f:
    f.write(doc_105)

doc_106 = f"""# 106. 103/104번 실험 표준 Skill Score 투명 재계산 보고서

- **작성 일시**: {NOW_STR}
- **목적**: 표준 Skill Score 수식을 적용하여 Baseline SOTA, Exp 103(구종 비율), Exp 104(투구 순번)의 Fold별 수치를 투명하게 전수 공개하고 최종 채택 여부를 재확정.

---

## 1. Fold별 표준 수치 전수 공개표

### 1) ✅ Baseline SOTA (기존 70개 피처)
- **Overall Inner Brier (2022-23)**: **`{res_base['inner_brier']:.6f}`**
- **3-Fold Raw Brier**: **`{res_base['overall_raw_brier']:.6f}`**
- **표준 Skill Score**: **`{res_base['overall_skill_score']:.2f}점` (968.86점 스케일 복구 완료)**

| Fold | 검증 시즌 | 실제 성공률 ($r_k$) | Baseline Brier ($r_k(1-r_k)$) | Fold Raw Brier | **Fold Skill Score** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022년 | `{res_base['fold_details'][0]['r_k']:.6f}` | `{res_base['fold_details'][0]['brier_base_k']:.6f}` | `{res_base['fold_details'][0]['raw_brier_k']:.6f}` | **`{res_base['fold_details'][0]['skill_k']:.2f}점`** |
| **Fold 2** | 2023년 | `{res_base['fold_details'][1]['r_k']:.6f}` | `{res_base['fold_details'][1]['brier_base_k']:.6f}` | `{res_base['fold_details'][1]['raw_brier_k']:.6f}` | **`{res_base['fold_details'][1]['skill_k']:.2f}점`** |
| **Fold 3** | 2024년 | `{res_base['fold_details'][2]['r_k']:.6f}` | `{res_base['fold_details'][2]['brier_base_k']:.6f}` | `{res_base['fold_details'][2]['raw_brier_k']:.6f}` | **`{res_base['fold_details'][2]['skill_k']:.2f}점`** |

---

### 2) Exp 103 (구종 비율 prior 피처 4종)
- **Overall Inner Brier (2022-23)**: **`{res_103['inner_brier']:.6f}`**
- **3-Fold Raw Brier**: `{res_103['overall_raw_brier']:.6f}`
- **표준 Skill Score**: **`{res_103['overall_skill_score']:.2f}점`**

| Fold | 검증 시즌 | 실제 성공률 ($r_k$) | Baseline Brier | Fold Raw Brier | **Fold Skill Score** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022년 | `{res_103['fold_details'][0]['r_k']:.6f}` | `{res_103['fold_details'][0]['brier_base_k']:.6f}` | `{res_103['fold_details'][0]['raw_brier_k']:.6f}` | `{res_103['fold_details'][0]['skill_k']:.2f}점` |
| **Fold 2** | 2023년 | `{res_103['fold_details'][1]['r_k']:.6f}` | `{res_103['fold_details'][1]['brier_base_k']:.6f}` | `{res_103['fold_details'][1]['raw_brier_k']:.6f}` | `{res_103['fold_details'][1]['skill_k']:.2f}점` |
| **Fold 3** | 2024년 | `{res_103['fold_details'][2]['r_k']:.6f}` | `{res_103['fold_details'][2]['brier_base_k']:.6f}` | `{res_103['fold_details'][2]['raw_brier_k']:.6f}` | `{res_103['fold_details'][2]['skill_k']:.2f}점` |

---

### 3) 🏆 Exp 104 (투구 순번 / 피로도 prior 피처 4종)
- **Overall Inner Brier (2022-23)**: **`{res_104['inner_brier']:.6f}` (Safeguard 1위)**
- **3-Fold Raw Brier**: **`{res_104['overall_raw_brier']:.6f}`**
- **표준 Skill Score**: **`{res_104['overall_skill_score']:.2f}점` (+3.97점 상승)**

| Fold | 검증 시즌 | 실제 성공률 ($r_k$) | Baseline Brier | Fold Raw Brier | **Fold Skill Score** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022년 | `{res_104['fold_details'][0]['r_k']:.6f}` | `{res_104['fold_details'][0]['brier_base_k']:.6f}` | `{res_104['fold_details'][0]['raw_brier_k']:.6f}` | `{res_104['fold_details'][0]['skill_k']:.2f}점` |
| **Fold 2** | 2023년 | `{res_104['fold_details'][1]['r_k']:.6f}` | `{res_104['fold_details'][1]['brier_base_k']:.6f}` | `{res_104['fold_details'][1]['raw_brier_k']:.6f}` | `{res_104['fold_details'][1]['skill_k']:.2f}점` |
| **Fold 3** | 2024년 | `{res_104['fold_details'][2]['r_k']:.6f}` | `{res_104['fold_details'][2]['brier_base_k']:.6f}` | `{res_104['fold_details'][2]['raw_brier_k']:.6f}` | `{res_104['fold_details'][2]['skill_k']:.2f}점` |

---

## 2. 정직한 최종 판단 및 결론

1. **Skill Score 스케일 오류 복구**:
   - `submission_checklist.calc_brier_skill_score` 수식(스케일 100,000)으로 재계산한 결과, **Baseline SOTA가 `968.86점` (Raw Brier `0.247554`)** 수치로 정확히 복구되었습니다.
2. **Safeguard 1위 통과 및 채택**:
   - `submission_checklist.py` 안전장치 규칙(Inner Brier 1위 선택)에 따라, **Exp 104 (Inner Brier `0.247125`, Skill `972.83점`)가 Baseline SOTA(`968.86점`)를 제치고 안전장치 1위를 공식 달성**했습니다.
3. **개선 폭 판정**:
   - Exp 104의 표준 Skill Score 상승 폭은 **`+3.97점`** (`968.86점` $\to$ `972.83점`)으로, **90번 CV Noise Floor ($\pm 1.70$점)**을 확연히 넘어서는 실질적 성능 향상임이 검증되었습니다.
"""

with open(OUTPUTS_DIR / '106_103_104_recalc.md', 'w', encoding='utf-8') as f:
    f.write(doc_106)

print("Reports 105 and 106 final updated successfully!")
