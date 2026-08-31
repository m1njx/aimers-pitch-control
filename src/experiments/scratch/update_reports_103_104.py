import sys
import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
RAW_DIR = OUTPUTS_DIR / 'raw'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(RAW_DIR / 'task103_104_summary.json', 'r', encoding='utf-8') as f:
    res_data = json.load(f)

res_base = res_data['baseline']
res_103 = res_data['exp103']
res_104 = res_data['exp104']

doc_103 = f"""# 103. trackman_history.csv 구종 분포 prior 피처 추가 실측 보고서

- **작성 일시**: {NOW_STR}
- **목적**: `trackman_history.csv`의 미활용 `pitch_type_group` 컬럼을 7-key 조인 기반 상황별 구종 비율(Fastball, Breaking, Offspeed 등) prior 피처로 동적 생성하여 CV 개선 여부를 실측.

---

## 1. 피처 생성 사양 및 누수(Leakage) 방지 검증
- **추가 피처 4종**: `tkm_pt_ratio_Fastball`, `tkm_pt_ratio_Breaking`, `tkm_pt_ratio_Offspeed`, `tkm_pt_ratio_Other`
- **누수 방지**: `season <= fold_max_season` strictly as-of 집계 필터링 준수.
- **표본 수 및 매칭률**: 상황별 42,267개 집계 그룹에서 100% 정상 추출 (매칭률 99.3%~99.9%).

---

## 2. Nested Validation (Inner Brier 22-23) 실측 비교표

| 모델 / 피처 설정 | Inner Brier (2022-23) | 2024 Held-Out Brier | 3-Fold Raw Brier | **Skill Score** | **Safeguard 판정** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline SOTA (기존 70피처) | `{res_base['inner_brier']:.6f}` | `{res_base['brier_2024']:.6f}` | `{res_base['total_raw_brier']:.6f}` | `{res_base['skill_score']:.2f}점` | Baseline |
| **Exp 103 (구종 비율 prior)** | **`{res_103['inner_brier']:.6f}`** | `{res_103['brier_2024']:.6f}` | **`{res_103['total_raw_brier']:.6f}`** | **`{res_103['skill_score']:.2f}점`** | **✅ Inner Brier 미세 개선 (+0.16점)** |

---

## 3. 원인 분석 및 판단
- Inner Brier가 `0.247155`에서 `0.247152`로 미세하게 좋아졌으나, 개선 폭이 `+0.16점`으로 **90번 CV Noise Floor ($\pm 1.70$점)** 이내에 존재합니다.
- 기존 속도/회전수 prior 피처와 일부 중복되어 개선 폭이 제한적입니다.
"""

with open(OUTPUTS_DIR / '103_pitch_type_prior.md', 'w', encoding='utf-8') as f:
    f.write(doc_103)

doc_104 = f"""# 104. 타석/경기 내 투구 순번 prior 피처 추가 실측 보고서

- **작성 일시**: {NOW_STR}
- **목적**: `trackman_history.csv`의 `pitch_no`(경기 내 누적 투구 순번) 및 `pitch_of_pa`(타석 내 투구 순번) 컬럼을 상황별 prior 평균치로 추출하여 투수 피로도 신호로 반영 가능한지 실측.

---

## 1. 피처 생성 사양 및 누수(Leakage) 방지 검증
- **추가 피처 4종**: `tkm_pitch_no_mean`, `tkm_pitch_no_std`, `tkm_pitch_of_pa_mean`, `tkm_pitch_of_pa_max`
- **누수 방지**: `season <= fold_max_season` strictly as-of 집계 준수.

---

## 2. Nested Validation (Inner Brier 22-23) 실측 비교표

| 모델 / 피처 설정 | Inner Brier (2022-23) | 2024 Held-Out Brier | 3-Fold Raw Brier | **Skill Score** | **Safeguard 판정** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline SOTA (기존 70피처) | `{res_base['inner_brier']:.6f}` | **`{res_base['brier_2024']:.6f}`** | `{res_base['total_raw_brier']:.6f}` | `{res_base['skill_score']:.2f}점` | 2위 |
| **✅ Exp 104 (투구순번/피로도 prior)** | **`{res_104['inner_brier']:.6f}`** | `{res_104['brier_2024']:.6f}` | **`{res_104['total_raw_brier']:.6f}`** | **`{res_104['skill_score']:.2f}점`** | **🏆 Safeguard 1위 통과 (+0.39점)** |

---

## 3. 원인 분석 및 종합 판단

1. **Inner Brier 안전장치 통과**:
   - `submission_checklist.py` 안전장치 규칙(Inner Brier 1위 기준 정렬)에 따라, **Exp 104 (Inner Brier `0.247125`)가 Baseline(`0.247155`) 및 Exp 103(`0.247152`)을 제치고 1위를 차지하여 안전장치를 정상 통과**했습니다.
2. **Held-out (2024) 및 Noise Floor 분석**:
   - Skill Score 수치가 `96.89점`에서 `97.28점`으로 `+0.39점` 미세 상승했습니다.
   - 단, 2024 Held-Out Brier는 `0.248325`에서 `0.248360`으로 인근 미세 악화되었고, 상승폭(`+0.39점`)이 **90번 CV Noise Floor ($\pm 1.70$점)** 범위 안쪽에 위치하므로 통계적으로 확고한 진성 개선으로 단정하기는 어렵습니다.
3. **다음 방향 제안**:
   - 현 baseline 파이프라인의 70개 피처는 이미 트랙맨 및 상황 정보의 대부분을 효율적으로 압축하고 있습니다. 향후 수치상 미세 개선보다는 더 높은 통계적 변동을 일으키는 파라미터 튜닝이나 앙상블 조합 탐색을 제안합니다.
"""

with open(OUTPUTS_DIR / '104_pitch_sequence_number.md', 'w', encoding='utf-8') as f:
    f.write(doc_104)

print("Reports 103 and 104 updated with exact figures successfully!")
