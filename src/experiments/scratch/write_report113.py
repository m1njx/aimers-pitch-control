import sys
import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

doc_113 = f"""# 113. 859.63점 수치 기원 추적 및 정직한 SSOT 재출발점 확정 보고서

- **작성 일시**: {NOW_STR}

---

## 1. [작업 1] 859.63점 최초 발생 지점 추적 결과

### 원본 스크립트 특정
- **파일**: `~/LG_data/scratch/run_feature_interaction_exp.py`
- **69번 보고서 (count_x_base 채택 결정)**에서 `Candidate 4` 평가 시 사용된 원본 스크립트.

### 원본 스크립트의 as_of_season 처리 방식 (코드 직접 인용)
```python
# run_feature_interaction_exp.py Line 56
prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
```
- **Fold별 as_of_season 할당 방식**: `fold.fold_max_season` (strict_as_of=True와 동일!)
- 따라서 859.63점은 `as_of_season=fold.fold_max_season` 방식으로 도출된 수치였음.

### 현재 strict_as_of=True 재실행 결과: 850.09점

이는 859.63점을 도출한 코드 경로(fold.fold_max_season)와 완전히 동일한 방식이지만 **결과가 다름**.

### 불일치 원인 분석
두 실행 간에 달라진 것이 있다면:
- **`PitchPreprocessor`나 `TrackmanFeatureBuilder` 구현 변경** 가능성
- **파이프라인 의존성 변경** (`config.MODEL_FEATURE_COLS`, categorical feature 목록 등)
- 또는 **69번 당시 스크립트에서 일부 추가 설정**이 있었을 가능성

### 결론
859.63점을 도출한 `run_feature_interaction_exp.py`가 현재 코드베이스에서 동일하게 실행되면 850.09점이 나옵니다.  
즉, **859.63점은 파이프라인 코드가 달랐던 시점의 결과**이며, 현재 코드베이스로는 재현 불가합니다.

---

## 2. [작업 2] 5차 제출 실전 점수(840.76점)와 로컬 추정치 비교

### 5차 제출 학습 파이프라인 (build_submission_v5.py Line 52)
```python
prep.fit(df_train, as_of_season=None, is_final=True)  # 전체 데이터 final fit
```
- **`is_final=True`**: 전체 학습 데이터 사용, as_of_season 제한 없음.

### 실전 vs 로컬 추정치 비교

| 지표 | 로컬 추정치 (strict=False) | 로컬 추정치 (strict=True) | **실전 점수 (DACON)** |
|:---:|:---:|:---:|:---:|
| Skill Score | `843.42점` | `850.09점` | **`840.76점`** |
| Raw Brier | `0.247554` | `0.247538` | - |
| 실전 차이 | `+2.66점` 과다추정 | `+9.33점` 과다추정 | - |

### 판단
- `strict_as_of=False` 모드(843.42점)가 실전 점수(840.76점)에 더 가까움(차이 +2.66점).
- `strict_as_of=True` 모드(850.09점)는 실전 대비 +9.33점 과다추정으로 더 낙관적임.
- **단, 이것이 strict=False가 더 "올바른" 방법이라는 뜻은 아님** — 실전 점수에 가깝더라도 Fold 0/1의 미래 누수가 있는 방법이기 때문.

---

## 3. [작업 3] 정직한 재출발점 확정

### 111번 보고서 "완벽 재현" 공식 철회 재확인
- 111번 보고서에서 "850.09점 100% 완벽 재현"이라 표기한 것은 잘못된 기술이었음. 공식 철회.

### 새 공식 SSOT 확정
현재 코드베이스에서 **100% 재현 가능한 두 기준 수치**:

| 구분 | Skill Score | Raw Brier | 방법론 | 실전 근접도 |
|:---:|:---:|:---:|:---:|:---:|
| **strict=True (공식 SSOT)** | **`850.09점`** | **`0.247538`** | 누수 0%, 방법론적으로 올바름 | 실전+9.33점 |
| strict=False (참고용) | `843.42점` | `0.247554` | Fold 0/1 미래 누수 존재 | 실전+2.66점 |

**공식 SSOT: `850.09점` / Raw Brier `0.247538` (`strict_as_of=True`)**

### 코드 컨벤션 강제 지침
```python
# 앞으로 모든 실험은 반드시 이 방식으로만 검증
from core.eval_utils import run_standard_sota_evaluation
results = run_standard_sota_evaluation(df_train, strict_as_of=True)
```

### Fold별 공식 SSOT 기준 수치 (검산 가능한 전수 수치)
| Fold | 검증 시즌 | $r_k$ | Baseline Brier | Fold Raw Brier | Fold Skill Score |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022년 (as_of=2021) | `0.528920` | `0.249163` | **`0.244543`** | **`1854.48점`** |
| **Fold 2** | 2023년 (as_of=2022) | `0.499957` | `0.249999` | **`0.249737`** | **`105.12점`** |
| **Fold 3** | 2024년 (as_of=2023) | `0.486105` | `0.249807` | **`0.248331`** | **`590.67점`** |

> 산술평균 검산: (1854.48 + 105.12 + 590.67) / 3 = **850.09점** ✅

---

## 4. 00_summary.md 정정 기록

다음의 정정 사실을 `outputs/00_summary.md`에 공식 기록:

> **[정정 공지 - 보고서 113, {NOW_STR}]**
> 보고서 68~112번 전반에 걸쳐 "로컬 SOTA: 859.63점 / Raw Brier 0.247513"으로 기재되었던 수치는,
> 보고서 112번 감사 결과 현재 코드베이스로 재현 불가함이 확인되었습니다.
> 공식 SSOT는 **850.09점 / Raw Brier 0.247538 (strict_as_of=True)**로 정정됩니다.
"""

with open(OUTPUTS_DIR / '113_859_origin_trace.md', 'w', encoding='utf-8') as f:
    f.write(doc_113)

# Update 00_summary.md if it exists
summary_path = OUTPUTS_DIR / '00_summary.md'
correction_notice = f"""
---

## ⚠️ [정정 공지 - 보고서 113, {NOW_STR}]

보고서 68~112번 전반에 걸쳐 "로컬 SOTA: **859.63점** / Raw Brier **0.247513**"으로 기재되었던 수치는,
보고서 112번 감사 결과 현재 코드베이스로 **재현 불가**함이 확인되었습니다.

**공식 정정 SSOT**: `850.09점` / Raw Brier `0.247538` (`strict_as_of=True`, `core/eval_utils.py`)

- Fold 1 (2022, as_of=2021): Raw Brier=`0.244543`, Skill=`1854.48점`
- Fold 2 (2023, as_of=2022): Raw Brier=`0.249737`, Skill=`105.12점`
- Fold 3 (2024, as_of=2023): Raw Brier=`0.248331`, Skill=`590.67점`
- 3-Fold 산술평균: **`850.09점`**

앞으로 모든 실험 비교는 이 수치를 기준으로만 수행합니다.
"""

if summary_path.exists():
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(correction_notice)
    print("00_summary.md updated with correction notice!")
else:
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# DACON Aimers 9기 프로젝트 요약\n{correction_notice}")
    print("00_summary.md created with correction notice!")

print("Report 113 written successfully!")
