import sys
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

doc_112 = """# 112. eval_utils.py 로직 라인별 감사 및 SSOT 불일치 최종 규명 보고서

- **작성 일시**: """ + NOW_STR + """
- **우선순위**: 🚨 최긴급 — 111번 보고서의 "완벽 재현" 주장 공식 철회

---

## 1. [작업 1] strict_as_of 로직 라인별 코드 감사

### `core/eval_utils.py` Line 68 (핵심 분기):

```python
as_of = fold.fold_max_season if strict_as_of else 2023
prep.fit(df_tr_f, as_of_season=as_of, is_final=False)   # Line 71
```

### Fold별 실제 as_of_season 값 (실행 결과 확인):

| Fold | val_season | fold_max_season | strict_as_of=True | strict_as_of=False |
|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022년 | **2021** | `as_of_season=2021` | `as_of_season=2023` |
| **Fold 2** | 2023년 | **2022** | `as_of_season=2022` | `as_of_season=2023` |
| **Fold 3** | 2024년 | **2023** | `as_of_season=2023` | `as_of_season=2023` |

- **108번에서 발견한 "as_of_season=2023 하드코딩" 버그 패턴**: `strict_as_of=False`일 때 완전히 동일한 패턴이 남아있음 ✅ 정확히 특정됨.
- `strict_as_of=True` 로직 자체는 올바르게 구현되어 있음 (Fold별 fold_max_season을 정확히 전달).

---

## 2. [작업 2] 두 모드 실제 측정값 vs 110번 SSOT 주장값 비교

### ❌ strict_as_of=False (as_of_season=2023 고정)
| Fold | Raw Brier | Skill Score |
|:---:|:---:|:---:|
| Fold 1 (2022) | `0.244555` | `1849.52점` |
| Fold 2 (2023) | `0.249775` | `90.07점` |
| Fold 3 (2024) | `0.248331` | `590.67점` |
| **Mean** | `0.247554` | **`843.42점`** |

### ❌ strict_as_of=True (as_of_season=fold_max_season)
| Fold | Raw Brier | Skill Score |
|:---:|:---:|:---:|
| Fold 1 (2022) | `0.244543` | `1854.48점` |
| Fold 2 (2023) | `0.249737` | `105.12점` |
| Fold 3 (2024) | `0.248331` | `590.67점` |
| **Mean** | `0.247538` | **`850.09점`** |

### 🎯 110번 SSOT 주장값 (재현 불가 확인됨)
| Fold | Raw Brier | Skill Score |
|:---:|:---:|:---:|
| Fold 1 (2022) | `0.244545` | `1853.64점` |
| Fold 2 (2023) | `0.249733` | `106.82점` |
| Fold 3 (2024) | `0.248261` | `619.79점` |
| **Mean** | `0.247513` | **`859.63점`** |

### 결론: **두 모드 모두 859.63점을 재현하지 못함**
- Fold 3의 Raw Brier (`0.248331` vs `0.248261`)가 양쪽 모두 다름 → 859.63점은 현재 코드베이스로 재현 불가
- 110번 보고서에 기재된 수치는 실제 스크립트 출력이 아닌 추정치였음을 확정

---

## 3. [작업 3] 정직한 최종 확정

### 111번 보고서 "완벽 재현" 주장 공식 철회
- 111번 보고서는 `strict_as_of=True` 모드에서 `850.09점`이 나왔음에도 "859.63점 100% 완벽 재현"으로 잘못 표기하였음. **공식 철회**.

### 현재 시점에서 재현 가능한 실제 SOTA 수치 확정

**현재 코드베이스로 100% 재현 가능한 수치는 두 가지:**

| 모드 | as_of_season | Raw Brier | Skill Score | 특성 |
|:---:|:---:|:---:|:---:|:---:|
| `strict_as_of=False` | 항상 2023 | `0.247554` | `843.42점` | Fold 0,1에 미래 트랙맨 누수 |
| `strict_as_of=True` | fold_max_season | `0.247538` | `850.09점` | 누수 0% 엄격 파이프라인 |

**공식 SSOT 확정**: `strict_as_of=True` 모드의 **`850.09점` / Raw Brier `0.247538`**
- 누수가 완전히 차단된 올바른 파이프라인이며, 두 수치 중 방법론적으로 더 신뢰할 수 있음
- 859.63점은 현재 코드로 재현 불가 → 더 이상 기준으로 사용 불가

### Exp 103, 104 기각 결론 유지
- `850.09점` 기준으로도 Exp 103(`850.36점`)은 baseline과 거의 동등, Exp 104(`832.31점`)는 열등
- **기각 결론 유지**
"""

with open(OUTPUTS_DIR / '112_eval_utils_bugfix.md', 'w', encoding='utf-8') as f:
    f.write(doc_112)

# Update core/eval_utils.py docstring to reflect correct SSOT
print("Report 112 written successfully!")
print("\nKEY FINDINGS:")
print("  strict_as_of=False: 843.42점 (as_of=2023 for all folds, Fold 0/1 have leakage)")
print("  strict_as_of=True:  850.09점 (as_of=fold_max_season, no leakage)")
print("  110번 SSOT claim:   859.63점 → CANNOT be reproduced with current codebase")
print("\nNew official SSOT: 850.09점 / 0.247538 (strict_as_of=True)")
