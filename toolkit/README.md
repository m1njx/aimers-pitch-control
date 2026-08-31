# reusable_toolkit — 다음 대회에 바로 쓰는 도구

> `../code/` 의 스크립트들은 이 대회 경로·컬럼이 박혀 있어 **그대로는 못 쓴다**.
> 여기 3개는 **대회 무관하게 범용화**했고, 실제 제출물로 **동작 검증까지 마쳤다**.
> 의존성은 `numpy` / `pandas` 뿐이다.

| 도구 | 언제 쓰나 | 검증 |
| :--- | :--- | :--- |
| `check_submission.py` | **제출 직전 매번** | v37 제출물로 8/8 통과 확인 |
| `check_row_independence.py` | 행 독립 규정이 있는 대회 | v37 로 2.6e-09 통과 확인 |
| `blend_math.py` | 앙상블 설계·arm 채택 판단 | 3개 자기검증 통과 |

---

## 1. `check_submission.py` — 0점 방지

```bash
python3 check_submission.py mysub.zip --data-dir ./data
python3 check_submission.py mysub.zip --data-dir ./data \
        --require test.csv sample_submission.csv --entry script.py --timeout 600
```

8개를 자동 검사한다. 각 항목은 이 대회에서 **실제로 사고가 났던 것**이다.

| # | 검사 | 없어서 잃은 것 |
| ---: | :--- | :--- |
| 1 | zip 위생 (`__pycache__` / 학습데이터 동봉) | 용량·정보 유출 |
| 2 | 진입점이 최상위에 있는가 | 실행 실패 |
| 3 | 모든 `.py` 에 `ast.parse` | **SyntaxError 로 슬롯 소실** |
| 4 | **새로 푼** 폴더에서 필수 데이터만 두고 실행 | `ModuleNotFoundError` |
| 5 | 출력 행수·범위·NaN·id 중복 | 0점 |
| 6 | 출력 id 순서가 sample_submission 과 같은가 | **위치대입으로 100% 오정렬** |
| 7 | 두 번 실행해 바이트 동일 | 재현 불가 |
| 8 | 실행 시간 여유 | 타임아웃 |

종료 코드 0 = 통과. CI 에 걸어도 된다.

> **6번이 특히 중요하다.** 여러 모델을 서브프로세스로 돌려 합치는 구조에서 각 모델의 출력
> 순서가 다를 수 있다. `.reindex(sub[id])` 없이 `.values` 로 대입하면 전 행이 어긋난다.

## 2. `check_row_independence.py` — 규정 위반 검출

```bash
python3 check_row_independence.py mysub.zip --data-dir ./data \
        --probe ./data/train.csv --n 3000 --solo 8
```

FULL / SHUFFLE / SUBSET / **SOLO(1행씩)** 4변형을 돌려 행별로 비교한다.
**1행만 넣어도 3,000행 프레임과 같은 값**이면 프레임 의존이 없다.

- 부동소수점 잡음: **1e-9** 수준
- 진짜 누출: **1e-3** 안팎
- 판정 임계 `--atol` 기본 1e-6

정적 코드 리뷰보다 강하다 — 어떤 경로로 새든 값이 달라지기 때문이다.

> **dtype 함정 자동 진단**: SOLO 가 특정 행에서만 크래시하면 모델 결함이 아니라 CSV dtype
> 추론일 수 있다. 이 대회 실측 사례 — 값이 전부 숫자인 문자열 카테고리 `'123'` 이 1행
> 프레임에서 `int64` 로 파싱되어 인코더가 죽었다. 스크립트가 그 후보 컬럼을 미리 지목한다.

## 3. `blend_math.py` — 앙상블 설계

```python
from blend_math import gram, opt, contribution, rho_of, required_rho, ceiling

M, u = gram([pA, pB, pC], y), None
w, u = opt(M)                              # 최적 가중치
gain, w = contribution([pA, pB, pC], y)    # C 를 추가하면 얼마나 오르나
rho = rho_of(pC, [pA, pB], y)              # 이득을 결정하는 유일한 값
need = required_rho(target_gain=12, base_skill=u)
```

```bash
python3 blend_math.py --demo    # 자기검증 3종
```

### 왜 이게 시간을 아껴 주나

Brier/MSE 는 예측 벡터의 **정확한 2차형식**이라, `(s_i, d_ij, 라벨률)` **6개 수치만으로**
모든 블렌드 조합의 점수가 결정된다. **arm 을 실제로 섞어보지 않고 계산할 수 있다.**
(자기검증 1번이 이걸 오차 8e-12 로 확인한다. 실제 대회 arm 5종에서는 최대오차 0.02였다.)

### 이 도구가 막아 주는 착각 두 가지

**"다양성을 키우면 점수가 오른다"** — 아니다. 자기검증 2번:

```
    배율   corr(C,B)   기여(아핀)      w_C
     1.0       0.080     1609.45   -1.600
     2.0      -0.196     1609.45   -0.800
     3.0      -0.278     1609.45   -0.533
```
고유방향을 3배로 키워 corr 을 0.08 → −0.28 로 낮춰도 **기여는 완전히 불변**이고
가중치만 1/m 로 준다. `corr` 를 목표로 최적화하면 **노이즈 주입으로 "달성" 되면서 점수가 깎인다.**

**"단독 스킬을 올리면 된다"** — 아니다. 스킬을 올리면 예측이 기존 arm 합의로 끌려가
다양성이 그만큼 깎인다(이 대회 실측 상관 **−0.98**). 단독 최고 모델이 최적 가중치 **0.000** 을
받은 사례가 있다.

**옳은 목표는 `ρ = corr(현재 잔차, 새 arm 의 고유방향)` 하나다.**
`required_rho(목표이득, 현재점수)` 로 필요치를 먼저 계산하고, 후보의 `rho_of` 가 그에 못 미치면
**학습을 시작하지 마라.** (이 대회에서 물리 피처 arm 은 ρ 0.00% 였다 — GPU 를 쓰기 전에
알았어야 했다.)

### `ceiling()` — 라이브러리 천장

```python
for lam, fit_gain, eval_gain in ceiling(preds_2022, y22, preds_2024, y24):
    print(lam, fit_gain, eval_gain)
```
**가중치를 fit 폴드에서만 적합**하고 eval 폴드에서 실현치를 본다.
한 폴드에서 적합·평가하면 반드시 과적합한다 — 이 대회 실측: `λ=1e2` 에서
**2022 +148.9 / 2024 −15.9**.

---

## 다음 대회 1일차 체크리스트

1. `check_row_independence.py` 를 **베이스라인이 돌자마자** 붙인다.
   나중에 붙이면 이미 오염된 것을 못 본다.
2. `check_submission.py` 를 제출 스크립트에 묶는다(통과해야 zip 이 나오게).
3. `blend_math.py` 로 **목표 점수 → 필요 ρ** 를 먼저 계산해 둔다.
   그 숫자가 있어야 "이 후보를 학습할 가치가 있나" 를 판단할 수 있다.
