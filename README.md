# aimers-pitch-control

**투구 제구 성공 확률 예측 — 모델링 파이프라인과 방법론**

<sub>2026 LG Aimers 9기 해커톤 · 본인 최고 Public LB **1,114.74** · 팀 최종 1,130.16</sub>

<details>
<summary><b>English summary</b></summary>

Pitch-control success probability prediction (binary classification, 1.47M rows × 48 features),
evaluated by Brier Skill Score. This repository contains **my part** of a 5-person team project:

- **`src/pipeline/`** — a 25-model GBDT + MLP ensemble (LB 1,032.14 standalone; my best full
  submission scored **1,114.74** and held the team record when submitted). The largest single
  gain in the whole competition came from **decomposing as-of cumulative features** (+146.8).
- **`toolkit/`** — three competition-agnostic tools, verified against real submissions:
  submission sanity checks, row-independence auditing, and closed-form blend math.
- **`playbook/`** — a catalog of **44 techniques** with runnable code and verdicts
  (24 adopted, 16 rejected, 4 shelved). Failed techniques are kept, with evidence.
- **`study_guide/`** — two study references: a **32-page** methodology guide
  (with a synergy matrix and seven combinations to avoid) and a **14-page** primer on
  deep learning, transformers, LLM agents, convex optimization and decision-focused learning.

Documentation is in Korean.
</details>

```bash
git clone https://github.com/m1njx/aimers-pitch-control.git
cd aimers-pitch-control

python3 toolkit/blend_math.py --demo      # 자기검증 3종
cd playbook && python3 run.py list        # 기법 44종 카탈로그
```

---

| | |
| :--- | :--- |
| 과제 | 투구별 제구 성공 확률 예측 (이진 분류, 147만 행 × 48피처) |
| 평가 | Brier Skill Score |
| **본인 최고 Public LB** | **1,114.74** (`v92` — 제출 시점 팀 최고 기록) |
| 담당 파이프라인(A arm) 단독 | 1,032.14 |
| 팀 최종 Public LB | 1,130.16 |
| 기간 | 2026-08-06 ~ 09-01 |

---

## 기술 스택

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=flat-square"/> <img src="https://img.shields.io/badge/XGBoost-006ACC?style=flat-square"/> <img src="https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black"/> <img src="https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
<br/>
<img src="https://img.shields.io/badge/Feature%20Engineering-4B8BBE?style=flat-square"/> <img src="https://img.shields.io/badge/Ensemble%20Learning-6E40C9?style=flat-square"/> <img src="https://img.shields.io/badge/Time%20Series%20CV-2E8B57?style=flat-square"/> <img src="https://img.shields.io/badge/Model%20Calibration-B7472A?style=flat-square"/> <img src="https://img.shields.io/badge/Data%20Leakage%20Audit-A0522D?style=flat-square"/>

| 구분 | 사용 | 어디에 |
| :--- | :--- | :--- |
| **모델** | LightGBM · XGBoost · CatBoost | GBDT 서브모델 (분류/회귀 두 목적함수) |
| | PyTorch | MLP 서브모델 5-seed 블렌딩 (**+36.2**) |
| | scikit-learn | 아핀 캘리브레이션 · 지표 · 분할 |
| **데이터** | Pandas · NumPy | 147만 행 × 48피처 전처리, as-of 분해 (**+146.8**) |
| | SciPy | logit/sigmoid 변환 · 최적화 · 통계 검정 |
| | joblib | 아티팩트 직렬화 (모델 25개 봉인) |
| **실행** | Google Colab (T4/A100) | GPU 학습 — 판정용·배포용 두 벌 규격 |
| **검증** | numpy + pandas **only** | `toolkit/` 3종은 의존성을 최소로 유지 |

> `toolkit/` 을 numpy·pandas만으로 묶은 것은 의도적입니다 — 다른 대회 저장소에
> 파일만 복사해 넣으면 바로 돌아가야 하기 때문입니다.

### 역량 배지가 가리키는 실제 코드

| 배지 | 이 저장소에서 | 근거 |
| :--- | :--- | ---: |
| **Feature Engineering** | `src/pipeline/agent2_asof_decomp2.py` — as-of 누적 피처 46개 분해 | **+146.8** |
| **Ensemble Learning** | `src/pipeline/ensemble_optimize.py` · `toolkit/blend_math.py` — 25모델 블렌드, 2차형식 닫힌 해 | **+36.2** |
| **Time Series CV** | `playbook/methods/validation.py` — 시즌 경계 3분할, 레짐 붕괴 구간 제외 | 폴드↔LB 오차 3.03 |
| **Model Calibration** | `playbook/methods/calibration.py` — 아핀 보정 (파라미터 2개) | **+12.9** |
| **Data Leakage Audit** | `toolkit/check_row_independence.py` — FULL/SHUFFLE/SUBSET/SOLO 행동 검증 | 누출 2.6e-09 |

---

## 담당 범위

5인 규모 팀 프로젝트에서 **예측 파이프라인 한 축(A arm)과 실험·검증 체계**를 맡았습니다.
이 저장소에는 **제가 작성·설계한 부분만** 담았습니다.
팀 공동 파트와 다른 담당자의 코드는 포함하지 않았습니다.

문서·학습자료에는 **팀 공동 파트에서 도입된 기법**도 다룹니다. 해당 위치에는
`[기여 구분]` 라벨을 붙여, 무엇이 팀 기법이고 무엇이 제가 수행한 검증·확장인지 구분했습니다.

### 수치로 본 기여

| 작업 | Public LB 변화 |
| :--- | ---: |
| **as-of 누적 피처 분해 (46개 신규 피처)** | **+146.8** |
| **MLP 5-seed 블렌딩** | **+36.2** (1,000점 최초 돌파) |
| 3-way 목적함수 앙상블 (분류/회귀/MLP) | +12.5 |
| 아핀 캘리브레이션 | +12.9 |
| 팀 기법 독립 검증 + 확장 축 18종 판정 | (기각 확정) |
| GPU 학습 실행 · 두 벌 규격 확립 | (판정 체계 확보) |
| A arm 단독 최고 | 1,032.14 |
| **전체 빌드 최고 (`v92`, 블렌드 가중치 재조정)** | **1,114.74** |

**as-of 분해 피처(+146.8)는 이 대회 전체에서 단일 변경 최대 이득**이었습니다.
상세: [docs/01_MY_CONTRIBUTION.md](docs/01_MY_CONTRIBUTION.md)

---

## 저장소 구성

```
├── src/
│   ├── pipeline/      담당 예측 파이프라인 (GBDT+MLP 25모델 앙상블)
│   │   ├── agent2_asof_decomp2.py     ★ as-of 누적 피처 분해 (+146.8)
│   │   ├── cfa_latent_features.py       확인적 요인분석 잠재변수
│   │   ├── game_theory_features.py      투수-타자 상호작용 피처
│   │   ├── trackman_features.py         투구 추적 데이터 집계
│   │   ├── train_mlp_only.py            MLP 서브모델 학습
│   │   ├── ensemble_optimize.py         블렌드 가중치 탐색
│   │   └── script.py                    추론 진입점
│   ├── analysis/      실험 분석 스크립트 (16종)
│   ├── gpu_colab/     Colab T4 학습 스크립트 (5종) — 판정용·배포용 두 벌 규격
│   └── experiments/   실험 코드 480종
│
├── toolkit/           ⭐ 대회 무관 재사용 도구 (동작 검증 완료)
│   ├── check_submission.py         제출물 0점 방지 8항목 자동 검사
│   ├── check_row_independence.py   행 독립성 규정 위반 검출
│   └── blend_math.py               앙상블 닫힌형 계산
│
├── playbook/          ⭐ 기법 카탈로그 44종 (채택 24 · 기각 16 · 보류 4)
│   ├── methods/                    기법 구현 (features · lookups · calibration ·
│   │                               ensemble · validation · rejected)
│   ├── config.py                   새 대회에 맞춰 이 파일만 고친다
│   └── run.py                      python3 run.py list
│
├── study_guide/       ⭐ 학습자료 PDF 2종
│   ├── 방법론_학습자료.pdf            32쪽 — 기초 개념 + 기법 44종 + 시너지 매트릭스
│   └── 딥러닝_LLM_최적화_정리.pdf     14쪽 — 이론과 대회 경험의 연결
└── docs/              방법론 · 교훈 · 검증 · GPU
```

---

## 핵심 산출물 셋

### 1. 검증 도구 (`toolkit/`)

대회 무관하게 쓸 수 있도록 일반화하고, 실제 제출물로 동작을 검증했습니다.

```bash
python3 toolkit/check_submission.py mysub.zip --data-dir ./data
python3 toolkit/check_row_independence.py mysub.zip --data-dir ./data --probe ./data/train.csv
python3 toolkit/blend_math.py --demo
```

**`check_submission.py`** — 8항목을 자동 검사합니다. 각 항목은 이 대회에서 실제로 사고가 났던 것입니다:
구문 오류(슬롯 소실) · 새로 푼 zip 실행(모듈 누락) · **행 정렬**(위치 대입 시 100% 오정렬) ·
결정성 · 의존성 · 실행 시간.

**`check_row_independence.py`** — "각 행을 독립적으로 예측하라"는 규정을 **행동으로** 검증합니다.
같은 행을 전체 / 셔플 / 부분집합 / **1행 단독** 프레임에 넣어 값이 달라지는지 봅니다.
정적 코드 리뷰보다 강합니다 — 어떤 경로로 새든 값이 달라지기 때문입니다.

**`blend_math.py`** — Brier가 예측 벡터의 정확한 2차형식이라는 성질을 이용해,
**모델을 실제로 섞지 않고** 최적 가중치·기여도·필요 조건을 계산합니다.
실제 모델 5종의 관측 기여도를 최대오차 0.02로 재현했습니다.

### 2. 기법 카탈로그 (`playbook/`)

시도한 기법 44종을 **실행 가능한 코드 + 판정 근거**로 정리했습니다.
**실패한 기법도 코드로 남겼습니다** — 지운 기법은 반드시 누군가 다시 제안하기 때문입니다.

```bash
cd playbook
python3 run.py list                      # 44종 (✅24 ❌16 ⏸4)
python3 run.py list --status REJECTED    # 실패한 것만
python3 run.py show <id>                 # 근거·주의사항·구현 위치
```

### 3. 학습자료 (`study_guide/`, PDF 2종)

**`방법론_학습자료.pdf` (32쪽)** — 각 기법을
**직관 → 형식 → 구현 → 장단점 → 🔗시너지 → ⚡안티시너지 → 실측** 형식으로 정리했습니다.
0부에 결정 트리·앙상블·부스팅·GBDT 기초를 두고, 9부에 **10×10 시너지 매트릭스**와
**금지 조합 7종**을 실었습니다.

**`딥러닝_LLM_최적화_정리.pdf` (14쪽)** — 트랜스포머·LLM·에이전트·볼록 최적화·
의사결정 중심 학습(DFL)·시계열. 마지막 장에서 **각 이론이 대회 경험을 어떻게 설명하는지**
연결했습니다 — 예를 들어 DFL의 *"예측 오차 최소화 ≠ 최적 의사결정"*은
*"단독 성능 최대화 ≠ 블렌드 기여 최대화"*와 같은 구조입니다.

---

## 대회에서 얻은 결론 셋

**1. 구조적 변경만 큰 이득을 낸다.**
새 피처 블록 · 독립 파이프라인 결합 · 엔티티 룩업만 +15 이상을 냈고,
캘리브레이션 상수나 블렌드 비율 같은 미세조정은 전부 ±1~13 안에 있었습니다.

**2. 자유도가 기법의 운명을 결정한다.**
정직한 홀드아웃 분할을 통과해도 자유도가 크면 실전에서 죽습니다.
파라미터 2개짜리 아핀 보정은 +12.9를 냈고,
파라미터 25,000개짜리 잔차 후처리 모델은 같은 검증을 통과하고도 실전에서 −4.1이었습니다.

**3. 정직한 null도 결과다.**
20개 이상의 방향을 근거와 함께 닫았고, 그것이 남은 시간을 지켰습니다.
`playbook`의 ❌16종이 그 기록입니다.

---

## 참고

- 대회 데이터는 주최측 약관에 따라 포함하지 않았습니다.
- 학습자료의 「실측」 수치는 **팀 프로젝트 전체의 측정 기록**이며,
  제 담당 범위는 [docs/01_MY_CONTRIBUTION.md](docs/01_MY_CONTRIBUTION.md)에 구분해 두었습니다.
- 파이프라인 실행: `pip install -r requirements.txt`
- **`toolkit/` 은 `numpy` + `pandas` 만으로 동작**합니다 — 다른 대회에 바로 복사해 쓸 수 있습니다.
- `src/experiments/` 의 스크립트는 원 프로젝트 경로(`~/LG_data`)를 참조합니다. 실행하려면 경로 조정이 필요합니다.
