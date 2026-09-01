# 07 — GPU·클라우드 연산

> 이 대회에서 GPU 를 **무엇에 썼고, 얼마를 벌었고, 무엇이 낭비였는지**.
> 그리고 다음에 쓸 때의 실행 절차(SOP).

---

## 0. 결론 먼저

| 용도 | 결과 |
| :--- | :--- |
| **CatBoost 20멤버 GPU 학습** (B arm) | ✅ 채택 — 블렌드의 한 축이 됐다 |
| **CatBoost GPU 게이트 후보** (`i2x_l2384_gpu` 등) | ✅ 일부 채택 |
| **트랙맨 물리 arm C 학습** (Colab T4, 다수 변형) | ❌ **전량 낭비** — ρ 0.00% |
| **super 번들 학습** (Colab) | ❌ 낭비 — 정직 재학습 시 AUC 0.4985 |

**핵심 교훈**: 물리 피처 arm 에 쓴 GPU 시간은 **전부 낭비였고, 그것을 미리 알 수 있었다.**
`val.rho_screen`(상한 스크린)을 **먼저 돌렸으면** ρ 0.00% 가 몇 분 만에 나왔다.

> **GPU 를 켜기 전에 상한 스크린을 돌려라.** 이게 이 문서에서 가장 중요한 한 줄이다.

---

## 1. 🚨 GPU ≠ CPU — 판정과 배포는 같은 장치로

동일 하이퍼파라미터·동일 데이터인데 **다른 모델이 나온다**(실측):

| | 값 |
| :--- | ---: |
| CatBoost GPU vs CPU AUC 차 | **+0.02** |
| 두 예측의 상관 | **0.939** |

<b>즉 CPU 로 판정하고 GPU 로 배포하면 "판정한 것과 다른 물건"을 내는 것이다.</b>
반대도 마찬가지다. **후보 판정과 최종 배포를 같은 장치로 통일**하라.

원인: GPU 구현이 히스토그램 분할·부동소수점 누적 순서가 다르고,
일부 파라미터(`cat_features` 처리 등)의 동작이 CPU 와 다르다.

---

## 2. RunPod SOP (팀 확정, pod 6개 운영 경험)

원문: `HANDOFF_v29.md` §4

### 2.1 반드시 지킬 두 가지

1. **Secure Cloud 만 쓴다.**
   Community 는 4회 중 **3회 GPU 불량** — `nvidia-smi` 는 보이는데
   `torch.cuda.is_available()` 이 False. 우회 3종 전부 실패.
   Secure 는 4/4 정상.
2. **stop 하지 말고 terminate 한다.**
   stop 은 GPU 슬롯 반납이라 재시작이 보장되지 않는다(Secure 포함, 2회 확인).
   **산출물은 생성 즉시 scp 로 회수**한다.

### 2.2 확정 구성

| 항목 | 값 |
| :--- | :--- |
| 카드 | **A40 48GB $0.44/h** (CatBoost 1.2.10 호환 확실). L4 / 4090 Secure 도 가능 |
| 이미지 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| 디스크 | 40/40GB |
| 포트 | 22/tcp |
| 새 pod 스테이징 시간 | **약 10분** |

### 2.3 스테이징 절차

```bash
K="-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
H=root@<ip>; P=<port>

# ① GPU 실물 확인 — False 면 즉시 terminate 하고 재생성
ssh $K -p $P $H 'python3 -c "import torch;print(torch.cuda.is_available())"'

# ② 데이터 전송 (md5 로 무결성 확인)
ssh $K -p $P $H 'mkdir -p /workspace/repo/data'
scp $K -P $P data/*.csv $H:/workspace/repo/data/

# ③ 코드 전송 (로그·캐시 제외)
tar czf /tmp/code.tgz --exclude='*.log' --exclude='__pycache__' harness experiments submit_v9/model
scp $K -P $P /tmp/code.tgz $H:/workspace/
ssh $K -p $P $H 'cd /workspace/repo && tar xzf ../code.tgz --no-same-owner'

# ④ 의존성 — 채점 이력이 있는 목록 그대로
ssh $K -p $P $H 'python3 -m pip install --break-system-packages \
  catboost==1.2.10 scikit-learn==1.8.0 pandas==2.3.3 numpy==2.4.4 joblib==1.5.3 lightgbm xgboost'

# ⑤ 피처 캐시 생성 후 학습
ssh $K -p $P $H 'cd /workspace/repo && python3 experiments/v11_cli/build_cache.py'
```

---

## 3. Colab 워크플로

> **실행 주체**: 아래 Colab 학습은 **내가 직접 실행**했다.
> 스크립트는 `code/gpu_colab/`, 회수한 산출물은 `~/LG_data/my_gpu_runs/` 에 있다.
> 클라우드(RunPod) 운영은 팀 공동 파트다.


내가 쓴 스크립트: `code/gpu_colab/` (5종)

| 파일 | 용도 |
| :--- | :--- |
| `colab_newarm_template.py` | ★ **새 arm 학습 표준 템플릿** — 두 벌 규격을 강제한다 |
| `colab_armc_fold2023.py` | 폴드 검증용 학습 (`train<2024` → 2024 예측) |
| `colab_armc_full.py` | 배포용 학습 (전 시즌) |
| `colab_arm_c_runner.py` | 러너 |
| `colab_super_b_runner.py` | 대체 구성 실험 |

### 3.1 두 벌 규격 — 이걸 어기면 측정이 불가능해진다

```
(a) 판정용 : train < Y  로 학습 → Y 예측 .npy   ← 이게 없으면 가치를 못 잰다
(b) 배포용 : 전 시즌 학습 → 제출물에 탑재
```

**실제로 겪은 일**: 배포용만 받았더니 홀드아웃 연도가 in-sample 이라
단독 스킬이 **1982**(정직 측정 730.9) 로 부풀었고, 측정 자체가 막혔다.

### 3.2 Colab 실무

- 데이터 업로드가 병목이다. **필요한 컬럼만 슬림 CSV** 로 만들어 올린다
  (이 대회: 368MB → **33.5MB**, 19컬럼만)
- 세션이 끊기면 산출물이 사라진다. **학습 직후 즉시 다운로드**
- T4 로 CatBoost 20시드가 실행 가능한 규모였다

---

## 4. 로컬 GPU/멀티스레드 함정

| 증상 | 원인 / 대처 |
| :--- | :--- |
| 프로세스가 **CPU 0.0%** 로 15시간 정지 | macOS OpenMP 데드락. 스레드 수를 1로 강제 |
| 두 번째 학습부터 죽음 | 한 프로세스에서 CatBoost 반복 학습 불가(4회 재현) → **학습 1회 = 프로세스 1개** |
| torch + faiss 동시 로드 시 세그폴트 | import 순서 조정 또는 프로세스 분리 |

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
```

> **진단법**: `ps -o %cpu` 가 0.0% 면 느린 게 아니라 **죽은 것**이다.

---

## 5. GPU 를 쓰기 전 체크리스트

1. **상한 스크린을 먼저 돌렸는가?** (`playbook` → `val.rho_screen`)
   → ρ 가 필요치에 못 미치면 **학습하지 마라**. 이 대회에서 이걸 안 해서 T4 시간을 태웠다.
2. **판정용·배포용 두 벌을 뽑을 계획인가?**
3. **판정과 배포를 같은 장치로 할 것인가?** (GPU≠CPU)
4. **산출물 회수 경로가 준비됐는가?** (Colab 세션 종료·pod terminate 대비)
5. **의존성 버전이 채점 서버와 호환되는가?**
   → 이 대회에서 의존성 한 줄 때문에 **채점 서버 설치 오류로 제출이 통째로 실패**한 적이 있다

---

## 6. 이 대회 GPU 투자 결산

| 대상 | 투입 | 회수 |
| :--- | :--- | :--- |
| CatBoost 20멤버 (B arm) | RunPod A40, pod 6개 | ✅ 블렌드 한 축 |
| 물리 arm C 변형 11종 | Colab T4 다수 세션 | ❌ ρ 0.00%, 전량 폐기 |
| super 번들 | Colab | ❌ AUC 0.4985 |

**성공 1건 / 실패 2건.** 실패 2건은 **상한 스크린 몇 분**이면 사전에 걸렀다.

관련: `01_METHODS.md` §4.4 · `02_LESSONS.md` B절 · `playbook/methods/validation.py::rho_screen`
