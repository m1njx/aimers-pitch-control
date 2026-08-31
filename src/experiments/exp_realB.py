"""exp_realB.py — 팀 arm B 의 폴드 예측을 우리가 직접 생성해 블렌드 목적함수를 푼다.

배경
----
`outputs/522` 의 채점식은 팀 잔차 `r_B` 를 요구한다:

    [G2]  Δu = Δs/4 − (1e5/2V)·E[Δ·r_B]
    [G3]  Δu = [Δs/2 + K1·E[Δ²]/4] + (K1/2)·E[Δ·Δ_AB]

`E[Δ·r_B]` 는 B 예측 없이는 원리상 못 잰다. 팀에게 요청하는 대신(`outputs/523`,
전달 불가로 폐기) **그쪽 gate_run.py 를 우리 트리에서 직접 돌려 B 를 재현한다.**

  · `~/LG_data/teamB/` = 팀 저장소 의 experiments/ + harness/ 격리 사본.
    그쪽 repo 는 건드리지 않는다. data 는 우리 open/data 로 심볼릭 링크.
  · `build_I_k.py` 가 `hist = train_labeled[train_labeled.season < s]` 로
    **폴드 안전**하다 — `outputs/517` §3 의 isf_n<0 붕괴는 학습 경로가 아니라
    *제출용* 앵커를 2024 행에 먹인 데서 나온 것이었다.
  · 설정 l2384 = lr .015 / depth 8 / l2_leaf_reg 384 / Bernoulli .9 / border 128.
    NOTES.md 기준 B arm 의 85% 가 이 구성이다.

검수 (반드시 통과해야 함)
------------------------
1. 길이 == len(harness/cache/y_{Y}.npy)
2. r_val == y.mean()
3. corr(p_A, p_B) 가 LB 실측 0.926 대역인가
4. **D_AB 가 LB 가 못박은 61.19 를 재현하는가** — 재현하면 로컬을 탐색
   목적함수로 쓸 자격이 생긴다. 로컬 절대 skill 은 LB 와 역상관이지만
   (`dacon-local-harness-invalid`) D 는 라벨을 안 쓰는 순수 기하량이다.
   ⚠️ 가정하지 말고 측정해서 확인할 것.

실행:
  cd ~/LG_data/teamB && ~/LG_data/venv311/bin/python3 experiments/v11_cli/gate_run.py \
      --tag l2384 --folds 2021,2022,2023,2024 --seeds 7,123,2025 --threads 6 \
      --cfg '{"learning_rate":0.015,"depth":8,"l2_leaf_reg":384,
              "bootstrap_type":"Bernoulli","subsample":0.9,"border_count":128,
              "rsm":1,"leaf_estimation_iterations":1}'
  venv311/bin/python3 harness/exp_realB.py
"""
import glob
import os
import sys

import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
PREDS = os.path.join(LG, 'teamB/out/preds')
FOLDS = [2021, 2022, 2023, 2024]
SEEDS = [7, 123, 2025, 31415, 8675309]
LB_D_AB = 61.190          # LB 3점이 못박은 값 (outputs/516/517)
LB_CORR = 0.926           # 팀 기록


def our_arm(y):
    bag = [dict(np.load(os.path.join(CACHE, f'pred_{y}_{s}.npz')))
           for s in SEEDS if os.path.exists(os.path.join(CACHE, f'pred_{y}_{s}.npz'))]
    return np.mean([predict(PROD, P) for P in bag], axis=0)


def their_arm(y):
    fs = sorted(glob.glob(os.path.join(PREDS, f'l2384_f{y}_s*.npy')))
    if not fs:
        return None, 0
    return np.mean([np.load(f) for f in fs], axis=0), len(fs)


def skill(p, yv, V):
    return 1e5 * (1.0 - ((p - yv) ** 2).mean() / V)


def main():
    print('=' * 78)
    print('검수 — 우리가 재현한 B 가 LB 의 B 와 같은 물건인가')
    print('=' * 78)
    print(f'  {"fold":>6} {"n_seed":>7} {"len ok":>7} {"corr(A,B)":>10} '
          f'{"s_A":>9} {"s_B":>9} {"D_AB":>8} {"u(50:50)":>10}')
    ok_folds = []
    for y in FOLDS:
        yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
        pB, nseed = their_arm(y)
        if pB is None:
            print(f'  {y:>6} {"—":>7}   (예측 없음 — gate_run 아직 미완)')
            continue
        if len(pB) != len(yv):
            print(f'  {y:>6} {nseed:>7} {"✗":>7}  길이 {len(pB)} != {len(yv)}')
            continue
        V = yv.mean() * (1 - yv.mean())
        pA = our_arm(y)
        rho = np.corrcoef(pA, pB)[0, 1]
        sA, sB = skill(pA, yv, V), skill(pB, yv, V)
        u = skill(0.5 * (pA + pB), yv, V)
        D = 2 * (u - 0.5 * (sA + sB))
        print(f'  {y:>6} {nseed:>7} {"✓":>7} {rho:10.4f} {sA:9.1f} {sB:9.1f} '
              f'{D:8.2f} {u:10.1f}')
        ok_folds.append((y, yv, V, pA, pB))

    if not ok_folds:
        print('\n예측이 아직 없다. gate_run 이 끝난 뒤 다시 실행할 것.')
        return

    print(f'\n  LB 정합값: corr {LB_CORR}, D_AB {LB_D_AB}')
    Ds = [2 * (skill(0.5 * (a + b), yv, V) - 0.5 * (skill(a, yv, V) + skill(b, yv, V)))
          for _, yv, V, a, b in ok_folds]
    print(f'  로컬 D_AB 평균 {np.mean(Ds):.2f} (범위 {min(Ds):.2f}~{max(Ds):.2f})')
    ratio = np.mean(Ds) / LB_D_AB
    print(f'  LB 대비 배율 {ratio:.2f}x  → '
          f'{"재현 (탐색 목적함수로 사용 가능)" if 0.7 < ratio < 1.4 else "미재현 — 스케일 보정 필요"}')

    print('\n' + '=' * 78)
    print('[G2] 진짜 r_B 로 파트너 인지 재가중 재판정 (522 §4)')
    print('=' * 78)
    V_PROD = np.array([0.40 * 0.20, 0.40 * 0.72, 0.40 * 0.08, 0.20, 0.40])
    COMPS = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
    S_PROD = np.array([-0.007, -0.008, -0.006, 0.0, 0.0])
    EPS = 1e-6

    def arm(v, y):
        out = []
        for s in SEEDS:
            f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
            if not os.path.exists(f):
                continue
            P = dict(np.load(f))
            raw = sum(w * np.clip(P[c] + sh, EPS, 1 - EPS)
                      for w, sh, c in zip(v, S_PROD, COMPS))
            out.append(np.clip(0.5 + 1.10 * (raw - 0.5) - 0.0045192086, EPS, 1 - EPS))
        return np.mean(out, axis=0)

    print(f'  {"fold":>6} {"t(mlp)":>7} {"Δs(arm)":>9} {"Δu(blend)":>10} {"배수":>7}')
    for t in [0.45, 0.55, 0.70]:
        cells = []
        for y, yv, V, pA, pB in ok_folds:
            v = V_PROD.copy(); tot = v.sum()
            v[4] = t * tot
            v[:4] = V_PROD[:4] / V_PROD[:4].sum() * (1 - t) * tot
            pC = arm(v, y)
            pC = np.clip(pA.mean() + (pC - pC.mean()) * (pA.std() / pC.std()),
                         EPS, 1 - EPS)          # sd 고정 (샤프닝 대조군)
            ds = skill(pC, yv, V) - skill(pA, yv, V)
            du = skill(0.5 * (pC + pB), yv, V) - skill(0.5 * (pA + pB), yv, V)
            cells.append(du)
            print(f'  {y:>6} {t:7.2f} {ds:+9.2f} {du:+10.2f} '
                  f'{du/ds if abs(ds) > 0.3 else float("nan"):7.2f}')
        c = np.array(cells)
        print(f'  {"":>6} {t:7.2f} {"평균":>9} {c.mean():+10.2f}  '
              f'전폴드양수={bool((c > 0).all())}\n')

    print('판정: PROD(t=0.40) 대비 Δu 가 전 폴드 양수이고 평균이 LB 노이즈 바닥 ±12 를')
    print('      넘으면 채택 후보. 이제 프록시가 아니라 실제 B 로 잰 값이다.')


if __name__ == '__main__':
    main()
