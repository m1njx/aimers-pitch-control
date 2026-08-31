"""
diag_blend_gradient.py — 517 마스터 항등식을 "실제로 채점 가능한 형태"로 축약한다.

517 은 u(w) = Σ wᵢsᵢ + Σ wᵢwⱼDᵢⱼ 를 유도했고, D 가 라벨 없이 계산된다는 점에서
"다양성 d 를 키우면 +1e5·d² 를 번다"는 전략을 제시했다. 그 표는 s 와 D 를
독립 손잡이처럼 다뤘지만 **둘은 같은 Δ 의 함수라 독립이 아니다.**
여기서는 그 결합을 풀어 실제 판정식을 얻는다.

기호: 우리 arm A, 팀 arm B, 50:50 블렌드 q=(p_A+p_B)/2.
      r_A=p_A−y, r_B=p_B−y, r_q=q−y.  V=r(1−r).  K1 = 1e5/V.
      우리 arm 을 C = A + Δ 로 바꾼다.

  [G1] Δu = −K1·( E[Δ·r_q] + E[Δ²]/4 )
       (블렌드 예측이 Δ/2 만큼 움직인 것의 Brier 변화. 그게 전부다.)

  [G2] Δu = Δs/4 − (K1/2)·E[Δ·r_B]
       Δs 는 우리 arm 단독 skill 델타.

[G2] 가 핵심이다. **우리 arm 을 X 점 올려도 블렌드에는 X/4 밖에 안 들어온다.**
나머지는 전적으로 "Δ 가 팀 잔차와 얼마나 반대로 정렬됐는가"가 정한다.
Δ 가 r_B 와 직교하면 배수는 정확히 1/4, Δ 가 r_A 만큼 r_B 도 줄이면 1/2 이다.
따라서 **우리 arm 개선의 블렌드 전달 배수는 1/4 ~ 1/2** 이고,
LB 노이즈 바닥 ±12 를 넘으려면 arm 델타가 **+24 ~ +48 점**이어야 한다.

실행: venv311/bin/python3 harness/diag_blend_gradient.py
"""
import os
import sys
import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict, skill  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023, 2024]
SEEDS = [7, 123, 2025, 31415, 8675309]

# LB 실측 3점 (516/517)
LB_A, LB_B, LB_BLEND = 1032.137582, 1016.4138496773, 1054.8707763763


def load_fold(y):
    yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
    bag = []
    for s in SEEDS:
        f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
        if os.path.exists(f):
            bag.append(dict(np.load(f)))
    return yv, bag


def bagged(cfg, bag):
    return np.mean([predict(cfg, P) for P in bag], axis=0)


def sk(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


# ---------------------------------------------------------------------------
# pseudo-B: 팀 arm 은 CatBoost 계열이다. 우리 캐시에서 CatBoost 쪽으로 치우친
# 변형을 만들어 corr(A,B) 를 팀 실측 0.926 에 맞춘다. **절대 skill 은 무효이고
# (dacon-local-harness-invalid) 여기서는 "배수" 라는 구조량만 본다.**
# ---------------------------------------------------------------------------
PSEUDO_B = dict(PROD)
PSEUDO_B.update(w_lgb=0.05, w_cb=0.95, w_xgb=0.00,
                w_gbdt=1.00, w_mlp=0.00, w_mse=0.00, scale=1.00, shift=0.0)


def section(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


def main():
    section('0. 항등식 [G1]/[G2] 수치 검증 (임의 Δ, 4시즌)')
    print(f'  {"fold":>6} {"V":>7} {"Δ종류":>10} {"Δu 실측":>10} '
          f'{"[G1]":>10} {"[G2]":>10} {"오차":>10}')
    rng = np.random.default_rng(0)
    for y in FOLDS:
        yv, bag = load_fold(y)
        V = yv.mean() * (1 - yv.mean())
        K1 = 1e5 / V
        pA = bagged(PROD, bag)
        pB = bagged(PSEUDO_B, bag)
        rA, rB = pA - yv, pB - yv
        rq = 0.5 * (pA + pB) - yv
        u0 = sk(0.5 * (pA + pB), yv, V)
        sA = sk(pA, yv, V)
        deltas = {
            'noise': rng.normal(0, 0.02, size=yv.size),
            'shift': np.full(yv.size, 0.004),
            'sharp': 0.08 * (pA - pA.mean()),
            'toward_B': 0.30 * (pB - pA),
        }
        for nm, D in deltas.items():
            pC = np.clip(pA + D, 1e-6, 1 - 1e-6)
            D = pC - pA  # 클립 후 실제 Δ
            u1 = sk(0.5 * (pC + pB), yv, V)
            du = u1 - u0
            g1 = -K1 * ((D * rq).mean() + (D * D).mean() / 4)
            ds = sk(pC, yv, V) - sA
            g2 = ds / 4 - (K1 / 2) * (D * rB).mean()
            print(f'  {y:>6} {V:7.4f} {nm:>10} {du:10.3f} {g1:10.3f} '
                  f'{g2:10.3f} {max(abs(du-g1), abs(du-g2)):10.2e}')

    section('1. 전달 배수 — 우리 arm 개선 X 점이 블렌드에 얼마나 들어오는가')
    print('  Δu = Δs/4 − (K1/2)·E[Δ·r_B]   →  배수 = Δu/Δs')
    print('  Δ⊥r_B 면 정확히 0.25, Δ 가 r_B 도 r_A 만큼 줄이면 0.50.\n')
    print(f'  {"fold":>6} {"corr(A,B)":>10} {"후보":>12} {"Δs(arm)":>10} '
          f'{"Δu(blend)":>11} {"배수":>7}')
    mult = []
    for y in FOLDS:
        yv, bag = load_fold(y)
        V = yv.mean() * (1 - yv.mean())
        pA = bagged(PROD, bag)
        pB = bagged(PSEUDO_B, bag)
        rho = np.corrcoef(pA, pB)[0, 1]
        u0 = sk(0.5 * (pA + pB), yv, V)
        sA = sk(pA, yv, V)
        # 실제로 우리가 만들어 본 "arm 변형" 들: 성분 가중치 섭동 = 진짜 개입의 대리
        cands = {}
        for nm, kw in [('mlp+0.05', dict(w_mlp=0.45, w_gbdt=0.35)),
                       ('mlp-0.05', dict(w_mlp=0.35, w_gbdt=0.45)),
                       ('mse+0.05', dict(w_mse=0.25, w_gbdt=0.35)),
                       ('scale1.15', dict(scale=1.15)),
                       ('scale1.05', dict(scale=1.05))]:
            c = dict(PROD); c.update(kw)
            cands[nm] = bagged(c, bag)
        for nm, pC in cands.items():
            ds = sk(pC, yv, V) - sA
            du = sk(0.5 * (pC + pB), yv, V) - u0
            if abs(ds) < 0.5:
                continue
            mult.append(du / ds)
            print(f'  {y:>6} {rho:10.4f} {nm:>12} {ds:+10.2f} {du:+11.2f} '
                  f'{du/ds:7.3f}')
    if mult:
        m = np.array(mult)
        print(f'\n  배수 분포: 중앙 {np.median(m):.3f}  '
              f'[{m.min():.3f}, {m.max():.3f}]  n={m.size}')

    section('2. 채택 하한 재계산 (프로토콜 5번 갱신)')
    for mu, lab in [(0.25, 'Δ⊥r_B (보수)'), (0.375, '중간'), (0.50, '상한')]:
        print(f'  배수 {mu:.2f} ({lab:>12}) → 블렌드 +12 를 넘으려면 '
              f'arm 델타 ≥ {12/mu:+6.1f}')
    print('\n  성분 개입이면 여기에 유효가중치를 한 번 더 나눈다:')
    EW = {'mlp': 0.50, 'cb_bin': 0.29, 'lgb_mse': 0.25,
          'lgb_bin': 0.08, 'xgb_bin': 0.03}
    for k, w in EW.items():
        print(f'    {k:>8} (w={w:.2f}) → 단독 성분 델타 ≥ '
              f'{12/0.25/w:+8.1f} ~ {12/0.50/w:+7.1f}')

    section('3. 517 §2 "다양성 표" 의 재해석')
    print('  517 은 Δs 와 ΔD 를 독립 손잡이로 놓고 d=0.05 → +250 을 제시했다.')
    print('  둘은 같은 Δ 의 함수다. Δ 가 순수 독립 노이즈면:')
    print(f'  {"d":>8} {"Δs":>10} {"ΔD":>10} {"Δu":>10}')
    for d in [0.005, 0.010, 0.0175, 0.025, 0.035, 0.050]:
        ds = -4e5 * d * d           # 독립 노이즈: Δs = −K1·E[Δ²],  K1=4e5
        dD = 2e5 * d * d            # ΔD = K·d²
        print(f'  {d:8.4f} {ds:10.1f} {dD:10.1f} {ds/2 + dD/2:10.1f}')
    print('  → 노이즈로 산 다양성의 블렌드 이득은 항상 −1e5·d² < 0. 517 표의')
    print('    "+250" 은 skill 을 공짜로 유지한다는 가정이 있어야만 성립한다.')
    print('    즉 그 이득은 "다양성" 이 아니라 "Δ 가 r_B 와 음의 정렬" 에서 온다.')

    section('4. λ-샤프닝 상한 (라벨 없이 D 만 키우는 시도) — 닫힌형')
    M = (LB_BLEND - 0.5 * (LB_A + LB_B)) * 2 * (2 * 0.25) / 1e5
    print(f'  LB 가 못박은 M = E[(p_A−p_B)²] = {M:.4e}, RMS = {M**0.5:.5f}')
    print('  p_C = p̄ + λ(p_A−p̄) 로 D 를 키울 때 (A 캘리브레이션 가정):')
    print('    Δu = −1e5·t²·Var_A + 2e5·t·Q,  t=λ−1,  Q = Var_A − Cov_AB ≈ M/2')
    print('    t* = Q/Var_A,  Δu_max = 1e5·Q²/Var_A')
    for sd in [0.045, 0.052, 0.060]:
        VarA = sd * sd
        Q = M / 2
        print(f'    sd_A={sd:.3f} → t*={Q/VarA:+.4f} (λ*={1+Q/VarA:.4f}), '
              f'Δu_max = {1e5*Q*Q/VarA:+.2f} 점')
    print('  → 1점 미만. **스케일/시프트 재조정으로 다양성을 사는 축은 닫힘.**')
    print('  로컬 실측 최적 t (pseudo-B 기준):')
    for y in FOLDS:
        yv, bag = load_fold(y)
        V = yv.mean() * (1 - yv.mean())
        pA = bagged(PROD, bag); pB = bagged(PSEUDO_B, bag)
        u0 = sk(0.5 * (pA + pB), yv, V)
        best = max(((sk(0.5 * (np.clip(pA + t * (pA - pA.mean()), 1e-6, 1-1e-6)
                               + pB), yv, V) - u0), t)
                   for t in np.linspace(-0.3, 0.3, 121))
        print(f'    {y}: t*={best[1]:+.3f}  Δu={best[0]:+.2f}  '
              f'sd_A={pA.std():.4f}')

    section('5. 결론')
    print('  · 목적함수는 결국 [G1] 하나다: 블렌드 잔차 r_q 와 음으로 정렬된 Δ 만 번다.')
    print('  · 우리 arm 단독 개선의 전달 배수는 1/4~1/2. 지난 이틀의 "+12 기준"은')
    print('    블렌드 기준으로는 +3~6 이었다 — 노이즈 바닥 아래다.')
    print('  · 라벨 없이 D 만 키우는 경로(노이즈/샤프닝)는 항등식상 전부 음수 또는 <1점.')
    print('  · 남은 유일한 자유도는 E[Δ·r_B] 이고, 그건 B 의 예측 없이는 못 잰다.')


if __name__ == '__main__':
    main()
