#!/usr/bin/env python3
"""exp_nn_ceiling.py — "새 tabular NN arm" 축의 상한을 재학습 0 으로 계산한다.

무엇을 묻는가
-------------
제출물은 우리 arm A 단독이 아니라 블렌드 B축 와의 50:50 블렌드다. 517/diag_blend_gradient
의 마스터 항등식:

    [G2]  Δu = Δs/4 − (K1/2)·E[Δ·r_B]           K1 = 1e5/V,  r_B = p_B − y

r_B 를 우리는 못 본다. 그러나 r_B = r_A − Δ_AB (Δ_AB = p_A − p_B) 로 쪼개면

    [G3]  Δu = ( Δs/2 + K1·E[Δ²]/4 )  +  (K1/2)·E[Δ·Δ_AB]
              └──── 캐시만으로 정확히 계산 가능 ────┘   └── B 없이는 못 재는 항 ──┘

첫 괄호는 "Δ 가 A–B 불일치 방향과 직교" 일 때의 값이고, 둘째 항은 Cauchy–Schwarz 로
|·| ≤ (K1/2)·d·m, d = √E[Δ²], m = √E[Δ_AB²] = 0.01749 (LB 3점이 못박음, 517 §5).

즉 **중심값은 캐시만으로 정확히 계산되고, 불확실성 폭도 LB 가 못박은 m 으로 정확히
계산된다.** 새 NN 을 한 개도 학습하지 않고 이 축의 상한이 나온다.

동치 확인: 등품질 교체(Δs=0) + 직교면 Δu = K1·d²/4 = 1e5·d² (V=0.25). 517 §2 의
"직교 다양성 이득 = 1e5·d²" 와 같은 식이다. 순수 독립 노이즈면 Δs = −K1 d² 라
Δu = −1e5·d² < 0 — diag_blend_gradient §3 과도 일치한다.

착수 전 확정 판정 기준 (결과를 보고 바꾸지 않는다)
--------------------------------------------------
G-A (1차 게이트, 중심값):  Δu_central = Δs/2 + K1·E[Δ²]/4  ≥ +12 (LB 노이즈 바닥)
     여기서 Δs 는 우리 arm 단독 skill 델타, Δ 는 arm 최종예측의 변화.
G-B (참고, CS 상한):       Δu_max = Δu_central + (K1/2)·d·m
     이 항은 기대값 0 이고 부호를 못 재므로 **판정에 쓰지 않는다**(517 §2 "상한은 상한").
δ 의 현실 범위는 **이미 학습된 대안 NN 캐시**(mlpdiv / rankgauss / emb_E1)에서 실측한다.
지어낸 δ 를 쓰지 않는다.

판정:
  · 실측 가능한 최대 δ 로도 G-A < 12  → **상한미달**. 2단계(신규 NN 학습) 진행하지 않는다.
  · G-A ≥ 12 인 δ 가 실측 범위 안  → 2단계 진행.

실행:
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  venv311/bin/python3 harness/exp_nn_ceiling.py
"""
import os
import sys
import numpy as np

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
INNER = [2021, 2022, 2023]
OUTER = 2024
FOLDS = INNER + [OUTER]
SEEDS = [7, 123, 2025, 31415, 8675309]

# LB 실측 3점 (516/517). m = A-B RMS 불일치.
LB_A, LB_B, LB_BLEND = 1032.137582, 1016.4138496773, 1054.8707763763
M_AB = (LB_BLEND - 0.5 * (LB_A + LB_B)) * 2 * (2 * 0.25) / 1e5   # E[(pA-pB)^2]
m_AB = M_AB ** 0.5                                               # 0.017492
C_STAR = 0.5 * (LB_A + LB_B) + 2 * (LB_BLEND - 0.5 * (LB_A + LB_B))  # c = 1085.47


def sk(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


def load(y):
    yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
    bag = []
    for s in SEEDS:
        f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
        if os.path.exists(f):
            bag.append(dict(np.load(f)))
    return yv, bag


def bagged(bag, mlp_override=None):
    """프로덕션과 동일한 예측 배깅. mlp_override 는 seed 별 mlp 예측 리스트."""
    out = []
    for i, P in enumerate(bag):
        Q = dict(P)
        if mlp_override is not None:
            Q['mlp'] = mlp_override[i]
        out.append(predict(dict(PROD), Q))
    return np.mean(out, axis=0)


def alt_mlp(tag, y, transform=None):
    """대안 캐시 디렉토리에서 seed 별 mlp 예측을 읽는다. 없으면 None."""
    d = os.path.join(LG, f'harness/cache_{tag}')
    out = []
    for s in SEEDS:
        f = os.path.join(d, f'pred_{y}_{s}.npz')
        if not os.path.exists(f):
            return None
        out.append(dict(np.load(f))['mlp'])
    if transform is not None:
        base = [dict(np.load(os.path.join(CACHE, f'pred_{y}_{s}.npz')))['mlp'] for s in SEEDS]
        out = [transform(b, a) for b, a in zip(base, out)]
    return out


def section(t):
    print('\n' + '=' * 84)
    print(t)
    print('=' * 84)


def gains(pA, pC, yv, V):
    """[G3] 의 두 항. 반환: (Δs, d, Δu_central, CS 반폭)"""
    K1 = 1e5 / V
    D = pC - pA
    ds = sk(pC, yv, V) - sk(pA, yv, V)
    d2 = float((D * D).mean())
    d = d2 ** 0.5
    central = ds / 2 + K1 * d2 / 4
    halfwidth = (K1 / 2) * d * m_AB
    return ds, d, central, halfwidth


def main():
    print(__doc__.split('실행:')[0].strip()[:0] or '', end='')
    print(f'LB 가 못박은 값:  m = RMS(p_A − p_B) = {m_AB:.6f}   c_AB = {C_STAR:.2f}')
    print(f'                  현재 블렌드 u0 = {LB_BLEND:.4f}')
    print('  주의: c_AB 는 "A 계열 arm 을 무한히 붙였을 때"의 천장이다(516 결과2).')
    print('        A 를 다른 계열 C 로 바꾸면 c 자체가 바뀌므로 u ≤ c_AB 는 성립하지 않는다.')

    # -----------------------------------------------------------------
    section('0. 항등식 [G3] 자체 검증 — pseudo-B 로 수치 확인 (오차가 0 이어야 한다)')
    PSEUDO_B = dict(PROD)
    PSEUDO_B.update(w_lgb=0.05, w_cb=0.95, w_xgb=0.00,
                    w_gbdt=1.00, w_mlp=0.00, w_mse=0.00, scale=1.00, shift=0.0)
    rng = np.random.default_rng(0)
    print(f'  {"fold":>6} {"Δ종류":>10} {"Δu 실측":>10} {"[G3]":>10} {"오차":>10}')
    for y in FOLDS:
        yv, bag = load(y)
        V = yv.mean() * (1 - yv.mean())
        K1 = 1e5 / V
        pA = bagged(bag)
        pB = np.mean([predict(dict(PSEUDO_B), P) for P in bag], axis=0)
        u0 = sk(0.5 * (pA + pB), yv, V)
        for nm, Draw in [('noise', rng.normal(0, 0.02, yv.size)),
                         ('shift', np.full(yv.size, 0.004))]:
            pC = np.clip(pA + Draw, 1e-6, 1 - 1e-6)
            D = pC - pA
            du = sk(0.5 * (pC + pB), yv, V) - u0
            ds = sk(pC, yv, V) - sk(pA, yv, V)
            g3 = ds / 2 + K1 * (D * D).mean() / 4 + (K1 / 2) * (D * (pA - pB)).mean()
            print(f'  {y:>6} {nm:>10} {du:10.3f} {g3:10.3f} {abs(du-g3):10.2e}')

    # -----------------------------------------------------------------
    section('1. 오라클 상한 — mlp 성분을 완벽한 예측(y)으로 바꿨을 때')
    print('  (달성 불가능한 절대 상한. 이것이 작으면 축 자체가 죽는다.)')
    print(f'  {"fold":>6} {"Δs(arm)":>10} {"d=RMS(Δ)":>10} {"Δu중심":>10} '
          f'{"CS±":>9} {"블렌드 u":>10}')
    for y in FOLDS:
        yv, bag = load(y)
        V = yv.mean() * (1 - yv.mean())
        pA = bagged(bag)
        pC = bagged(bag, mlp_override=[yv] * len(bag))
        ds, d, cen, hw = gains(pA, pC, yv, V)
        print(f'  {y:>6} {ds:+10.1f} {d:10.5f} {cen:+10.1f} {hw:9.1f} '
              f'{LB_BLEND + cen:10.1f}')

    # -----------------------------------------------------------------
    section('2. 실측 δ — 이미 학습된 "다른 NN" 들이 실제로 얼마나 다른가')
    print('  δ = 성분 수준 RMS(mlp_alt − mlp_base). arm 수준 d ≈ scale·w_mlp·δ = 0.44·δ')
    print('  mlpdiv 캐시는 0.5*base+0.5*alt 를 저장하므로 alt 를 복원해서 잰다.')
    print(f'\n  {"변형":>22} {"fold":>6} {"δ(성분)":>10} {"d(arm)":>10} '
          f'{"Δs(arm)":>10} {"Δu중심":>10} {"CS±":>8}')
    variants = {
        'mlpdiv alt (256,128)': ('mlpdiv', lambda b, a: 2 * a - b),   # alt = 2*mix - base
        'rankgauss (입력변환)': ('rankgauss', None),
        'emb_E1 (투수임베딩)': ('emb_E1', None),
        'hyperdiv (lgb만 변경)': ('hyperdiv', None),
    }
    summary = {}
    for nm, (tag, tf) in variants.items():
        rows = []
        for y in INNER:
            alt = alt_mlp(tag, y, tf)
            if alt is None:
                continue
            yv, bag = load(y)
            V = yv.mean() * (1 - yv.mean())
            base_mlp = [P['mlp'] for P in bag]
            delta_c = float(np.mean([((a - b) ** 2).mean()
                                     for a, b in zip(alt, base_mlp)])) ** 0.5
            pA = bagged(bag)
            pC = bagged(bag, mlp_override=alt)
            ds, d, cen, hw = gains(pA, pC, yv, V)
            rows.append((y, delta_c, d, ds, cen, hw))
            print(f'  {nm:>22} {y:>6} {delta_c:10.5f} {d:10.5f} {ds:+10.1f} '
                  f'{cen:+10.1f} {hw:8.1f}')
        if rows:
            summary[nm] = rows
            a = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
            print(f'  {nm:>22} {"평균":>6} {a[:,0].mean():10.5f} {a[:,1].mean():10.5f} '
                  f'{a[:,2].mean():+10.1f} {a[:,3].mean():+10.1f} {a[:,4].mean():8.1f}')

    # 시드간 차이 = "같은 아키텍처, 다른 초기화" 의 δ (참조 하한)
    print('\n  참조 A: 같은 아키텍처 시드간 δ (순수 초기화 노이즈의 크기)')
    for y in INNER:
        yv, bag = load(y)
        mm = [P['mlp'] for P in bag]
        pair = [float(((mm[i] - mm[j]) ** 2).mean()) ** 0.5
                for i in range(len(mm)) for j in range(i + 1, len(mm))]
        print(f'    {y}: 시드쌍 δ 평균 {np.mean(pair):.5f}')

    print('\n  참조 B: δ 의 체계적 성분 vs 시드노이즈 성분 분해')
    print('    δ_sys = RMS(배깅 alt − 배깅 base)  ← 배깅 후 살아남는 진짜 차이')
    print('    δ_seed = √(δ² − δ_sys²)            ← 배깅으로 사라지는 노이즈')
    print('    517: 시드노이즈로 산 다양성은 c 를 정확히 0 만큼 올린다(노이즈 불변성).')
    print(f'    {"변형":>22} {"fold":>6} {"δ":>9} {"δ_sys":>9} {"δ_seed":>9} {"sys비중":>8}')
    for nm, (tag, tf) in variants.items():
        if nm not in summary:
            continue
        for y in INNER:
            alt = alt_mlp(tag, y, tf)
            if alt is None:
                continue
            yv, bag = load(y)
            base_mlp = [P['mlp'] for P in bag]
            d2 = float(np.mean([((a - b) ** 2).mean() for a, b in zip(alt, base_mlp)]))
            dsys = float(((np.mean(alt, 0) - np.mean(base_mlp, 0)) ** 2).mean())
            print(f'    {nm:>22} {y:>6} {d2**0.5:9.5f} {dsys**0.5:9.5f} '
                  f'{max(d2-dsys,0)**0.5:9.5f} '
                  f'{(dsys/d2 if d2 > 0 else 0.0):8.2%}')

    # -----------------------------------------------------------------
    section('2b. 교차 계열 앵커 — mlp 를 완전히 다른 모델 계열로 통째 교체')
    print('  "아키텍처 계열이 다른 NN" 이 낼 수 있는 다양성의 현실적 최대치를,')
    print('  이미 가진 완전히 다른 계열(GBDT)로 대리 측정한다. NN 끼리의 차이는')
    print('  이보다 클 수 없다고 보는 것이 합리적이다.')
    print(f'\n  {"교체 대상":>22} {"fold":>6} {"δ(성분)":>10} {"d(arm)":>10} '
          f'{"Δs(arm)":>10} {"Δu중심":>10} {"1e5·d²":>9}')
    for nm, key in [('mlp→lgb_mse', 'lgb_mse'), ('mlp→cb_bin', 'cb_bin'),
                    ('mlp→lgb_bin', 'lgb_bin')]:
        rows = []
        for y in INNER:
            yv, bag = load(y)
            V = yv.mean() * (1 - yv.mean())
            alt = [P[key] for P in bag]
            base_mlp = [P['mlp'] for P in bag]
            dc = float(np.mean([((a - b) ** 2).mean()
                                for a, b in zip(alt, base_mlp)])) ** 0.5
            pA = bagged(bag)
            pC = bagged(bag, mlp_override=alt)
            ds, d, cen, hw = gains(pA, pC, yv, V)
            rows.append((dc, d, ds, cen))
            print(f'  {nm:>22} {y:>6} {dc:10.5f} {d:10.5f} {ds:+10.1f} '
                  f'{cen:+10.1f} {1e5*d*d:9.1f}')
        a = np.array(rows)
        print(f'  {nm:>22} {"평균":>6} {a[:,0].mean():10.5f} {a[:,1].mean():10.5f} '
              f'{a[:,2].mean():+10.1f} {a[:,3].mean():+10.1f} '
              f'{1e5*(a[:,1]**2).mean():9.1f}')

    # -----------------------------------------------------------------
    section('3. δ → Δu 환산표 (등품질 교체 가정 Δs=0, V=0.25)')
    print('  Δu_central = 1e5·(0.44·δ)² = 1.936e4·δ²')
    print(f'  {"δ(성분)":>10} {"d(arm)":>10} {"Δu중심":>10} {"CS 상한":>10}')
    for dc in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100]:
        d = 0.44 * dc
        cen = 1e5 * d * d
        print(f'  {dc:10.4f} {d:10.5f} {cen:+10.2f} {cen + 2e5*d*m_AB:+10.2f}')
    need = (12.0 / 1e5) ** 0.5 / 0.44
    print(f'\n  → Δu_central ≥ 12 를 만들려면 성분 δ ≥ {need:.4f} '
          f'(등품질 교체를 가정해도)')
    print(f'  → mlp 를 절반만 섞으면(0.5*old+0.5*new) δ 가 반이 되어 이득은 1/4 이다. '
          f'그때 필요한 원 δ ≥ {2*need:.4f}')

    # -----------------------------------------------------------------
    section('4. 필요조건 역산 — 통과하려면 신규 NN 이 무엇을 해내야 하는가')
    print('  두 경로가 있고 둘 다 [G3] 의 항이다:')
    print('    (i)  skill 경로   Δu ≈ Δs/2   →  블렌드 +12 에 arm Δs ≥ +24')
    print('                      (보수적 [G2] Δ⊥r_B 가정이면 Δs ≥ +48)')
    print(f'         mlp 유효가중치 {PROD["w_mlp"]:.2f} 이므로 성분 단독 델타는 '
          f'{24/PROD["w_mlp"]:.0f} ~ {48/PROD["w_mlp"]:.0f} 점 필요')
    print('    (ii) 다양성 경로  Δu ≈ 1e5·d²  →  블렌드 +12 에 배깅 arm d ≥ '
          f'{(12/1e5)**0.5:.5f}')
    print(f'         = 배깅 성분 수준 δ_sys ≥ {(12/1e5)**0.5/0.44:.4f}')
    print('\n  두 경로의 트레이드오프 프론티어 (Δu_central = Δs/2 + 1e5·d² = 12):')
    print(f'  {"Δs(arm)":>10} {"필요 d":>10} {"필요 δ_sys":>12}')
    for ds in [-100, -50, -25, 0, +10, +20]:
        rem = 12 - ds / 2
        if rem <= 0:
            print(f'  {ds:+10.0f} {"불필요":>10} {"—":>12}')
            continue
        dneed = (rem / 1e5) ** 0.5
        print(f'  {ds:+10.0f} {dneed:10.5f} {dneed/0.44:12.4f}')
    print('\n  참고: 팀 천장 c_AB = 1085.5 는 "A 계열을 더 붙이는" 경로의 천장이다.')
    print('  A 를 다른 계열로 교체하는 이 실험은 그 천장의 지배를 받지 않는다.')
    print('  그러나 아래 5절의 실측 δ_sys 가 필요치에 얼마나 못 미치는지가 실질 제약이다.')

    # -----------------------------------------------------------------
    section('5. 판정 (게이트 G-A: Δu_central ≥ +12)')
    if summary:
        print(f'  {"변형":>22} {"d(배깅arm)":>11} {"Δs":>9} {"Δu중심":>9} {"G-A":>6}')
        for nm, rows in summary.items():
            a = np.array([[r[2], r[3], r[4]] for r in rows])
            if a[:, 0].mean() == 0:
                continue
            print(f'  {nm:>22} {a[:,0].mean():11.5f} {a[:,1].mean():+9.1f} '
                  f'{a[:,2].mean():+9.1f} {"통과" if a[:,2].mean()>=12 else "미달":>6}')
        best = max((kv for kv in summary.items()
                    if np.mean([r[2] for r in kv[1]]) > 0),
                   key=lambda kv: np.mean([r[4] for r in kv[1]]))
        dbest = float(np.mean([r[2] for r in best[1]]))
        cbest = float(np.mean([r[4] for r in best[1]]))
        print(f'\n  실측 최선 = {best[0]}: d={dbest:.5f}, Δu_central={cbest:+.1f}')
        print(f'  다양성만으로도(Δs=0 가정) 1e5·d² = {1e5*dbest*dbest:+.2f}')
        print(f'  필요치 대비 d 배율 = {(12/1e5)**0.5/dbest:.2f}배 '
              f'(분산 기준 {12/(1e5*dbest*dbest):.2f}배)')

    # -----------------------------------------------------------------
    section('6. 최종 상한 — in-sample 최적 mlp 대체 (의도적으로 낙관적)')
    print('  mlp 슬롯을 {현행 mlp, 이미 학습된 대안 NN 3종, GBDT 3성분} 의 임의')
    print('  볼록결합으로 바꿀 수 있다고 두고, 폴드마다 Δu_central 을 in-fold 로')
    print('  최대화한다. 라벨을 보고 고르는 것이므로 **달성 불가능한 상한**이다.')
    print('  이 상한마저 +12 미만이면 이 축은 산술적으로 닫힌다.')
    from scipy.optimize import minimize  # noqa: E402
    print(f'\n  {"fold":>6} {"V":>7} {"n_pool":>7} {"Δs":>9} {"d":>9} '
          f'{"Δu중심(상한)":>13}')
    pool_specs = [('mlpdiv', lambda b, a: 2 * a - b), ('rankgauss', None),
                  ('emb_E1', None)]
    for y in INNER:
        yv, bag = load(y)
        V = yv.mean() * (1 - yv.mean())
        K1 = 1e5 / V
        pool = [[P['mlp'] for P in bag]]
        for tag, tf in pool_specs:
            a = alt_mlp(tag, y, tf)
            if a is not None:
                pool.append(a)
        for k in ['lgb_mse', 'cb_bin', 'lgb_bin']:
            pool.append([P[k] for P in bag])
        pA = bagged(bag)

        def neg(w):
            w = np.abs(w); w = w / w.sum()
            mix = [sum(wi * pool[i][s] for i, wi in enumerate(w))
                   for s in range(len(bag))]
            pC = bagged(bag, mlp_override=mix)
            D = pC - pA
            ds = sk(pC, yv, V) - sk(pA, yv, V)
            return -(ds / 2 + K1 * (D * D).mean() / 4)

        # 상한이므로 최적화가 국소해에 갇히면 안 된다 — 다중 재시작.
        starts = [np.full(len(pool), 1.0 / len(pool))]
        for i in range(len(pool)):
            v = np.full(len(pool), 0.05); v[i] = 1.0
            starts.append(v / v.sum())
        rng2 = np.random.default_rng(1)
        starts += [rng2.dirichlet(np.ones(len(pool))) for _ in range(8)]
        best_w, best_v = None, np.inf
        for s0 in starts:
            r = minimize(neg, s0, method='Nelder-Mead',
                         options=dict(maxiter=8000, maxfev=8000, fatol=1e-4))
            if r.fun < best_v:
                best_v, best_w = r.fun, r.x
        w = np.abs(best_w); w = w / w.sum()
        mix = [sum(wi * pool[i][s] for i, wi in enumerate(w))
               for s in range(len(bag))]
        pC = bagged(bag, mlp_override=mix)
        ds, d, cen, hw = gains(pA, pC, yv, V)
        print(f'  {y:>6} {V:7.4f} {len(pool):>7} {ds:+9.1f} {d:9.5f} {cen:+13.1f}')
        print(f'         가중치 = ' + ' '.join(f'{v:.3f}' for v in w))


if __name__ == '__main__':
    main()
