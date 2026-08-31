"""
exp_partner_dedup.py — 파트너 인지 재가중 (partner-aware reweighting).

[G2]  Δu = Δs/4 − (K1/2)·E[Δ·r_B]
      → 블렌드를 올리는 자유도는 "우리 arm 을 올리는 것"이 아니라
        "팀 잔차 r_B 와 반대로 정렬되는 것"이다.

가설:
  우리 arm A 의 성분 가중치는 **단독 skill** 최대화로 정해졌다(w_cb 유효 0.288).
  그런데 팀 arm B 는 CatBoost 다. 팀 블렌드 안에서 보면 우리 cb 성분은
  B 와 중복이고, 중복 성분에 실린 가중치는 낭비다. 블렌드 목적함수로 다시
  최적화하면 cb→mlp 쪽으로 무게가 옮겨가면서 Δu > 0 이 나와야 한다.

  · 재학습 0. 캐시 예측의 재가중일 뿐이다.
  · 규정4 저촉 없음(행 독립 예측의 선형결합).
  · 닫힌 축 20개와 다른 종류다 — 피처가 아니라 **팀 블렌드 안에서의 역할**을 바꾼다.

⚠️ 프록시 한계 (반드시 같이 읽을 것):
  진짜 r_B 가 없어 우리 cb_bin 을 팀 CatBoost 대역으로 쓴다.
  corr(cb_bin, mlp)=0.913 은 LB 실측 corr(A,B)=0.926 과 같은 대역이지만,
  **우리 cb 는 팀 cb 보다 우리 arm 과 더 겹친다**(같은 피처·같은 파이프라인).
  따라서 여기서 나오는 이득은 이 메커니즘의 **상한**이다. 확정은 팀 예측이 와야 한다.
  로컬 절대 skill 은 LB 와 역상관이므로(dacon-local-harness-invalid) 절대값이
  아니라 **부호와 방향의 폴드 일관성**만 본다.

실행: venv311/bin/python3 harness/exp_partner_dedup.py
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023, 2024]
SEEDS = [7, 123, 2025, 31415, 8675309]
COMPS = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
EPS = 1e-6

# PROD 를 5성분 유효 가중치로 편 것 (predict() 와 동일한 값을 낸다)
V_PROD = np.array([0.40 * 0.20, 0.40 * 0.72, 0.40 * 0.08, 0.20, 0.40])
S_PROD = np.array([-0.007, -0.008, -0.006, 0.0, 0.0])


def sk(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


def load(y):
    yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
    per_seed = []
    for s in SEEDS:
        f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
        if os.path.exists(f):
            per_seed.append(dict(np.load(f)))
    return yv, per_seed


def arm(v, per_seed, scale=1.10, shift=-0.0045192086):
    """성분 가중치 v 로 만든 arm 의 배깅 예측. v=V_PROD 면 predict(PROD,·) 와 동일."""
    out = []
    for P in per_seed:
        raw = sum(w * np.clip(P[c] + s, EPS, 1 - EPS)
                  for w, s, c in zip(v, S_PROD, COMPS))
        out.append(np.clip(0.5 + scale * (raw - 0.5) + shift, EPS, 1 - EPS))
    return np.mean(out, axis=0)


def pseudo_b(per_seed):
    """팀 CatBoost arm 의 대역폭 스탠드인: cb_bin 단독, 자체 캘리브레이션."""
    out = []
    for P in per_seed:
        raw = np.clip(P['cb_bin'] - 0.008, EPS, 1 - EPS)
        out.append(np.clip(0.5 + 1.10 * (raw - 0.5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(out, axis=0)


def optimize(per_seed, yv, V, pB=None):
    """v 를 심플렉스 위에서 최적화. pB 가 있으면 50:50 블렌드 skill 을,
    없으면 단독 arm skill 을 최대화한다."""
    def neg(v):
        v = np.abs(v); v = v / v.sum() * V_PROD.sum()
        p = arm(v, per_seed)
        if pB is not None:
            p = 0.5 * (p + pB)
        return -sk(p, yv, V)
    r = minimize(neg, V_PROD.copy(), method='Nelder-Mead',
                 options=dict(maxiter=4000, xatol=1e-5, fatol=1e-4))
    v = np.abs(r.x); v = v / v.sum() * V_PROD.sum()
    return v


def main():
    print('=' * 84)
    print('파트너 인지 재가중 — 단독 최적 v  vs  블렌드 최적 v')
    print('=' * 84)
    rows = []
    for y in FOLDS:
        yv, per_seed = load(y)
        V = yv.mean() * (1 - yv.mean())
        pA = arm(V_PROD, per_seed)
        pB = pseudo_b(per_seed)
        rho = np.corrcoef(pA, pB)[0, 1]
        sA, sB = sk(pA, yv, V), sk(pB, yv, V)
        u0 = sk(0.5 * (pA + pB), yv, V)
        D = 2 * (u0 - 0.5 * (sA + sB))

        v_solo = optimize(per_seed, yv, V, None)
        v_team = optimize(per_seed, yv, V, pB)
        p_solo, p_team = arm(v_solo, per_seed), arm(v_team, per_seed)

        u_solo = sk(0.5 * (p_solo + pB), yv, V)
        u_team = sk(0.5 * (p_team + pB), yv, V)

        print(f'\n--- {y}   corr(A,B̃)={rho:.4f}  s_A={sA:8.1f}  s_B̃={sB:8.1f}  '
              f'D={D:6.1f}  u0={u0:8.1f}')
        print(f'    {"":>12} ' + ' '.join(f'{c:>9}' for c in COMPS)
              + f' {"Δs(arm)":>10} {"Δu(blend)":>11}')
        print(f'    {"PROD":>12} ' + ' '.join(f'{w:9.3f}' for w in V_PROD)
              + f' {0.0:+10.2f} {0.0:+11.2f}')
        print(f'    {"단독최적":>12} ' + ' '.join(f'{w:9.3f}' for w in v_solo)
              + f' {sk(p_solo,yv,V)-sA:+10.2f} {u_solo-u0:+11.2f}')
        print(f'    {"블렌드최적":>12} ' + ' '.join(f'{w:9.3f}' for w in v_team)
              + f' {sk(p_team,yv,V)-sA:+10.2f} {u_team-u0:+11.2f}')
        rows.append(dict(y=y, v_team=v_team, v_solo=v_solo,
                         du_team=u_team - u0, du_solo=u_solo - u0,
                         ds_team=sk(p_team, yv, V) - sA))

    print('\n' + '=' * 84)
    print('폴드 일관성 — in-fold 최적화는 과적합이므로 leave-one-fold-out 로 재검증')
    print('=' * 84)
    print(f'  {"적용 폴드":>10} {"가중치 출처":>28} {"Δs(arm)":>10} {"Δu(blend)":>11}')
    oof = []
    for y in FOLDS:
        yv, per_seed = load(y)
        V = yv.mean() * (1 - yv.mean())
        pA, pB = arm(V_PROD, per_seed), pseudo_b(per_seed)
        u0, sA = sk(0.5 * (pA + pB), yv, V), sk(pA, yv, V)
        v = np.mean([r['v_team'] for r in rows if r['y'] != y], axis=0)
        p = arm(v, per_seed)
        du = sk(0.5 * (p + pB), yv, V) - u0
        ds = sk(p, yv, V) - sA
        oof.append(du)
        print(f'  {y:>10} {"다른 3폴드 평균 v_team":>28} {ds:+10.2f} {du:+11.2f}')
    o = np.array(oof)
    t = o.mean() / (o.std(ddof=1) / np.sqrt(o.size)) if o.size > 1 else np.nan
    print(f'\n  LOFO Δu: 평균 {o.mean():+.2f}  전폴드양수={bool((o>0).all())}  t={t:.2f}')

    v_final = np.mean([r['v_team'] for r in rows], axis=0)
    print(f'\n  4폴드 평균 v_team = ' + ' '.join(f'{c}={w:.3f}'
                                              for c, w in zip(COMPS, v_final)))
    print(f'  PROD 대비 이동      = ' + ' '.join(
        f'{c}={w-p:+.3f}' for c, w, p in zip(COMPS, v_final, V_PROD)))
    print('\n  ⚠️ 절대값은 프록시 상한이다. 채택 판정은 팀 실제 예측이 온 뒤.')


if __name__ == '__main__':
    main()
