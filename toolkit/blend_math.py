#!/usr/bin/env python3
"""blend_math.py — 앙상블 블렌드의 닫힌형 계산 (대회 무관, 범용).

Brier/MSE 계열 지표는 예측 벡터의 **정확한 2차형식**이다. 그래서 arm 을 실제로 섞어보지 않고
**종이 위에서** 최적 가중치·기여도·필요 스킬을 계산할 수 있다.

    skill_i = C · (1 − MSE_i / V)                    (Brier skill 이면 C=1e5, V=r(1−r))
    u(w)    = wᵀ M w,  Σw = 1
    M_ii    = s_i
    M_ij    = (s_i + s_j)/2 + (C / 2V) · d_ij²       d_ij = ‖p_i − p_j‖_RMS

2026 Aimers 실측 검증: 실제 arm 5종의 관측 기여도를 **최대오차 0.02** 로 재현했다.

## 이 도구가 답해 주는 것

1. 새 arm 을 추가하면 얼마나 오르나?              `contribution()`
2. 목표 점수에 필요한 arm 스펙은?                  `required_rho()` / `required_skill()`
3. 우리 arm 라이브러리의 천장은?                   `ceiling()`
4. "다양성을 키우면" 오르나?                       → **아니다.** `demo_scale_invariance()`

## 핵심 사실 (실측으로 확인)

- **고유방향의 크기는 이득과 무관하다.** 직교성분을 m 배 키워도 최적 기여는 **불변**,
  가중치 w 만 정확히 1/m 로 준다. → "corr 를 낮춰라" 는 잘못된 목표다.
- **단독 스킬도 레버가 아니다.** 스킬을 올리면 예측이 기존 arm 합의로 끌려가 다양성이 깎인다
  (실측 상관 −0.98). 단독 최고 모델이 최적 가중치 0.000 을 받은 사례가 있다.
- **이득은 ρ = corr(현재 잔차, 새 arm 의 고유방향) 하나로 결정된다.**

    python3 blend_math.py --demo        # 자기검증 + 불변성 시연
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


def skill(p, y, C=1e5):
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    return C * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def gram(preds, y, C=1e5):
    """M_ij — 이 행렬 하나가 모든 블렌드 조합의 점수를 결정한다."""
    y = np.asarray(y, float); r = y.mean(); V = r * (1 - r)
    E = [np.asarray(p, float) - y for p in preds]
    n = len(E)
    return np.array([[C * (1 - (E[i] * E[j]).mean() / V) for j in range(n)] for i in range(n)])


def gram_from_stats(s, d, y_rate, C=1e5):
    """예측 벡터 없이 **6개 수치만으로** M 을 세운다 (s_i, d_ij, 라벨률)."""
    V = y_rate * (1 - y_rate); k = C / (2 * V); n = len(s)
    return np.array([[s[i] if i == j else (s[i] + s[j]) / 2 + k * d[i][j] ** 2
                      for j in range(n)] for i in range(n)])


def opt(M, nonneg=True):
    """심플렉스(또는 아핀) 위 최적 가중치. n 이 작으면 부분집합 전수가 가장 안전하다."""
    n = len(M)
    best = (-np.inf, None)
    idxs = [list(c) for k in range(1, n + 1) for c in combinations(range(n), k)] if nonneg \
        else [list(range(n))]
    for I in idxs:
        A = M[np.ix_(I, I)]
        try:
            v = np.linalg.solve(A, np.ones(len(I)))
        except np.linalg.LinAlgError:
            continue
        t = v.sum()
        if abs(t) < 1e-12:
            continue
        w = v / t
        if nonneg and (w < -1e-9).any():
            continue
        u = float(w @ A @ w)
        if u > best[0]:
            full = np.zeros(n); full[I] = w
            best = (u, full)
    return best[1], best[0]


def contribution(preds, y, C=1e5, nonneg=True):
    """마지막 arm 을 추가했을 때의 기여 (전체 최적 − 그것을 뺀 최적).

    ⚠️ `nonneg=True` 는 심플렉스(가중치 ≥ 0) 위 최적이다 — 실제 제출에서 흔히 쓰는 제약.
       `nonneg=False` 는 **아핀** 최적(음수 허용)이고, `required_rho` 의 항등식
       Δ=(C−s)·ρ² 은 이쪽을 가정한다. 두 값이 다르면 최적해가 경계에 붙은 것이다."""
    M = gram(preds, y, C)
    w_all, u_all = opt(M, nonneg)
    _, u_base = opt(M[:-1, :-1], nonneg)
    return u_all - u_base, w_all


def unique_direction(p_new, preds, y):
    """새 arm 오차를 기존 arm 오차들의 **아핀 부분공간**에 투영하고 남은 성분."""
    y = np.asarray(y, float)
    E = np.stack([np.asarray(p, float) - y for p in preds], 1)
    e = np.asarray(p_new, float) - y
    A = E[:, 1:] - E[:, :1] if E.shape[1] > 1 else np.zeros((len(y), 0))
    base = E[:, 0]
    if A.shape[1]:
        c = np.linalg.lstsq(A, e - base, rcond=None)[0]
        foot = base + A @ c
    else:
        foot = base
    return e - foot, foot


def rho_of(p_new, preds, y):
    """ρ = corr(현재 최적 블렌드의 잔차, 새 arm 의 고유방향). 이득을 결정하는 유일한 값."""
    M = gram(list(preds), y)
    w, _ = opt(M)
    y = np.asarray(y, float)
    resid = sum(wi * (np.asarray(p, float) - y) for wi, p in zip(w, preds))
    orth, _ = unique_direction(p_new, preds, y)
    if orth.std() < 1e-15:
        return 0.0
    return float(abs(np.corrcoef(resid, orth)[0, 1]))


def required_rho(target_gain, base_skill, C=1e5):
    """목표 이득을 내려면 ρ 가 얼마여야 하나. Δ = (C − s)·ρ²"""
    return float(np.sqrt(max(target_gain, 0) / (C - base_skill)))


def ceiling(preds_fit, y_fit, preds_eval, y_eval, lams=(1e2, 1e3, 1e4, 1e5, 1e6)):
    """라이브러리 천장 — **가중치는 fit 폴드에서만** 적합하고 eval 폴드에서 실현치를 본다.
    한 폴드에서 적합·평가하면 반드시 과적합한다(실측: 2022 +148.9 / 2024 −15.9)."""
    yf, ye = np.asarray(y_fit, float), np.asarray(y_eval, float)
    bf = np.mean(preds_fit, 0); be = np.mean(preds_eval, 0)
    Xf = np.stack([np.asarray(p, float) - bf for p in preds_fit], 1)
    Xe = np.stack([np.asarray(p, float) - be for p in preds_eval], 1)
    G, g = Xf.T @ Xf, Xf.T @ (yf - bf)
    base_e = skill(be, ye)
    out = []
    for lam in lams:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), g)
        out.append((lam, skill(np.clip(bf + Xf @ w, 1e-6, 1 - 1e-6), yf) - skill(bf, yf),
                    skill(np.clip(be + Xe @ w, 1e-6, 1 - 1e-6), ye) - base_e))
    return out


def demo():
    rng = np.random.default_rng(0)
    n = 60_000
    y = (rng.random(n) < .48).astype(float)
    def arm(sig, noise):
        z = sig * (y - .5) + noise * rng.standard_normal(n)
        return np.clip(.48 + .05 * np.tanh(z), 1e-6, 1 - 1e-6)
    hidden = rng.standard_normal(n)                 # C 만 보는 별도 정보원
    A, B = arm(.9, 1.), arm(.85, 1.)
    zc = .5 * (y - .5) + .55 * hidden + 1. * rng.standard_normal(n)
    Cc = np.clip(.48 + .05 * np.tanh(zc), 1e-6, 1 - 1e-6)
    print('[1] gram_from_stats 가 실제 gram 을 재현하는가')
    M = gram([A, B, Cc], y)
    s = [skill(A, y), skill(B, y), skill(Cc, y)]
    d = [[0, 0, 0]] * 3
    d = [[float(np.sqrt(((x - z) ** 2).mean())) for z in (A, B, Cc)] for x in (A, B, Cc)]
    M2 = gram_from_stats(s, d, y.mean())
    print(f'    최대 오차 {np.abs(M - M2).max():.2e}  -> '
          f'{"✅ 6개 수치만으로 계산 가능" if np.abs(M-M2).max() < 1e-6 else "🔴"}')
    print('\n[2] 고유방향 크기 불변성 — "다양성을 키우면 오른다" 검증')
    orth, foot = unique_direction(Cc, [A, B], y)
    print(f'    {"배율":>6}{"corr(C,B)":>12}{"기여(아핀)":>12}{"w_C":>9}')
    for m in (1., 2., 3.):
        pc = y + foot + m * orth
        g, w = contribution([A, B, pc], y, nonneg=False)
        print(f'    {m:>6.1f}{np.corrcoef(pc, B)[0,1]:>12.3f}{g:>12.2f}{w[2]:>9.3f}')
    print('    -> 기여는 불변, w 만 1/m 로 준다. **크기는 레버가 아니다.**')
    print('\n[3] 필요 ρ 역산')
    _, u2 = opt(gram([A, B], y), nonneg=False)
    for t in (12, 30, 50):
        print(f'    +{t} 이득에 필요한 ρ = {100*required_rho(t, u2):.2f}%   (현 base {u2:.1f})')
    r = rho_of(Cc, [A, B], y)
    g, _ = contribution([A, B, Cc], y, nonneg=False)
    print(f"    실제 C 의 ρ = {100*r:.2f}%  ->  예측 이득 {(1e5-u2)*r*r:+.2f}"
          f"  /  실측 {g:+.2f}   {'✅ 일치' if abs((1e5-u2)*r*r-g) < max(1.0, .05*abs(g)) else '🔴'}")


if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        demo()
    else:
        print(__doc__)
