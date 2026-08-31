"""ensemble.py — 블렌드 수학. `toolkit/blend_math.py` 의 플레이북판."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from ._base import method

EPS = 1e-6


def brier_skill(p, y, C=1e5):
    p, y = np.asarray(p, float), np.asarray(y, float)
    r = y.mean()
    return C * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


@method(id='ens.gram', stage=8, status='ADOPTED', cost='low',
        title='블렌드 Gram 행렬 — 종이 계산으로 실험 대체',
        gain='실제 arm 5종의 관측 기여를 최대오차 0.02 로 재현',
        evidence='Brier 는 예측벡터의 정확한 2차형식. 6개 수치로 3-arm 문제가 풀린다',
        requires=[], note='★ 대회 첫날 지표가 2차형식인지 먼저 확인하라.')
def gram(preds, y, C=1e5):
    y = np.asarray(y, float); r = y.mean(); V = r * (1 - r)
    E = [np.asarray(p, float) - y for p in preds]; n = len(E)
    return np.array([[C * (1 - (E[i] * E[j]).mean() / V) for j in range(n)] for i in range(n)])


def opt(M, nonneg=True):
    n = len(M); best = (-np.inf, None)
    idxs = ([list(c) for k in range(1, n + 1) for c in combinations(range(n), k)]
            if nonneg else [list(range(n))])
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
            f = np.zeros(n); f[I] = w; best = (u, f)
    return best[1], best[0]


@method(id='ens.rho_gate', stage=8, status='ADOPTED', cost='low',
        title='ρ — 이득을 결정하는 유일한 값',
        gain='Arm C 축을 상한 논증으로 종결 (필요 1.10% vs 가용 0.72%)',
        evidence='Δ = (C − s)·ρ². 다양성 크기도 단독 스킬도 레버가 아니다',
        requires=[],
        note='★★ 새 arm 학습 전에 반드시 이걸 재라. 크기를 3배로 키워도 기여는 불변이고 '
             'w 만 1/m 로 준다 — "corr 를 낮추자"는 잘못된 목표다.')
def rho_of(p_new, preds, y):
    y = np.asarray(y, float)
    E = np.stack([np.asarray(p, float) - y for p in preds], 1)
    w, _ = opt(gram(list(preds), y))
    resid = sum(wi * (np.asarray(p, float) - y) for wi, p in zip(w, preds))
    A = E[:, 1:] - E[:, :1] if E.shape[1] > 1 else np.zeros((len(y), 0))
    e = np.asarray(p_new, float) - y; base = E[:, 0]
    foot = base + A @ np.linalg.lstsq(A, e - base, rcond=None)[0] if A.shape[1] else base
    orth = e - foot
    return 0.0 if orth.std() < 1e-15 else float(abs(np.corrcoef(resid, orth)[0, 1]))


def required_rho(target_gain, base_skill, C=1e5):
    return float(np.sqrt(max(target_gain, 0) / (C - base_skill)))


@method(id='ens.gated_blend', stage=8, status='ADOPTED', cost='low',
        title='게이팅 블렌드 — 행 속성별 가중치',
        gain='arm 의 약점 구간을 우회 (이 대회: 특정 리그 행만 0.55→0.20)',
        evidence='행 자신의 컬럼으로만 가중치를 정하므로 행 독립이 유지된다',
        requires=[], note='어떤 arm 이 특정 구간에서 구조적으로 약할 때.')
def gated_blend(pA, pB, mask, w_default=0.55, w_masked=0.20):
    w = np.where(np.asarray(mask, bool), w_masked, w_default)
    return w * np.asarray(pA, float) + (1 - w) * np.asarray(pB, float)


@method(id='ens.library_ceiling', stage=8, status='ADOPTED', cost='low',
        title='라이브러리 천장 — 가진 것 전부로 얼마까지 가능한가',
        gain='57종 전체 조합의 진짜 천장 +2.73 (교사 포함 시 +148.8 로 오염됐었다)',
        evidence='가중치는 fit 폴드에서만 적합, eval 폴드에서 실현치를 본다',
        requires=[], note='★ 캠페인을 계속할지 접을지 결정할 때. 릿지 강도를 반드시 훑을 것.')
def library_ceiling(preds_fit, y_fit, preds_eval, y_eval, lams=(1e2, 1e3, 1e4, 1e5, 1e6)):
    yf, ye = np.asarray(y_fit, float), np.asarray(y_eval, float)
    bf, be = np.mean(preds_fit, 0), np.mean(preds_eval, 0)
    Xf = np.stack([np.asarray(p, float) - bf for p in preds_fit], 1)
    Xe = np.stack([np.asarray(p, float) - be for p in preds_eval], 1)
    G, g = Xf.T @ Xf, Xf.T @ (yf - bf)
    base = brier_skill(be, ye); out = []
    for lam in lams:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), g)
        out.append((lam,
                    brier_skill(np.clip(bf + Xf @ w, EPS, 1 - EPS), yf) - brier_skill(bf, yf),
                    brier_skill(np.clip(be + Xe @ w, EPS, 1 - EPS), ye) - base))
    return out


@method(id='ens.stacking', stage=8, status='REJECTED', cost='med',
        title='스태킹 (메타 모델 결합)',
        gain='단순 가중 블렌드를 넘지 못함',
        evidence='stack_lr / stack_cb 시도. 최종 채택은 전부 선형 블렌드',
        requires=[], note='자유도가 크고 OOF 구성이 조금만 틀려도 누출. 시간 외삽에 특히 약하다.')
def stacking(oof_preds_fit, y_fit, preds_apply, model=None):
    from sklearn.linear_model import LogisticRegression
    X = np.stack([np.asarray(p, float) for p in oof_preds_fit], 1)
    Xa = np.stack([np.asarray(p, float) for p in preds_apply], 1)
    m = model or LogisticRegression(max_iter=1000)
    m.fit(X, np.asarray(y_fit, float).astype(int))
    return np.clip(m.predict_proba(Xa)[:, 1], EPS, 1 - EPS)


@method(id='ens.seed_bagging', stage=2, status='ADOPTED', cost='med',
        title='시드 예측 배깅',
        gain='표준 채택. 단 시드를 더 늘리는 이득은 빠르게 포화',
        evidence='5-seed 표준. 15-seed SWA 로 바꾸니 LB 1032→1020',
        requires=[], note='가중치 평균이 아니라 **예측 평균**이다.')
def seed_bagging(pred_list):
    return np.mean([np.asarray(p, float) for p in pred_list], 0)
