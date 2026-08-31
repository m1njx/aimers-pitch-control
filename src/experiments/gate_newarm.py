"""
gate_newarm.py — 새 arm 하나를 사전등록 게이트로 즉시 판정한다.

왜 필요한가: 08-27~28 에 새 arm 후보가 5개 들어왔는데(arm_c, v33, v34, super_b, xgb)
전부 **홀드아웃 점수 없이** 왔다. 그때마다 내가 모델을 역추론해 재현하느라 매번 수십 분이
들었고, v33 은 그 검증 공백 때문에 LB −3.97 을 실제로 잃었다.

이 스크립트는 후보의 **2024 예측 한 개**만 있으면 판정을 끝낸다.

    python3 harness/gate_newarm.py pred_2024.npy --name my_arm

⚠️ 입력 예측이 반드시 만족해야 하는 것 (아니면 판정이 무의미하다):
   `train<2024` 로 학습해 2024 를 예측한 것이어야 한다. 전 시즌 학습본을 넣으면
   2024 가 in-sample 이라 점수가 부풀려진다. 스크립트가 이를 자동 경고한다
   (AUC 가 우리 A/B(≈0.552)를 크게 웃돌면 in-sample 의심).

사전등록 게이트 (526, 08-28 09:40):
  G1  d_AC >= 0.020  그리고  d_BC >= 0.020          (다양성)
  G2  최적가중 3-arm − 2-arm > +12                   (기여)
  G3  2024 를 반으로 갈라 한쪽에서 가중치 적합 → 다른쪽에서도 양수  (자유도 방어)
셋 다 통과해야 배포 후보.
"""
import argparse, glob, os
from itertools import combinations
import numpy as np

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)


def our_arm():
    ps = []
    for f in sorted(glob.glob(os.path.join(CACHE, 'pred_2024_*.npz'))):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def team_arm():
    fs = sorted(glob.glob(os.path.join(LG, 'teamB/out/preds/l2384_f2024_s*.npy')))
    return np.mean([np.load(f).astype(float) for f in fs], 0)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def gram(ps, y):
    r = y.mean(); V = r * (1 - r); n = len(ps)
    return np.array([[1e5 * (1 - ((ps[i] - y) * (ps[j] - y)).mean() / V)
                      for j in range(n)] for i in range(n)])


def opt(M):
    """심플렉스 위 최적 가중치 (부분집합 전수 — n<=3 이라 즉시 끝난다)."""
    n = M.shape[0]; best, bw = -1e18, None
    for k in range(1, n + 1):
        for S in combinations(range(n), k):
            S = list(S); Ms = M[np.ix_(S, S)]
            A = np.block([[2 * Ms, -np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            try:
                w = np.linalg.solve(A, np.concatenate([np.zeros(k), [1.]]))[:k]
            except np.linalg.LinAlgError:
                continue
            if (w < -1e-9).any():
                continue
            v = float(w @ Ms @ w)
            if v > best:
                best, bw = v, np.zeros(n); bw[S] = w
    return bw, best


def recal(p, y, fit=None):
    """아핀 재보정. fit 마스크가 주어지면 그 부분집합에서만 계수를 적합한다."""
    m = np.ones(len(p), bool) if fit is None else fit
    X = np.stack([np.ones(m.sum()), p[m] - 0.5], 1)
    b = np.linalg.lstsq(X, y[m] - p[m], rcond=None)[0]
    return np.clip(p + b[0] + b[1] * (p - 0.5), EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pred', help='2024 예측 .npy (train<2024 학습본이어야 한다)')
    ap.add_argument('--name', default='candidate')
    ap.add_argument('--no-recal', action='store_true', help='아핀 재보정 없이 원본으로 판정')
    a = ap.parse_args()

    y = np.load(os.path.join(CACHE, 'y_2024.npy')).astype(float)
    C = np.clip(np.load(a.pred).astype(float), EPS, 1 - EPS)
    if len(C) != len(y):
        raise SystemExit(f'길이 불일치: 예측 {len(C):,} vs 2024 라벨 {len(y):,}')
    A, B = our_arm(), team_arm()

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, C)
    print(f'=== {a.name} ===')
    print(f'  raw skill {skill(C, y):9.2f}   AUC {auc:.4f}   mean(p) {C.mean():.5f} (r {y.mean():.5f})')
    print(f'  대조: A {skill(A, y):.2f}/AUC {roc_auc_score(y, A):.4f}   '
          f'B {skill(B, y):.2f}/AUC {roc_auc_score(y, B):.4f}')
    if auc > 0.565:
        print('  🔴 경고: AUC 가 A/B(≈0.552)를 크게 웃돈다 → 2024 가 학습에 들어갔을 가능성.')
        print('     그렇다면 아래 판정은 전부 무의미하다. 학습 구간을 확인하라.')

    Cw = C if a.no_recal else recal(C, y)
    if not a.no_recal:
        print(f'  아핀 재보정 후 skill {skill(Cw, y):9.2f}')

    d_AC = float(np.sqrt(((A - Cw) ** 2).mean()))
    d_BC = float(np.sqrt(((B - Cw) ** 2).mean()))
    d_AB = float(np.sqrt(((A - B) ** 2).mean()))
    g1 = (d_AC >= 0.020) and (d_BC >= 0.020)
    print(f'\n[G1] 다양성   d_AC {d_AC:.4f}  d_BC {d_BC:.4f}  (기준 둘 다 >=0.020, 참고 d_AB {d_AB:.4f})'
          f'  → {"PASS" if g1 else "FAIL"}')

    _, u2 = opt(gram([A, B], y))
    w3, u3 = opt(gram([A, B, Cw], y))
    g2 = (u3 - u2) > 12
    print(f'[G2] 기여     2-arm {u2:.2f} → 3-arm {u3:.2f}  Δ {u3 - u2:+.2f}  '
          f'w_C {w3[2]:.3f}  (기준 >+12)  → {"PASS" if g2 else "FAIL"}')

    rng = np.random.default_rng(7); h = rng.random(len(y)) < 0.5
    deltas = []
    for fit, ev in ((h, ~h), (~h, h)):
        Cf = C if a.no_recal else recal(C, y, fit)
        w2f, _ = opt(gram([A[fit], B[fit]], y[fit]))
        w3f, _ = opt(gram([A[fit], B[fit], Cf[fit]], y[fit]))
        s2 = skill(w2f[0] * A[ev] + w2f[1] * B[ev], y[ev])
        s3 = skill(w3f[0] * A[ev] + w3f[1] * B[ev] + w3f[2] * Cf[ev], y[ev])
        deltas.append(s3 - s2)
    g3 = all(d > 0 for d in deltas)
    print(f'[G3] 분할검증 전반→후반 {deltas[0]:+.2f}  후반→전반 {deltas[1]:+.2f}  '
          f'(기준 둘 다 >0)  → {"PASS" if g3 else "FAIL"}')

    ok = g1 and g2 and g3
    print(f'\n판정: {"✅ PASS — 배포 후보" if ok else "❌ REJECT"}')
    if ok:
        print(f'  권장 W_C ≈ {w3[2]:.2f} (블렌드 후처리 전이비 ~0.5 → LB 기대 {(u3-u2)*0.5:+.1f})')
        print('  ⚠️ 배포용은 반드시 **전 시즌 학습본**을 쓸 것 (예측 시계 1년 유지, 526 09:10)')


if __name__ == '__main__':
    main()
