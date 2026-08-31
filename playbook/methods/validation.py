"""validation.py — 검증 방법론. **가장 비싸게 배운 부분이다.**"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import method


@method(id='val.time_folds', stage=4, status='ADOPTED', cost='low',
        title='시계열 전방 폴드 + 예측 시계 정합',
        gain='폴드 설계 오류로 arm 가치를 과대평가한 사례를 차단',
        evidence='train<Y→Y 는 1년 격차인데 배포가 2년이면 로컬이 과대평가한다',
        requires=['time_col', 'fold_edges'],
        note='★ 새 arm 평가 시 "학습에 쓴 마지막 시점"과 "평가 시점"의 간격을 먼저 적어라.')
def time_folds(df, time_col, edges):
    """반환: [(train_mask, eval_mask, horizon), ...]"""
    t = df[time_col].to_numpy()
    out = []
    for i in range(len(edges) - 1):
        cut, ev = edges[i], edges[i + 1]
        out.append((t <= cut, t == ev, ev - cut))
    return out


@method(id='val.regime_check', stage=4, status='ADOPTED', cost='low',
        title='레짐 변화 점검 — 폴드를 쓰기 전에',
        gain='2군 리그가 r 0.709→0.473 으로 붕괴한 것을 발견, 해당 행 제외',
        evidence='세그먼트별 라벨률·상관이 급변하면 그 폴드는 못 믿는다',
        requires=['time_col'], note='분리 채점하거나 제외한다.')
def regime_check(df, time_col, target_col, segment_col=None):
    g = [time_col] + ([segment_col] if segment_col else [])
    r = df.groupby(g)[target_col].agg(['mean', 'count']).reset_index()
    if segment_col:
        piv = r.pivot(index=time_col, columns=segment_col, values='mean')
        r = r.merge(piv.diff().abs().max(axis=1).rename('drift'), left_on=time_col, right_index=True)
    return r


@method(id='val.noise_floor', stage=3, status='ADOPTED', cost='med',
        title='노이즈 바닥 측정 — 이게 없으면 모든 판정이 무의미',
        gain='±12 를 실측해 "개선" 기록 다수를 무효화',
        evidence='동일 설정에서 시드만 바꿔 재실행한 변동폭',
        requires=[], note='⚠️ 시드 노이즈다. 결정적 사후처리 변경엔 0 이므로 그대로 쓰면 안 된다.')
def noise_floor(score_fn, seeds, **kw):
    s = np.array([score_fn(seed=sd, **kw) for sd in seeds], float)
    return dict(mean=float(s.mean()), sd=float(s.std(ddof=1)),
                floor=float(2 * s.std(ddof=1)), scores=s.tolist())


@method(id='val.random_controls', stage=4, status='ADOPTED', cost='low',
        title='무작위 대조군 — 대량 탐색의 필수품',
        gain='제3축·타겟 스윕에서 "발견"이 전부 대조군 대역임을 밝혀냄',
        evidence='실후보 최고 +0.49 vs 무작위 최고 +0.94 → 즉시 기각',
        requires=[], note='★★ 후보를 10개 이상 훑는 모든 탐색에 필수.')
def with_random_controls(candidates: dict, n_random, n_rows, rng=None, kind='normal'):
    rng = rng or np.random.default_rng(0)
    out = dict(candidates)
    for i in range(n_random):
        out[f'RANDOM:{i:02d}'] = (rng.standard_normal(n_rows) if kind == 'normal'
                                  else rng.integers(0, 3, n_rows))
    return out


def judge_with_controls(results: pd.DataFrame, score_col='gain', name_col='name',
                        min_gain=12.0):
    r = results.copy()
    r['rnd'] = r[name_col].astype(str).str.startswith('RANDOM')
    floor = float(r.loc[r.rnd, score_col].max()) if r.rnd.any() else 0.0
    thr = max(min_gain, floor)
    r['pass'] = (~r.rnd) & (r[score_col] > thr)
    return r.sort_values(score_col, ascending=False), floor, thr


@method(id='val.paired_bootstrap', stage=4, status='ADOPTED', cost='low',
        title='페어드 부트스트랩',
        gain='작은 차이의 신뢰구간 판정',
        evidence='두 예측의 차이를 부트스트랩 — 독립 부트스트랩보다 훨씬 좁다',
        requires=[], note='REAL 판정(구간이 0 을 넘지 않음)의 근거.')
def paired_bootstrap(p_a, p_b, y, metric, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); n_rows = len(y)
    a, b, yy = map(lambda v: np.asarray(v, float), (p_a, p_b, y))
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        d[i] = metric(b[idx], yy[idx]) - metric(a[idx], yy[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), real=bool(lo > 0))


@method(id='val.rho_screen', stage=7, status='ADOPTED', cost='med',
        title='상한 스크린 — 학습 전에 천장을 잰다',
        gain='트랙맨 물리 블록 ρ 0.00% 를 밝혀 GPU 낭비를 막음',
        evidence='정보원으로 잔차를 직접 회귀하면 그 위의 어떤 모델도 그 값을 못 넘는다',
        requires=[], note='★★ 새 arm·피처블록 제안을 받으면 가장 먼저.')
def rho_screen(X, residual, folds, model_factory, metric, base_pred, y_eval, C=1e5):
    """folds = (train_idx, beta_idx, eval_idx) — 3분할 중첩으로 누출 차단."""
    tr, bi, ev = folds
    m = model_factory(); m.fit(X[tr], residual[tr])
    hb = m.predict(X[bi]); rb = residual[bi]
    beta = float(np.dot(hb, rb) / max(np.dot(hb, hb), 1e-12))
    he = m.predict(X[ev])
    base = metric(base_pred, y_eval)
    gain = metric(np.clip(base_pred + beta * he, 1e-6, 1 - 1e-6), y_eval) - base
    return dict(gain=float(gain), beta=beta,
                rho=float(np.sqrt(max(gain, 0) / (C - base))))


@method(id='val.transfer_ratio', stage=9, status='ADOPTED', cost='low',
        title='전이비 추적 (로컬 → 실전)',
        gain='축마다 0.16~1.0 으로 다름을 실측. 상수 하나로 환산하면 틀린다',
        evidence='엔티티 룩업 0.76 / 다른 파이프라인 ~1.0 / 내부 가중치 ~0.16',
        requires=[], note='제출할 때마다 (로컬Δ, 실전Δ) 쌍을 기록하라.')
def transfer_ratio(pairs):
    p = np.array(pairs, float)
    m = p[:, 0] != 0
    return dict(n=int(m.sum()), ratio=float(np.median(p[m, 1] / p[m, 0])),
                pairs=p.tolist())


@method(id='val.nested_cv', stage=4, status='ADOPTED', cost='high',
        title='중첩 검증',
        gain='inner ±11.98 / outer ±36.87 — outer 가 3배',
        evidence='튜닝은 inner 에서만, 최종 검증은 outer 에서 한 번만',
        requires=[], note='inner 에서 잡은 개선의 상당수가 outer 에서 사라진다.')
def nested_note():
    return ('튜닝 폴드와 평가 폴드를 분리하라. 같은 폴드로 튜닝하고 평가한 점수는 '
            '낙관 편향이며, 이 대회에서 그 편향은 3배였다.')
