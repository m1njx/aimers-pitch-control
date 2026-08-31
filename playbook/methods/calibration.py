"""calibration.py — 캘리브레이션·사후보정 계열."""
from __future__ import annotations
import numpy as np
from ._base import method

EPS = 1e-6


@method(id='cal.affine', stage=9, status='ADOPTED', cost='low',
        title='아핀 보정 (SCALE / SHIFT)',
        gain='LB +12.90 (1016.13)',
        evidence='p = 0.5 + SCALE·(p−0.5) + SHIFT. 파라미터 2개라 과적합 없음',
        requires=[],
        note='⚠️ 최적점이 좁다. SCALE 1.10→1.15 는 −7.41, SHIFT −0.0045→−0.01 은 −27.35. '
             '외삽 금지. LB 로 직접 최적화하는 것이 가장 정확하다.')
def affine(p, scale=1.10, shift=0.0):
    return np.clip(0.5 + scale * (np.asarray(p, float) - 0.5) + shift, EPS, 1 - EPS)


@method(id='cal.logit_cell_offset', stage=9, status='ADOPTED', cost='low',
        title='로짓 셀 오프셋',
        gain='채택 (U=0.35). U 확대 시도는 −0.47 / −1.69',
        evidence='문맥 셀마다 로짓에 상수를 더하되 전역 스케일 U 로 강도 통제',
        requires=['context_cols'],
        note='U 하나로 자유도를 묶는 것이 핵심. 오프셋은 train 에서만 적합.')
def logit_cell_offset(p, cell_codes, offsets: dict, U=0.35):
    pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = np.log(pc / (1 - pc))
    c = np.asarray(cell_codes)
    for k, v in offsets.items():
        z[c == k] += U * v
    return np.clip(1. / (1. + np.exp(-z)), EPS, 1 - EPS)


def fit_cell_offsets(p, y, cell_codes, min_n=200):
    """train 폴드에서 셀별 로짓 잔차 평균을 구한다."""
    pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = np.log(pc / (1 - pc)); c = np.asarray(cell_codes); out = {}
    for k in np.unique(c):
        m = c == k
        if m.sum() < min_n:
            continue
        yk = np.asarray(y, float)[m].mean()
        yk = min(max(yk, EPS), 1 - EPS)
        out[int(k)] = float(np.log(yk / (1 - yk)) - z[m].mean())
    return out


@method(id='cal.lb_quadratic_fit', stage=9, status='ADOPTED', cost='low',
        title='LB 2~3점으로 계수 최적값 역산',
        gain='b* 를 로컬로는 못 찾을 값으로 특정 (+17.09 획득에 기여)',
        evidence='한 계수만 바꾼 제출 2~3회 → 2차식 적합 → 꼭짓점',
        requires=[],
        note='★ 최고점 기준 대회면 하방이 0 이라 거의 공짜. 반드시 **한 번에 하나만** 바꿀 것.')
def lb_quadratic_fit(points):
    """points = [(계수값, LB점수), ...] 최소 3점. 반환: (최적계수, 예상최대)"""
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    if len(x) < 3:
        raise ValueError('최소 3점 필요 (2점이면 방향만 알 수 있다)')
    a, b, c = np.polyfit(x, y, 2)
    if a >= 0:
        return None, '위로 볼록이 아님 — 최적점이 구간 밖'
    xs = -b / (2 * a)
    return float(xs), float(a * xs * xs + b * xs + c)


@method(id='cal.shrinkage_decomposition', stage=9, status='ADOPTED', cost='low',
        title='분산 수축 분해 — 이득이 정보인가 수축인가',
        gain='증류 축의 이득 +85.2 중 정보항 −83.4 를 밝혀 종결',
        evidence='Brier 는 분산만 줄여도 좋아진다. 반드시 분해할 것',
        requires=[],
        note='★ 새 후보의 이득을 보고할 때 항상 함께 보고하라.')
def shrinkage_decomposition(p_base, p_cand, y, s_opt=None, metric=None):
    """반환: (전체이득, 수축으로 설명되는 몫, 순수 정보 몫)"""
    from .ensemble import brier_skill
    metric = metric or brier_skill
    base = metric(p_base, y)
    total = metric(p_cand, y) - base
    if s_opt is None:
        grid = np.arange(0.5, 1.51, 0.01)
        s_opt = float(grid[int(np.argmax([metric(affine(p_base, s), y) for s in grid]))])
    shrink_gain = metric(affine(p_base, s_opt), y) - base
    d_cand = np.asarray(p_cand, float) - np.asarray(p_base, float)
    d_shrink = affine(p_base, s_opt) - np.asarray(p_base, float)
    denom = float(np.dot(d_shrink, d_shrink))
    proj = d_shrink * (float(np.dot(d_cand, d_shrink)) / denom) if denom > 1e-15 else 0
    info = metric(np.clip(np.asarray(p_base, float) + (d_cand - proj), EPS, 1 - EPS), y) - base
    return float(total), float(shrink_gain), float(info)


@method(id='cal.isotonic', stage=9, status='SHELVED', cost='low',
        title='Isotonic 회귀 캘리브레이션',
        gain='미시도 — 자유도가 커서 시간 외삽에 위험하다고 판단',
        evidence='홀드아웃이 크고 시간 외삽이 없으면 유효한 표준 기법',
        requires=[],
        note='⏸ 시간 축이 없는 대회(랜덤 분할)라면 우선순위를 올려도 된다.')
def isotonic_calibrate(p_fit, y_fit, p_apply):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds='clip', y_min=EPS, y_max=1 - EPS)
    ir.fit(np.asarray(p_fit, float), np.asarray(y_fit, float))
    return np.clip(ir.predict(np.asarray(p_apply, float)), EPS, 1 - EPS)


@method(id='cal.quantile_realign', stage=9, status='REJECTED', cost='low',
        title='구간별 로짓 분위수 재정렬',
        gain='시도, 이득 없음',
        evidence='예측 분위 구간마다 로짓을 목표 분포로 정렬',
        requires=[], note='아핀 보정이 이미 잡는 것을 자유도만 늘려 반복한다.')
def quantile_realign(p_fit, y_fit, p_apply, n_bins=20):
    pf = np.asarray(p_fit, float); pa = np.asarray(p_apply, float)
    edges = np.quantile(pf, np.linspace(0, 1, n_bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    bf = np.digitize(pf, edges[1:-1]); ba = np.digitize(pa, edges[1:-1])
    out = pa.copy()
    for b in range(n_bins):
        m = bf == b
        if m.sum() < 50:
            continue
        out[ba == b] = np.clip(np.asarray(y_fit, float)[m].mean(), EPS, 1 - EPS)
    return out
