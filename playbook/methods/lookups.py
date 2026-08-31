"""lookups.py — 엔티티 룩업 계열.

[기여 구분] 이 계열 기법은 **팀 공동 파트에서 도입**됐다. 여기 담은 것은
그 레시피의 정리와, 내가 수행한 축·타겟 스윕(전부 기각)의 구현이다.
검증 기록: src/analysis/{verify_pcxh, axis3_sweep, target_sweep, bcxh_test}.py
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from ._base import method

EPS = 1e-6


def _logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.where(z != 0., 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z))), p)


def _skill(p, y, C=1e5):
    r = y.mean()
    return C * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


# ══════════════════════════════════════════════════════════════════════
@method(id='lookup.entity_residual', stage=6, status='ADOPTED', cost='low',
        title='이중 중심화 엔티티 잔차 룩업 (pcxh)',
        gain='2024 정직측정 +19.70 → LB +15.21',
        evidence='엔티티×셀×제3축 상호작용만 남긴 표를 사후 로짓 시프트로 적용. 5요소 필수',
        requires=['entity_cols', 'context_cols'],
        note='이 대회 단일 기법 최대 이득. 새 대회에서 베이스 모델 직후 최우선 시도.')
def entity_residual_lookup(train: pd.DataFrame, entity: str, cell: np.ndarray,
                           bucket: np.ndarray, third: np.ndarray, y: np.ndarray,
                           K: float = 100.0):
    """dev = EB[r(e,cell,third)] − r(e,bucket) − [리그 r(cell,third) − 리그 r(bucket)]

    5요소 — 하나라도 빠지면 이득이 0 이 된다:
      1 이중 중심화  2 제3축  3 EB 수축  4 관측량 게이트  5 β 는 다른 폴드에서 적합 ×0.8
    반환: (cell표, coarse표) — `apply_lookup` 에 넣는다.
    """
    e = train[entity].to_numpy()
    D = pd.DataFrame(dict(e=e, cell=cell, bk=bucket, t=third, y=y))
    lg_all = D.y.mean()
    lg_ct = D.groupby(['cell', 't']).y.mean()
    lg_bk = D.groupby('bk').y.mean()
    lg_t = D.groupby('t').y.mean()

    gp = D.groupby(['e', 'bk']).y.agg(['sum', 'count'])
    parent = (gp['sum'] + K * lg_bk.reindex(gp.index.get_level_values('bk')).to_numpy()) \
        / (gp['count'] + K)
    g = D.groupby(['e', 'cell', 't', 'bk']).y.agg(['sum', 'count']).reset_index()
    par = parent.reindex(pd.MultiIndex.from_arrays([g['e'], g['bk']])).to_numpy()
    r_cell = (g['sum'].to_numpy() + K * par) / (g['count'].to_numpy() + K)
    g['dev'] = (r_cell - par) - (
        lg_ct.reindex(pd.MultiIndex.from_arrays([g['cell'], g['t']])).to_numpy()
        - lg_bk.reindex(g['bk']).to_numpy())

    ge = D.groupby('e').y.agg(['sum', 'count'])
    r_e = (ge['sum'] + K * lg_all) / (ge['count'] + K)
    g2 = D.groupby(['e', 't']).y.agg(['sum', 'count']).reset_index()
    p2 = r_e.reindex(g2['e']).to_numpy()
    r_t = (g2['sum'].to_numpy() + K * p2) / (g2['count'].to_numpy() + K)
    g2['dev'] = (r_t - p2) - (lg_t.reindex(g2['t']).to_numpy() - lg_all)

    return g.set_index(['e', 'cell', 't'])['dev'], g2.set_index(['e', 't'])['dev']


def apply_lookup(df, entity, cell, third, tables, betas, count, n_min):
    """행 자신의 키로만 조회한다 → 규정 안전(행 독립)."""
    T1, T2 = tables
    e = df[entity].to_numpy()
    x1 = np.nan_to_num(T1.reindex(pd.MultiIndex.from_arrays([e, cell, third])).to_numpy(float))
    x2 = np.nan_to_num(T2.reindex(pd.MultiIndex.from_arrays([e, third])).to_numpy(float))
    gate = (np.asarray(count, float) >= n_min).astype(float)
    return gate * (betas[0] * x1 + betas[1] * x2)


def fit_betas(p, y, x1, x2, shrink=0.8):
    """β 는 **적용할 폴드가 아닌 다른 폴드**에서 적합하고 ×0.8 수축한다."""
    w = np.clip(p * (1 - p), 1e-9, None)
    b = np.linalg.lstsq(np.stack([x1 * w, x2 * w], 1), y - p, rcond=None)[0]
    return b * shrink


# ══════════════════════════════════════════════════════════════════════
@method(id='lookup.contrast_target', stage=6, status='ADOPTED', cost='low',
        title='대비 타겟 룩업 (ctr)',
        gain='+11.00 → LB +5.47',
        evidence='같은 키에 타겟만 교체(결과 A − 결과 B 대비). 원 타겟과 다른 정보를 싣는다',
        requires=['entity_cols', 'cumulative_prefixes'],
        note='디코딩(3.5)으로 얻은 보조 라벨이 있을 때만 가능')
def contrast_target_lookup(train, entity, cell, bucket, third, target_a, target_b, K=200.0):
    """타겟 = (결과 A − 결과 B). 나머지는 entity_residual_lookup 과 동일."""
    t = np.asarray(target_a, float) - np.asarray(target_b, float)
    m = ~np.isnan(t)
    return entity_residual_lookup(train[m], entity, cell[m], bucket[m], third[m], t[m], K)


# ══════════════════════════════════════════════════════════════════════
@method(id='lookup.target_encoding_eb', stage=6, status='ADOPTED', cost='low',
        title='EB 수축 타겟 인코딩',
        gain='라벨조건부 룩업 계열로 +52 ~ +76',
        evidence='고카디널리티 엔티티를 과거 라벨 평균으로 인코딩, EB 로 수축',
        requires=['entity_cols'],
        note='⚠️ 수축 없이 쓰면 재앙 — 무수축 버전이 LB 103(정상 1032)을 냈다')
def target_encoding_eb(train, entity, y, K=100.0, parent_rate=None):
    e = train[entity].to_numpy()
    g = pd.DataFrame(dict(e=e, y=np.asarray(y, float))).groupby('e').y.agg(['sum', 'count'])
    pr = float(np.mean(y)) if parent_rate is None else parent_rate
    return ((g['sum'] + K * pr) / (g['count'] + K)).rename('te')


# ══════════════════════════════════════════════════════════════════════
@method(id='lookup.axis_sweep', stage=6, status='ADOPTED', cost='med',
        title='제3축 스윕 (무작위 대조군 포함)',
        gain='이 대회에선 통과 0 — 그러나 **판정 절차 자체가 자산**',
        evidence='11후보 최고 +0.49 < 무작위 대조군 최고 +0.94 → 즉시 기각',
        requires=['context_cols'],
        note='제3축이 무엇이어야 하는지는 재봐야 안다. 반드시 대조군과 함께.')
def axis_sweep(fold_fit, fold_eval, entity, build_fn, apply_fn, axes: dict,
               n_random=20, seed=0, threshold=12.0):
    """축 후보 여러 개를 훑되 **무작위 축을 섞어** 위양성 바닥을 함께 잰다."""
    rng = np.random.default_rng(seed)
    cand = dict(axes)
    for i in range(n_random):
        cand[f'RANDOM:{i:02d}'] = rng.integers(0, 3, len(fold_eval['df']))
    rows = []
    for name, ax in cand.items():
        try:
            gain = apply_fn(fold_fit, fold_eval, entity, ax, build_fn)
        except Exception as ex:
            rows.append(dict(name=name, gain=np.nan, err=str(ex)[:60],
                             rnd=name.startswith('RANDOM'))); continue
        rows.append(dict(name=name, gain=gain, err='', rnd=name.startswith('RANDOM')))
    R = pd.DataFrame(rows)
    floor = R[R.rnd].gain.max() if R.rnd.any() else 0.0
    R['pass'] = (~R.rnd) & (R.gain > max(threshold, floor))
    return R.sort_values('gain', ascending=False), float(floor)
