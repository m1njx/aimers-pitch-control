"""rejected.py — 실패·보류한 기법들. **코드로 남기는 이유**가 있다.

지운 기법은 반드시 누군가 다시 제안한다. 코드와 근거를 함께 두면
"이미 해봤고 이런 결과였다"를 5초 만에 보여줄 수 있다.
그리고 **다른 대회에서는 유효할 수 있다** — 실패는 이 데이터에서의 실패다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import method

EPS = 1e-6


@method(id='rej.batter_symmetric_lookup', stage=6, status='REJECTED', cost='low',
        title='대칭 엔티티 룩업 (상대편 엔티티로 같은 표를 만든다)',
        gain='2024 전이 −0.43, **누출 오라클조차 +0.06**',
        evidence='여지 자체가 없음이 증명됨. 재시도 금지',
        requires=['entity_cols'],
        note='해석: 타겟이 한쪽 엔티티가 통제하는 결과라 비대칭이 실재한다. '
             '새 대회에서 양쪽이 대등하게 결과를 통제한다면 다시 볼 것.')
def batter_symmetric_lookup(*a, **k):
    from .lookups import entity_residual_lookup
    return entity_residual_lookup(*a, **k)          # 구현은 동일, 엔티티만 교체


@method(id='rej.third_axis_context', stage=6, status='REJECTED', cost='med',
        title='제3축을 문맥 변수로 교체 (아웃/주자/이닝/리버리지/점수차/월/표리)',
        gain='11후보 최고 +0.49 < 무작위 대조군 +0.94',
        evidence='문맥 변수는 엔티티와 안정적 상호작용을 만들지 않는다',
        requires=['context_cols'],
        note='⚠️ 한 축(상대의 좌우)만 특별했다. 새 대회에서 "플래툰"에 해당하는 '
             '매치업 성격의 축이 있으면 그것부터 시도하라.')
def third_axis_context(*a, **k):
    from .lookups import axis_sweep
    return axis_sweep(*a, **k)


@method(id='rej.alt_targets', stage=6, status='REJECTED', cost='low',
        title='룩업 타겟을 다른 결과 차원으로 교체',
        gain='7후보 최고 +1.89 < 무작위 대조군 +2.44',
        evidence='복원한 ball/strike 차원은 추가 정보가 없었다',
        requires=['cumulative_prefixes'],
        note='단 하나(reverse−middle 대비)만 먹혔다(+11.00). 차원을 늘린다고 되는 게 아니다.')
def alt_targets(decoded: pd.DataFrame, combos: list):
    """decoded 의 컬럼들로 만들 수 있는 타겟 후보를 생성한다."""
    out = {}
    for a, b in combos:
        out[f'{a}-{b}'] = decoded[a].to_numpy(float) - decoded[b].to_numpy(float)
    for c in decoded.columns:
        out[c] = decoded[c].to_numpy(float)
    return out


@method(id='rej.residual_posthoc_model', stage=9, status='REJECTED', cost='med',
        title='잔차 후처리 모델 (GBDT 로 체인 잔차를 예측해 더한다)',
        gain='정직한 3분할에서 +6.65 → **실전 LB −4.1**',
        evidence='다른 시점에서 −22.35, β 부호가 폴드마다 뒤집힘, 시드 하나로 +6.65→+3.21',
        requires=[],
        note='⚠️⚠️ 이 대회 최대 교훈. **정직한 홀드아웃 통과만으로는 부족하다.** '
             '자유도가 큰 모델(400트리×63리프)은 연도별 오차를 외운다. '
             '엔티티 룩업이 살아남은 이유와 대조하라 — 그쪽은 EB+게이트로 자유도가 극히 낮다.')
def residual_posthoc_model(X_tr, r_tr, X_beta, r_beta, X_ev, p_ev, model_factory):
    m = model_factory(); m.fit(X_tr, r_tr)
    hb = m.predict(X_beta)
    beta = float(np.dot(hb, r_beta) / max(np.dot(hb, hb), 1e-12))
    return np.clip(p_ev + beta * m.predict(X_ev), EPS, 1 - EPS), beta


@method(id='rej.recency_weighting', stage=5, status='REJECTED', cost='low',
        title='최근성 재가중 / EWMA 피처',
        gain='단조 하락. EWMA 최근성 이득(+98)은 **100% 경기내 리키지**',
        evidence='같은 경기 안의 미래 정보가 새어 들어갔다',
        requires=['time_col'],
        note='⚠️ 이득이 크게 나오면 리키지부터 의심하라. 사건 경계(경기·세션)를 넘지 않는지 확인.')
def recency_weighting(df, time_col, half_life):
    t = pd.to_numeric(df[time_col], errors='coerce').to_numpy(float)
    return np.power(0.5, (t.max() - t) / max(half_life, 1e-9))


@method(id='rej.entity_embedding', stage=5, status='REJECTED', cost='high',
        title='엔티티 임베딩 (학습된 벡터를 피처로)',
        gain='5시드 재검증에서 +12.0 → +3.1 로 붕괴',
        evidence='이웃 구성 전부 baseline 미만',
        requires=['entity_cols'],
        note='초기 소수 시드에서 크게 나왔던 것이 시드를 늘리자 사라졌다 — '
             '시드 규율의 필요성을 보여준 사례.')
def entity_embedding_note():
    return '임베딩 자체보다 EB 수축 룩업이 같은 정보를 더 안정적으로 준다.'


@method(id='rej.knowledge_distillation', stage=2, status='REJECTED', cost='high',
        title='지식 증류',
        gain='사전기준 통과(+85.2, t=6.38)했으나 **정보항 −83.4**',
        evidence='이득의 100% 가 분산 수축이었다',
        requires=[],
        note='⚠️ 그리고 교사가 특권 피처를 쓰고 있었다. 반드시 '
             'cal.shrinkage_decomposition 으로 분해하고 교사 피처를 확인하라.')
def distill_targets(teacher_pred, y, alpha=0.5):
    return alpha * np.asarray(teacher_pred, float) + (1 - alpha) * np.asarray(y, float)


@method(id='rej.deep_tabular_variety', stage=2, status='REJECTED', cost='high',
        title='신경망 종류 늘리기 (ResNet / Transformer / TabNet / TabM)',
        gain='ResNet+Transformer −24, +TabNet −39, TabM 블렌딩 −70.36',
        evidence='outer fold 일반화가 구조적으로 약하다',
        requires=[],
        note='⚠️ MLP **하나**는 크게 성공했다(+36.24). 종류를 늘리는 것이 실패한다.')
def deep_variety_note():
    return 'MLP 한 종만 블렌드 멤버로. 종류 추가는 이 대회에서 일관되게 손해였다.'


@method(id='rej.prior_shift', stage=9, status='REJECTED', cost='low',
        title='사전확률 시프트 보정',
        gain='오라클조차 음수',
        evidence='테스트 라벨률 추정으로 전체를 이동',
        requires=[], note='라벨률이 안정적이면 얻을 게 없다.')
def prior_shift(p, target_rate):
    pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    cur = pc.mean()
    z = np.log(pc / (1 - pc)) + (np.log(target_rate / (1 - target_rate))
                                 - np.log(cur / (1 - cur)))
    return np.clip(1 / (1 + np.exp(-z)), EPS, 1 - EPS)


@method(id='rej.segment_drift', stage=5, status='REJECTED', cost='low',
        title='세그먼트 드리프트 보정',
        gain='12/12 전방검증 음수',
        evidence='세그먼트별 시간 추세를 외삽해 보정',
        requires=['time_col'], note='추세가 다음 기간에 이어지지 않았다.')
def segment_drift(df, seg_col, time_col, target_col, train_mask):
    g = df[train_mask].groupby([seg_col, time_col])[target_col].mean().reset_index()
    return g.groupby(seg_col).apply(
        lambda x: np.polyfit(x[time_col], x[target_col], 1)[0] if len(x) > 2 else 0.0)


@method(id='rej.matchup_prior', stage=6, status='REJECTED', cost='low',
        title='엔티티×엔티티 직접 매치업 사전확률',
        gain='전방검증 −12.1 (단독 측정은 +258.6 이었다)',
        evidence='단독 측정과 전방 전이가 정반대. 4건 전부 전이 실패',
        requires=['entity_cols'],
        note='⚠️ 매치업 표본이 희소해 수축해도 남는 게 없다. '
             '"단독 측정치"를 이득 추정으로 인용하지 마라.')
def matchup_prior(train, e1, e2, y, K=200.0):
    D = pd.DataFrame(dict(a=train[e1].to_numpy(), b=train[e2].to_numpy(),
                          y=np.asarray(y, float)))
    g = D.groupby(['a', 'b']).y.agg(['sum', 'count'])
    return ((g['sum'] + K * D.y.mean()) / (g['count'] + K)).rename('mprior')


@method(id='rej.joint_refit', stage=9, status='REJECTED', cost='low',
        title='사후보정 계수 동시 재적합 (순차 대신)',
        gain='순방향 폴드 +0.05, 평균 +1.32 (기준 +1.6 미달)',
        evidence='이중계상 손실은 실재하나 미미하다',
        requires=[],
        note='⏸ 보정 단계가 5개 이상으로 늘면 다시 볼 가치가 있다.')
def joint_refit(p, y, X_terms):
    w = np.clip(p * (1 - p), 1e-9, None)
    return np.linalg.lstsq(np.stack([x * w for x in X_terms], 1), y - p, rcond=None)[0]


# ── 보류(SHELVED): 원리상 가능하나 이 대회에선 여건이 안 됐다 ──────────
@method(id='shelf.conditional_arm_weight', stage=8, status='SHELVED', cost='med',
        title='arm 가중치를 구간별로 세분화',
        gain='미측정 — 필요한 홀드아웃 예측을 확보하지 못했다',
        evidence='전역 상수 W_C=0.10 이 구간별로 다를 수 있다',
        requires=[],
        note='⏸ 다음에는 **판정용 홀드아웃 예측을 접수 규격으로 요구**하면 바로 잴 수 있다.')
def conditional_arm_weight(pA, pB, pC, segment, weights: dict, w_default=0.10):
    s = np.asarray(segment)
    wc = np.full(len(s), w_default, float)
    for k, v in weights.items():
        wc[s == k] = v
    ab = np.asarray(pA, float) * 0.55 + np.asarray(pB, float) * 0.45
    return (1 - wc) * ab + wc * np.asarray(pC, float)


@method(id='shelf.coverage_threshold', stage=7, status='SHELVED', cost='med',
        title='상태 복원의 신뢰도 임계 완화 (커버리지 확대)',
        gain='미측정 — 필요한 피처 캐시(924MB)가 없었다',
        evidence='정밀도 ≥0.77 구간만 보정 중. 0.70~0.75 로 넓히면?',
        requires=[],
        note='⏸ 임계를 낮추면 커버리지가 늘지만 오분류도 는다. 폴드에서 곡선을 그려라.')
def coverage_threshold_sweep(precision_by_bin: dict, thresholds=(0.70, 0.73, 0.75, 0.77, 0.80)):
    return {t: [b for b, p in precision_by_bin.items() if p >= t] for t in thresholds}


@method(id='shelf.isotonic_time_stable', stage=9, status='SHELVED', cost='low',
        title='시간 안정 Isotonic',
        gain='미시도',
        evidence='자유도가 커서 시간 외삽에 위험하다고 판단해 보류',
        requires=[], note='⏸ 랜덤 분할 대회라면 우선순위를 올려라.')
def shelved_isotonic_note():
    return '시간 축이 없으면 Isotonic 이 아핀보다 낫다. 이 대회는 시간 외삽이 핵심이라 보류했다.'
