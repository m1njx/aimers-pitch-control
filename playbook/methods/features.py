"""features.py — 피처 엔지니어링 계열."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._base import method


@method(id='feat.asof_decompose', stage=5, status='ADOPTED', cost='low',
        title='as-of 누적 피처 분해',
        gain='LB +146.80 — 이 대회 단일 변경 최대 이득',
        evidence='누적률을 장기실력/최근폼/관측량/변동성으로 분해해 46개 추가',
        requires=['cumulative_prefixes'],
        note='주최자가 준 as-of 컬럼은 누출 걱정이 없다. 그대로 넣지 말고 반드시 분해할 것.')
def asof_decompose(df: pd.DataFrame, base_col: str, recent_cols: list, n_col: str):
    """원시 누적값 대신 차분·비율·신뢰도를 명시적으로 만든다."""
    out = pd.DataFrame(index=df.index)
    base = pd.to_numeric(df[base_col], errors='coerce')
    out[f'{base_col}__level'] = base
    out[f'{base_col}__volume'] = np.log1p(pd.to_numeric(df[n_col], errors='coerce').fillna(0))
    for c in recent_cols:
        r = pd.to_numeric(df[c], errors='coerce')
        out[f'{c}__dev'] = r - base                      # 최근 폼 편차
    if len(recent_cols) >= 2:
        a = pd.to_numeric(df[recent_cols[0]], errors='coerce')
        b = pd.to_numeric(df[recent_cols[-1]], errors='coerce')
        out[f'{base_col}__volatility'] = (a - b).abs()   # 변동성
    return out


@method(id='feat.era_relative', stage=5, status='REJECTED', cost='low',
        title='시즌 상대화 (era normalization)',
        gain='단조 하락 → 확정 REJECT',
        evidence='누적 피처의 시즌별 수준 밀림을 보정. 이 대회에선 이득 없음',
        requires=['time_col'],
        note='⚠️ 기법 자체는 유효하다 — 팀 C arm 은 이걸 쓰고 +7.56 을 냈다. '
             '구현·맥락이 결과를 가른다. 새 대회에서 분포 밀림이 크면 다시 볼 것.')
def era_relative(df, cols, time_col, train_mask):
    """시즌 평균은 **학습 프레임에서만** 계산해 표로 저장, 미지 시즌은 마지막 값 대체.
    → 행 자신의 time_col 만 보는 lookup 이라 행 독립."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce')
        tbl = v[train_mask].groupby(df.loc[train_mask, time_col]).mean()
        last = tbl.iloc[-1] if len(tbl) else 0.0
        out[f'{c}__rel'] = v - df[time_col].map(tbl).fillna(last)
    return out


@method(id='feat.decode_hidden_labels', stage=7, status='ADOPTED', cost='low',
        title='누적 통계 차분으로 숨은 라벨 복원',
        gain='5개 결과 차원 확보 (검산 99.95% 일치)',
        evidence='outcome = rate(다음행)·n(다음행) − rate(이번행)·n(이번행)',
        requires=['cumulative_prefixes', 'entity_cols'],
        note='★ 재사용 가치 최상. 누적 컬럼이 있으면 비용 거의 0 으로 라벨이 늘어난다.')
def decode_hidden_labels(df, entity_col, n_col, rate_cols, order_cols, verify_col=None):
    """연속된 동일 엔티티 행의 차분으로 각 행의 개별 결과를 복원한다.
    반드시 **알려진 라벨 차원으로 검산**하라 (verify_col)."""
    x = df.copy(); x['_o'] = np.arange(len(x))
    x = x.sort_values(list(order_cols) + [n_col, '_o'])
    g = x.groupby(list(order_cols), sort=False)
    n0 = pd.to_numeric(x[n_col], errors='coerce').to_numpy(float)
    n1 = g[n_col].shift(-1).to_numpy(float)
    ok = (n1 - n0) == 1
    out = {}
    for c in rate_cols:
        r0 = pd.to_numeric(x[c], errors='coerce').to_numpy(float)
        v = np.round(g[c].shift(-1).to_numpy(float) * n1 - r0 * n0)
        out[c] = np.where(ok & np.isin(v, [0., 1.]), v, np.nan)
    R = pd.DataFrame(out, index=x.index).reindex(df.index)
    acc = None
    if verify_col and verify_col in R.columns:
        m = R[verify_col].notna()
        acc = float((R.loc[m, verify_col] == df.loc[m, verify_col]).mean())
    return R, acc, float(np.mean(ok))


@method(id='feat.state_reconstruction', stage=7, status='ADOPTED', cost='high',
        title='상태 복원 (창 대수)',
        gain='LB +17.09 (+6.06 확장)',
        evidence='시즌누적 − 직전경기누적 = 현재 진행분. 유일해가 안 나오면 밴드 탐색',
        requires=['entity_cols', 'cumulative_prefixes'],
        note='대수가 복잡하고 커버리지가 제한적이지만 아무도 안 쓰는 정보가 나온다.')
def state_reconstruction(df, total_n, anchor_n, recent_n_list, total_s, anchor_s, recent_s_list):
    """반환: (진행 인덱스 j, 진행 성공수 s, 복원가능 마스크)"""
    tn = pd.to_numeric(df[total_n], errors='coerce').to_numpy(float)
    an = pd.to_numeric(df[anchor_n], errors='coerce').to_numpy(float)
    rn = sum(pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy(float) for c in recent_n_list)
    ts = pd.to_numeric(df[total_s], errors='coerce').to_numpy(float)
    a_s = pd.to_numeric(df[anchor_s], errors='coerce').to_numpy(float)
    rs = sum(pd.to_numeric(df[c], errors='coerce').fillna(0).to_numpy(float) for c in recent_s_list)
    j = tn - (an + rn); s = ts - (a_s + rs)
    ok = np.isfinite(j) & np.isfinite(s) & (j >= 0) & (s >= 0) & (s <= j)
    return j, s, ok


def state_correction(p, j, s, ok, b=1.2, K=20.0):
    """q = sigmoid(logit p + b·(s − j·p)/(j + K)) — 복원된 행에만 적용."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc)) + np.where(ok, b * (s - j * pc) / (j + K), 0.0)
    return np.where(ok, 1. / (1. + np.exp(-z)), p)


@method(id='feat.domain_physics', stage=5, status='REJECTED', cost='med',
        title='도메인 물리 파생 피처',
        gain='3종 +1.37 / 9종 −361.17 / arm 화 시 ρ 0.00%',
        evidence='대부분 조인키의 결정함수라 정보량이 0 이었다',
        requires=[],
        note='⚠️ 만들기 전에 반드시 `check_determinism` 을 돌려라(아래).')
def check_determinism(df, new_feature: str, join_keys: list) -> float:
    """새 파생 피처가 기존 조인키의 결정함수인가? 반환값 0 이면 **정보량 0**."""
    return float(df.groupby(join_keys)[new_feature].std().max())


# ─────────────────────────────────────────────────────────────
# 아래 두 기법은 A arm 에 **출하됐지만 단독 이득을 분리 측정하지 않았다.**
# 정직하게 그렇게 적어 둔다 — "썼다" 와 "효과를 쟀다" 는 다르다.
# ─────────────────────────────────────────────────────────────

@method(id='feat.cfa_latent', stage=5, status='ADOPTED', cost='low',
        title='이론 지정 잠재변수 (확인적 요인분석, CFA)',
        gain='A arm 에 잠재변수 4개로 출하. ⚠️ 단독 이득은 분리 측정하지 않았다',
        evidence='탐색적 PCA 와 달리 **어떤 관측변수가 어떤 요인에 실리는지 사전 지정**한다. '
                 '요인이 도메인 개념(투수 역량·경기 압박·투구 물리·카운트 상태)에 대응해 해석 가능',
        requires=['context_cols'],
        note='⚠️ 이 대회에서 A/B 로 분리 측정하지 않은 채 출하했다. 다음엔 반드시 단독으로 재라 — '
             '측정 안 한 피처는 "효과가 있었다" 고 말할 수 없다. '
             'GBDT 는 원 피처에서 상호작용을 스스로 찾으므로 요인 축약의 순이득은 작을 수 있다.')
def cfa_latent(df, spec: dict, n_iter=1000, random_state=0):
    """spec = {'요인명': [관측변수, ...]} — 요인마다 1-factor 모델을 적합해 점수를 만든다.

    train 에서 fit 한 스케일러·요인모델을 그대로 test 에 적용해야 행 독립이 유지된다.
    반환: (요인점수 DataFrame, 적합된 객체 dict)
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FactorAnalysis

    scores, fitted = {}, {}
    for name, cols in spec.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        Z = df[cols].astype(float)
        Z = Z.fillna(Z.median())
        sc = StandardScaler().fit(Z)
        fa = FactorAnalysis(n_components=1, max_iter=n_iter,
                            random_state=random_state).fit(sc.transform(Z))
        scores[name] = fa.transform(sc.transform(Z))[:, 0]
        fitted[name] = (cols, sc, fa)
    return pd.DataFrame(scores, index=df.index), fitted


@method(id='feat.interaction_by_theory', stage=5, status='ADOPTED', cost='low',
        title='이론이 지정한 상호작용 항 (곱 피처)',
        gain='A arm 에 출하. ⚠️ 단독 이득은 분리 측정하지 않았다',
        evidence='트리는 상호작용을 분할로 근사하지만 **연속적인 곱**은 잘 못 만든다. '
                 '도메인이 "이 셋이 동시에 성립할 때"를 지목해 주면 곱으로 넣어 준다',
        requires=['context_cols'],
        note='이 대회에선 게임이론 관점의 곱 2종을 넣었다 — '
             '(압박 × 접전 × 투수능력), (2아웃 × 득점권 × 레버리지 × 투수능력). '
             '⚠️ 곱 피처는 값이 커지기 쉽고 결측 전파가 빠르다. NaN 을 명시적으로 채울 것.')
def interaction_by_theory(df, terms: dict, fill=0.5):
    """terms = {'새피처명': [컬럼 또는 (컬럼, 변환함수), ...]} — 지정한 항들의 곱.

    예: {'gt_pressure_perf': [('li', np.log1p), 'closeness', 'pitcher_success']}
    """
    import numpy as np
    import pandas as pd
    out = {}
    for name, parts in terms.items():
        v = np.ones(len(df))
        for p in parts:
            col, fn = p if isinstance(p, tuple) else (p, None)
            if col not in df.columns:
                v = None
                break
            x = df[col].astype(float).fillna(fill).to_numpy()
            v = v * (fn(x) if fn else x)
        if v is not None:
            out[name] = v
    return pd.DataFrame(out, index=df.index)
