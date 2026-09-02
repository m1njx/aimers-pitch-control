"""trees.py — GBDT. 이 대회 파이프라인의 뼈대.

arm 3개가 전부 트리 위에 서 있다.

    A arm : GBDT(분류) 0.38 + LightGBM(회귀) 0.25   (+ MLP 0.42)
    B arm : CatBoost 20멤버
    C arm : LightGBM 0.275 + XGBoost 0.275          (+ 임베딩MLP 0.35)

**세 라이브러리를 다 쓴 것은 취향이 아니라 설계다.** 블렌드 이득은 단독 성능이 아니라
ρ(오차 방향의 차이)에서 나오는데, 성장 전략이 다른 라이브러리는 서로 다른 행에서 틀린다.

⚠️ 라이브러리는 함수 안에서 import 한다 — 미설치 환경에서도 카탈로그 조회는 되어야 한다.
"""
from __future__ import annotations
import numpy as np
from ._base import method


@method(id='gbdt.library_trio', stage=2, status='ADOPTED', cost='med',
        title='세 라이브러리를 각각 세우고 블렌드한다 (LightGBM · XGBoost · CatBoost)',
        gain='arm 3개가 전부 다른 라이브러리에 앵커됨. 최종 블렌드의 토대',
        evidence='성장 전략이 달라 틀리는 행이 다르다 — leaf-wise(LGBM) vs '
                 'level-wise(XGB) vs ordered boosting(CatBoost)',
        requires=['task'],
        note='★ 어느 하나가 "제일 좋은" 것을 고르려 하지 말 것. 단독 최고 모델이 '
             '블렌드 가중치 0.000 을 받은 사례가 이 대회에 실제로 있었다. '
             '고르는 게 아니라 **섞는 것**이고, 섞을 값은 ens.gram 으로 푼다.')
def library_trio(X, y, X_valid, *, task='binary', seed=0, n_estimators=1200,
                 learning_rate=0.03, num_leaves=63, which=('lgb', 'xgb', 'cat')):
    """세 라이브러리를 같은 데이터로 학습해 예측 dict 를 돌려준다.

    반환: {'lgb': pred, 'xgb': pred, 'cat': pred} — 결합은 `ens.gram` + `opt`.
    """
    X = np.asarray(X, np.float32); y = np.asarray(y, float)
    Xv = np.asarray(X_valid, np.float32)
    out = {}

    if 'lgb' in which:
        import lightgbm as lgb
        obj = 'binary' if task == 'binary' else 'regression'
        m = lgb.LGBMRegressor if obj == 'regression' else lgb.LGBMClassifier
        g = m(n_estimators=n_estimators, learning_rate=learning_rate,
              num_leaves=num_leaves, subsample=0.8, colsample_bytree=0.8,
              random_state=seed, verbose=-1).fit(X, y)
        out['lgb'] = g.predict_proba(Xv)[:, 1] if obj == 'binary' else g.predict(Xv)

    if 'xgb' in which:
        import xgboost as xgb
        g = xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic' if task == 'binary' else 'reg:squarederror',
            random_state=seed, verbosity=0).fit(X, y)
        out['xgb'] = g.predict(Xv)

    if 'cat' in which:
        # ⚠️ 분류는 Classifier, 회귀는 Regressor. Regressor 에 Logloss 를 넘기면 에러다.
        # ⚠️ cat_features 는 넘기지 않는다 — 이 대회에서 −15.49 였다 (rej.catboost_cat_features)
        if task == 'binary':
            from catboost import CatBoostClassifier
            g = CatBoostClassifier(
                iterations=n_estimators, learning_rate=learning_rate, depth=6,
                loss_function='Logloss', random_seed=seed, verbose=0).fit(X, y)
            out['cat'] = g.predict_proba(Xv)[:, 1]
        else:
            from catboost import CatBoostRegressor
            g = CatBoostRegressor(
                iterations=n_estimators, learning_rate=learning_rate, depth=6,
                loss_function='RMSE', random_seed=seed, verbose=0).fit(X, y)
            out['cat'] = g.predict(Xv)

    return out


@method(id='gbdt.objective_split', stage=2, status='ADOPTED', cost='low',
        title='같은 트리를 목적함수만 바꿔 멤버로 늘린다 (분류 + 회귀)',
        gain='LB +12.5 (3-way 목적함수 앙상블). A arm 은 GBDT(분류)+LightGBM(회귀) 조합',
        evidence='분류는 로짓 공간, 회귀는 확률 공간에서 오차를 잰다. '
                 '채점식이 Brier 면 회귀 쪽이 지표와 직접 정렬된다',
        requires=[],
        note='★ 가장 싼 다양성이다 — 피처·데이터·구조가 그대로고 손실만 바꾼다. '
             '새 모델 계열을 도입하기 전에 이걸 먼저 하라.')
def objective_split(X, y, X_valid, *, lib='lgb', **kw):
    """같은 라이브러리를 분류/회귀 두 벌로 학습한다.

    반환: {'binary': pred, 'regression': pred}
    """
    return {t: library_trio(X, y, X_valid, task=t, which=(lib,), **kw)[lib]
            for t in ('binary', 'regression')}


@method(id='rej.catboost_cat_features', stage=2, status='REJECTED', cost='med',
        title='CatBoost 의 범주형 자동 처리(cat_features / CTR) 켜기',
        gain='**−15.49** — 켜는 쪽이 손해였다',
        evidence='고카디널리티 엔티티 ID 에 켰다. val 2024 행의 19.9% 가 '
                 'train 에 없는 투수라 자동 타겟 인코딩이 일반화되지 않는다',
        requires=[],
        note='⚠️ "CatBoost 는 범주형에 강하다" 는 통설이 이 데이터에선 틀렸다. '
             '엔티티 ID 는 서수 int 로 넘기고, 룩업이 필요하면 '
             'lookup.target_encoding_eb 처럼 **직접 EB 수축을 통제**하는 편이 나았다. '
             '라이브러리 기본값·권장 설정은 반드시 재보고 결정할 것.')
def catboost_cat_features_note():
    return ('cat_features 는 끄고 ID 를 서수 int64 로. '
            '범주형 신호가 필요하면 직접 만든 EB 룩업(+52~76)을 쓴다.')
