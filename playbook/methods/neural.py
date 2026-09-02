"""neural.py — 표형 데이터에서의 신경망.

이 대회의 결론은 한 줄로 요약된다:

    **MLP 한 종은 크게 성공했고(+36.24), 종류를 늘리는 것은 일관되게 실패했다.**

ResNet·Transformer·TabNet·TabM 은 전부 음수였다 (→ `rej.deep_tabular_variety`).
엔티티 임베딩과 지식 증류도 기각됐다 (→ `rej.entity_embedding`, `rej.knowledge_distillation`).
그러므로 여기 ADOPTED 로 남는 것은 **평범한 MLP 하나**와 **목적함수를 나누는 법** 둘뿐이다.

⚠️ torch 는 함수 안에서 import 한다 — torch 없는 환경에서도 카탈로그 조회가 되어야 한다.
"""
from __future__ import annotations
import numpy as np
from ._base import method


@method(id='nn.mlp_member', stage=2, status='ADOPTED', cost='med',
        title='MLP 를 블렌드 멤버로 (시드 배깅)',
        gain='LB +36.24 — 1,000점 최초 돌파. A arm 내 가중치 0.42 로 GBDT(0.38)보다 크다',
        evidence='GBDT 와 오차 방향이 다르다. 단독 성능이 아니라 ρ 로 값을 낸다',
        requires=['n_seeds'],
        note='★ GBDT 기준선이 선 뒤에 붙일 것. 종류는 늘리지 말 것 — '
             'ResNet/Transformer/TabNet/TabM 전부 음수였다(rej.deep_tabular_variety). '
             '단일 시드는 분산이 커서 그대로 쓰면 안 된다. 최소 5시드 예측 평균.')
def mlp_member(X, y, X_valid, *, objective='mse', seeds=(0, 1, 2, 3, 4),
               hidden=(256, 128, 64), epochs=12, batch_size=4096, lr=1e-3,
               weight_decay=1e-5, dropout=0.1, device=None):
    """MLP 를 시드별로 학습해 **예측을 평균**한다.

    가중치 평균이 아니라 예측 평균이다 — 신경망은 가중치 공간에서 치환 대칭이라
    가중치를 평균하면 망가진다.

    objective: 'mse'  회귀(Brier 와 같은 손실) — 이 대회에서 채택된 쪽
               'bce'  분류
    반환: (X_valid 예측, 시드별 예측 스택)
    """
    import torch
    import torch.nn as nn

    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    Xtr = torch.tensor(np.asarray(X, np.float32))
    ytr = torch.tensor(np.asarray(y, np.float32)).view(-1, 1)
    Xva = torch.tensor(np.asarray(X_valid, np.float32)).to(dev)

    def build(seed):
        torch.manual_seed(seed)
        layers, d = [], Xtr.shape[1]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        return nn.Sequential(*layers).to(dev)

    loss_fn = nn.MSELoss() if objective == 'mse' else nn.BCEWithLogitsLoss()
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    out = []

    for s in seeds:
        net = build(s)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(s))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=max(1, epochs * len(dl)))
        net.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                z = net(xb)
                # mse 는 확률 공간에서, bce 는 로짓 공간에서 손실을 잰다
                loss = loss_fn(torch.sigmoid(z) if objective == 'mse' else z, yb)
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(Xva)).cpu().numpy().ravel()
        out.append(p)

    P = np.stack(out)
    return P.mean(0), P


@method(id='nn.multi_objective', stage=2, status='ADOPTED', cost='med',
        title='목적함수를 나눠 멤버를 만든다 (분류 + 회귀)',
        gain='LB +12.5 — 같은 구조·같은 피처인데 손실만 바꿔 다양성을 얻었다',
        evidence='BCE 는 로짓 공간, MSE 는 확률 공간에서 오차를 재므로 '
                 '틀리는 행이 다르다. 채점식이 Brier 면 MSE 쪽이 지표와 정렬된다',
        requires=[],
        note='★ 새 모델을 만들기 전에 **가진 모델의 목적함수를 바꿔보는 것이 훨씬 싸다.** '
             '피처·구조·데이터가 그대로라 구현 비용이 거의 0 인데 ρ 는 실제로 생긴다.')
def multi_objective(X, y, X_valid, **kw):
    """같은 MLP 를 목적함수만 바꿔 두 벌 학습하고 각각의 예측을 돌려준다.

    반환: {'mse': 예측, 'bce': 예측} — 결합 가중치는 `ens.gram` 으로 푼다.
    """
    return {o: mlp_member(X, y, X_valid, objective=o, **kw)[0] for o in ('mse', 'bce')}


@method(id='nn.pred_bagging_not_weight', stage=2, status='ADOPTED', cost='low',
        title='신경망은 가중치가 아니라 예측을 평균한다',
        gain='단일 시드 대비 분산 감소. 5시드 미만은 노이즈를 4~5배 과소평가한다',
        evidence='은닉 유닛 치환 대칭 때문에 서로 다른 해의 가중치 평균은 무의미하다',
        requires=[],
        note='⚠️ 시드 하나로 판정한 실험이 실전에서 −4.1 을 냈다. 최소 5시드.')
def pred_bagging(preds):
    """시드별 예측 스택 (n_seeds, n_rows) → 평균 예측과 시드 간 표준편차."""
    P = np.asarray(preds, float)
    return P.mean(0), P.std(0)
