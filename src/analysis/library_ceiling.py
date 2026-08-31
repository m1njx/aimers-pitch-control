"""라이브러리 천장 — 우리가 가진 arm 58종 전부로 1150 이 가능한가?

분석키트 `out/preds/` 에 폴드 홀드아웃 예측이 태그 58종(양 폴드) 있다.
`_f2024_` = `<2024 학습 → 2024 예측` 이라 정직하다.

두 가지를 잰다:
  (1) 단일 arm 스크린 — 각 arm 이 현재 체인 잔차와 얼마나 상관하는가(ρ), ΔG2 는 얼마인가
  (2) **다중 arm 천장** — 58종 전부를 동시에 써서 얻을 수 있는 최대치.
      가중치는 **2022 에서 적합**하고 **2024 에서 평가**한다(누출 차단).
      이게 "우리 재료로 갈 수 있는 끝" 이다.

베이스 체인: game_type 게이트 A/B 블렌드 + pcxh + ctr (내 정직 재구성).
⚠️ ingame 보정과 U 오프셋은 재현하지 않았다 → 베이스가 실제보다 약간 약하다
   → 여기서 나오는 여지는 **실제보다 낙관적**이다(상한).
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, skill  # noqa: E402

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
PRED = os.path.dirname(os.path.abspath(__file__)) + '/preds/out/preds'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
PCXH_B = (0.508947879968906 * .8, 1.27948246622023 * .8, 1000)
CTR_B = (-0.5180599172482175 * .8, -0.9338837106747617 * .8, 2000)


def prod(yr):
    ps = []
    for f in sorted(glob.glob(f'{CACHE}/pred_{yr}_*.npz')):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def teamB(yr):
    fs = sorted(glob.glob(f'{LG}/teamB/out/preds/l2384_f{yr}_s*.npy'))
    return np.mean([np.load(f).astype(float) for f in fs], 0) if fs else None


def logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.where(z != 0., 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z))), p)


def pcxh_shift(d, tdir, bc, bh_, nmin):
    cell = pd.read_csv(f'{tdir}/pcxh_cell.csv').set_index(['pitcher_id', 'cell', 'bh'])['dev_cell']
    hand = pd.read_csv(f'{tdir}/pcxh_hand.csv').set_index(['pitcher_id', 'ph', 'bh'])['dev_hand']
    b = d.balls_before.fillna(0).to_numpy().astype(int)
    s = d.strikes_before.fillna(0).to_numpy().astype(int)
    c = np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2)
    pid = d.pitcher_id.to_numpy().astype(np.int64)
    bh = np.clip(d.batter_hand.fillna(1).to_numpy().astype(int) - 1, 0, 1)
    ph = np.clip(d.pitcher_hand.fillna(1).to_numpy().astype(int) - 1, 0, 1)
    n = d.asof_pitcher_n.fillna(0).to_numpy().astype(float)
    dc = cell.reindex(pd.MultiIndex.from_arrays([pid, c, bh])).to_numpy(float)
    dh = hand.reindex(pd.MultiIndex.from_arrays([pid, ph, bh])).to_numpy(float)
    cov = ~np.isnan(dc)
    dc = np.where(cov, dc, 0.); dh = np.where(cov & ~np.isnan(dh), dh, 0.)
    return (n >= nmin) * (bc * dc + bh_ * dh)


tr = pd.read_csv(f'{LG}/open/data/train.csv')
Y, CH, D = {}, {}, {}
for yr in (2022, 2024):
    d = tr[tr.season == yr].reset_index(drop=True)
    y = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    A, B = prod(yr), teamB(yr)
    wa = np.where(d.game_type.astype(str).values == 'F', 0.2, 0.55)
    p = wa * A + (1 - wa) * B if B is not None else A
    p = logit_add(p, pcxh_shift(d, f'{PAR}/pcxh_parity_le2023', *PCXH_B))
    p = logit_add(p, pcxh_shift(d, f'{PAR}/ctr_parity_le2023', *CTR_B))
    Y[yr], CH[yr], D[yr] = y, np.clip(p, EPS, 1 - EPS), d
    print(f'  체인 {yr}: skill {skill(p, y):8.2f}  (B arm {"있음" if B is not None else "없음"})', flush=True)

tags = sorted(set(os.path.basename(f).split('_f2024')[0] for f in glob.glob(f'{PRED}/*_f2024_*.npy')) &
              set(os.path.basename(f).split('_f2022')[0] for f in glob.glob(f'{PRED}/*_f2022_*.npy')))
ARM = {}
for t in tags:
    ok = True; v = {}
    for yr in (2022, 2024):
        fs = sorted(glob.glob(f'{PRED}/{t}_f{yr}_s*.npy'))
        a = np.mean([np.load(f).astype(float) for f in fs], 0)
        if len(a) != len(Y[yr]):
            ok = False; break
        v[yr] = a
    if ok:
        ARM[t] = v
print(f'\n사용 가능 arm {len(ARM)}종 (길이 일치)\n', flush=True)

# ---------- (1) 단일 arm 스크린 ----------
V24 = Y[2024].mean() * (1 - Y[2024].mean())
base24 = skill(CH[2024], Y[2024])
rows = []
for t, v in ARM.items():
    r = Y[2024] - CH[2024]
    dvec = v[2024] - CH[2024]
    # 이 arm 방향으로 움직여서 얻는 최대 이득 (계수는 2022 에서 적합)
    r22, d22 = Y[2022] - CH[2022], v[2022] - CH[2022]
    b = float(np.dot(d22, r22) / max(np.dot(d22, d22), 1e-12))
    got = skill(np.clip(CH[2024] + b * dvec, EPS, 1 - EPS), Y[2024])
    rho = abs(np.corrcoef(r, dvec)[0, 1])
    rows.append(dict(tag=t, s=skill(v[2024], Y[2024]), beta=b,
                     d24=got - base24, rho=rho))
R = pd.DataFrame(rows).sort_values('d24', ascending=False)
print(f'2024 체인 기준선 {base24:.2f}   게이트 +12 에 필요한 ρ = '
      f'{100*np.sqrt(12/(1e5-base24)):.2f}%\n')
print(f'{"arm":<18}{"단독skill":>10}{"β@2022":>9}{"2024 Δ":>10}{"ρ":>8}')
print('-' * 58)
for _, x in R.head(12).iterrows():
    print(f'{x.tag:<18}{x.s:>10.1f}{x.beta:>9.3f}{x.d24:>+10.2f}{100*x.rho:>7.2f}%')
print('-' * 58)
print(f'단일 arm 최고 Δ = {R.d24.max():+.2f}   (통과 {int((R.d24 > 12).sum())}개 / {len(R)})')

# ---------- (2) 다중 arm 천장 (2022 적합 → 2024 평가) ----------
names = list(ARM)
X22 = np.stack([ARM[t][2022] - CH[2022] for t in names], 1)
X24 = np.stack([ARM[t][2024] - CH[2024] for t in names], 1)
r22 = Y[2022] - CH[2022]
G = X22.T @ X22; g = X22.T @ r22
print(f'\n다중 arm 천장 — {len(names)}종 동시 사용, 가중치는 2022 에서만 적합')
print(f'{"릿지 λ":>10}{"2022 이득":>12}{"2024 이득 ★":>14}')
print('-' * 38)
best = -9e9
for lam in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    w = np.linalg.solve(G + lam * np.eye(len(names)), g)
    i22 = skill(np.clip(CH[2022] + X22 @ w, EPS, 1 - EPS), Y[2022]) - skill(CH[2022], Y[2022])
    i24 = skill(np.clip(CH[2024] + X24 @ w, EPS, 1 - EPS), Y[2024]) - base24
    best = max(best, i24)
    print(f'{lam:>10.0e}{i22:>+12.2f}{i24:>+14.2f}')
print('-' * 38)
print(f'\n★ 라이브러리 58종 전체를 쓴 2024 최대 이득 = {best:+.2f} 폴드 ≈ {best*0.76:+.1f} LB')
print(f'   1150 까지 필요한 +37 LB (= 약 +49 폴드) 대비 -> '
      f'{"도달 가능" if best*0.76 > 37 else "❌ 도달 불가"}')
