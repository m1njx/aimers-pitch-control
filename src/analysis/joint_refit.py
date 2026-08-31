"""제안 #3(동시 재적합) + #4(n_min 그리드) 실측.

v38 실패의 교훈을 적용한다:
  - **양방향 폴드**로 본다 (2022→2024 와 2024→2022). 한 방향만 보면 v38 처럼 속는다.
  - **부호 안정성**을 먼저 본다. 폴드 간 부호가 뒤집히면 즉시 기각.
  - 2023R 도 같이 본다 (LB 최근사 폴드).
  - 결정적 계산이라 시드 잡음은 없다(모델 학습이 아니라 계수 적합).

#3 순차 vs 동시:
   현재 = pcxh 4계수를 먼저 적합해 얹고, 그 위에서 ctr 4계수를 적합.
   동시 = pcxh·ctr 의 4개 편차항을 **한 번에** 최소제곱으로 적합.
   동시가 낫다면 순차에서 이중계상 손실이 있다는 뜻.

#4 n_min:
   현재 pcxh 1000 / ctr 2000. 격자로 훑되 **다른 폴드에서 고른 값**을 평가폴드에서 확인.

[기여 구분] 여기서 다루는 엔티티 잔차 룩업 기법 자체는 **팀 공동 파트에서 도입**됐다.
이 스크립트는 그 기법에 대해 내가 수행한 **독립 검증·확장 시도**다.
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, skill  # noqa: E402

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
SHIP = dict(pcxh=(0.5113952298199526, 1.0317445073629654, 1000),
            ctr=(-0.59814543382707, -0.7189782550594095, 2000))


def prod(yr):
    ps = []
    for f in sorted(glob.glob(f'{CACHE}/pred_{yr}_*.npz')):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def teamB(yr):
    fs = sorted(glob.glob(f'{LG}/teamB/out/preds/l2384_f{yr}_s*.npy'))
    return np.mean([np.load(f).astype(float) for f in fs], 0)


def la(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.clip(1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z))), EPS, 1 - EPS)


def devs(d, tdir):
    """게이트를 걸지 않은 원시 편차 2개 (게이트는 나중에 n_min 으로 적용)."""
    cell = pd.read_csv(f'{tdir}/pcxh_cell.csv').set_index(['pitcher_id', 'cell', 'bh'])['dev_cell']
    hand = pd.read_csv(f'{tdir}/pcxh_hand.csv').set_index(['pitcher_id', 'ph', 'bh'])['dev_hand']
    b = d.balls_before.fillna(0).to_numpy().astype(int)
    s = d.strikes_before.fillna(0).to_numpy().astype(int)
    c = np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2)
    pid = d.pitcher_id.to_numpy().astype(np.int64)
    bh = np.clip(d.batter_hand.fillna(1).to_numpy().astype(int) - 1, 0, 1)
    ph = np.clip(d.pitcher_hand.fillna(1).to_numpy().astype(int) - 1, 0, 1)
    dc = cell.reindex(pd.MultiIndex.from_arrays([pid, c, bh])).to_numpy(float)
    dh = hand.reindex(pd.MultiIndex.from_arrays([pid, ph, bh])).to_numpy(float)
    cov = ~np.isnan(dc)
    return np.where(cov, dc, 0.), np.where(cov & ~np.isnan(dh), dh, 0.)


tr = pd.read_csv(f'{LG}/open/data/train.csv')
F = {}
for yr in (2022, 2023, 2024):
    d = tr[tr.season == yr].reset_index(drop=True)
    y = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    m = (d.game_type.values == 'R') if yr == 2023 else np.ones(len(y), bool)
    wa = np.where(d.game_type.astype(str).values == 'F', 0.2, 0.55)
    ab = np.clip(wa * prod(yr) + (1 - wa) * teamB(yr), EPS, 1 - EPS)
    pc, ph_ = devs(d, f'{PAR}/pcxh_parity_le2023')
    cc, ch_ = devs(d, f'{PAR}/ctr_parity_le2023')
    n = d.asof_pitcher_n.fillna(0).to_numpy().astype(float)
    F[yr] = dict(y=y[m], base=ab[m], n=n[m],
                 X=np.stack([pc[m], ph_[m], cc[m], ch_[m]], 1))
    print(f'  fold {yr}{"R" if yr == 2023 else ""}: n={m.sum():,}  base={skill(ab[m], y[m]):8.2f}', flush=True)


def gate(f, nm_p, nm_c):
    g = np.stack([f['n'] >= nm_p, f['n'] >= nm_p, f['n'] >= nm_c, f['n'] >= nm_c], 1).astype(float)
    return f['X'] * g


def fit(f, Xg, mode):
    """mode='joint' 4계수 동시 / 'seq' pcxh 먼저 -> ctr 나중 (현재 방식)"""
    y, p = f['y'], f['base']
    if mode == 'joint':
        w = np.clip(p * (1 - p), 1e-9, None)
        return np.linalg.lstsq(Xg * w[:, None], y - p, rcond=None)[0]
    w = np.clip(p * (1 - p), 1e-9, None)
    b1 = np.linalg.lstsq(Xg[:, :2] * w[:, None], y - p, rcond=None)[0]
    p1 = la(p, Xg[:, :2] @ b1)
    w1 = np.clip(p1 * (1 - p1), 1e-9, None)
    b2 = np.linalg.lstsq(Xg[:, 2:] * w1[:, None], y - p1, rcond=None)[0]
    return np.concatenate([b1, b2])


def apply_(f, Xg, b, mode):
    if mode == 'joint':
        return la(f['base'], Xg @ b)
    return la(la(f['base'], Xg[:, :2] @ b[:2]), Xg[:, 2:] @ b[2:])


print('\n' + '=' * 74)
print('[#3] 순차 vs 동시 재적합 — 계수는 fit 폴드, 평가는 다른 폴드')
print(f'{"fit":>6}{"eval":>7}{"순차":>11}{"동시":>11}{"차이":>9}{"부호일치":>10}')
print('-' * 74)
res = []
for fit_y, ev_y in [(2022, 2024), (2024, 2022), (2022, 2023), (2024, 2023)]:
    Xf = gate(F[fit_y], 1000, 2000); Xe = gate(F[ev_y], 1000, 2000)
    row = {}
    for mode in ('seq', 'joint'):
        b = fit(F[fit_y], Xf, mode)
        base = skill(F[ev_y]['base'], F[ev_y]['y'])
        row[mode] = skill(apply_(F[ev_y], Xe, b, mode), F[ev_y]['y']) - base
        row[mode + '_b'] = b
    sg = bool(np.all(np.sign(row['joint_b']) == np.sign(row['seq_b'])))
    res.append((fit_y, ev_y, row['seq'], row['joint'], row['joint_b']))
    tag = f'{ev_y}{"R" if ev_y == 2023 else ""}'
    print(f'{fit_y:>6}{tag:>7}{row["seq"]:>+11.2f}{row["joint"]:>+11.2f}'
          f'{row["joint"]-row["seq"]:>+9.2f}{"O" if sg else "X":>10}')
print('-' * 74)
B = np.stack([r[4] for r in res])
print('동시적합 계수의 폴드 간 부호 일관성:')
for i, nm in enumerate(['pcxh_cell', 'pcxh_hand', 'ctr_cell', 'ctr_hand']):
    s = np.sign(B[:, i])
    print(f'  {nm:<11} {np.round(B[:, i], 3)}  -> {"✅ 일치" if len(set(s)) == 1 else "🔴 뒤집힘"}')
gain = np.mean([r[3] - r[2] for r in res])
print(f'\n동시적합의 평균 이득 = {gain:+.2f} 폴드 ≈ {gain*0.76:+.1f} LB')

print('\n' + '=' * 74)
print('[#4] n_min 그리드 — 2022 에서 고르고 2024·2023R 에서 확인')
grid = [500, 1000, 1500, 2000, 3000]
best = None
print(f'{"n_p":>6}{"n_c":>6}{"2022(적합)":>12}{"2024":>10}{"2023R":>10}')
print('-' * 74)
for npp in grid:
    for nc in grid:
        b = fit(F[2022], gate(F[2022], npp, nc), 'seq')
        v = {}
        for ey in (2022, 2024, 2023):
            base = skill(F[ey]['base'], F[ey]['y'])
            v[ey] = skill(apply_(F[ey], gate(F[ey], npp, nc), b, 'seq'), F[ey]['y']) - base
        if best is None or v[2022] > best[0]:
            best = (v[2022], npp, nc, v[2024], v[2023])
        if (npp, nc) in [(1000, 2000), (500, 500), (3000, 3000), (1500, 1500)]:
            print(f'{npp:>6}{nc:>6}{v[2022]:>+12.2f}{v[2024]:>+10.2f}{v[2023]:>+10.2f}'
                  + ('   <- 출하값' if (npp, nc) == (1000, 2000) else ''))
print('-' * 74)
b = fit(F[2022], gate(F[2022], 1000, 2000), 'seq')
ship = {ey: skill(apply_(F[ey], gate(F[ey], 1000, 2000), b, 'seq'), F[ey]['y'])
        - skill(F[ey]['base'], F[ey]['y']) for ey in (2024, 2023)}
print(f'2022 가 고른 조합: n_p={best[1]} n_c={best[2]} (2022 {best[0]:+.2f})')
print(f'  -> 2024 실현 {best[3]:+.2f} (출하값 {ship[2024]:+.2f}, 차 {best[3]-ship[2024]:+.2f})')
print(f'  -> 2023R 실현 {best[4]:+.2f} (출하값 {ship[2023]:+.2f}, 차 {best[4]-ship[2023]:+.2f})')
