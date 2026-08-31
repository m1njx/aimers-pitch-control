"""제안 #3 (동시 재적합) 양방향 폴드 판정 — 자체 테이블 가족으로.

팀의 표를 정확히 복제하지는 못했다(corr 0.994, 최대오차 ~1sd). 그러나 #3 의 질문은
**"적합 절차"** 에 관한 것이지 특정 표에 관한 것이 아니다. 일관된 표 가족을 내가 만들면
양방향 폴드 판정이 가능하다.

표 (내 레시피, bucket=sign(strikes-balls), EB K=50, 이중 중심화):
  pcxh : 타겟 = control_success
  ctr  : 타겟 = reverse − middle  (asof 차분으로 디코딩, success 차원 검산 99.95%)
  ≤2021 판 → 2022 에서 β 적합 / ≤2023 판 → 2024 에서 β 적합

판정 (v38 교훈 반영, 착수 전 확정):
  G1  양방향 전이가 **둘 다** 동시>순차
  G2  동시적합 4계수의 부호가 두 폴드에서 일치
  G3  이득이 폴드내 상한(+3.25)의 절반 이상 실현
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, skill  # noqa: E402

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
K = 50.0
NP, NC = 1000, 2000


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


def keys(d):
    b = d.balls_before.fillna(0).to_numpy().astype(np.int64)
    s = d.strikes_before.fillna(0).to_numpy().astype(np.int64)
    return (d.pitcher_id.to_numpy().astype(np.int64),
            np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2),
            np.sign(s - b).astype(np.int64) + 1,
            np.clip(d.batter_hand.fillna(1).to_numpy().astype(np.int64) - 1, 0, 1),
            np.clip(d.pitcher_hand.fillna(1).to_numpy().astype(np.int64) - 1, 0, 1))


def decode(df):
    x = df.copy(); x['_o'] = np.arange(len(x))
    x = x.sort_values(['season', 'pitcher_id', 'asof_pitcher_n', '_o'])
    g = x.groupby(['season', 'pitcher_id'], sort=False)
    n0 = x.asof_pitcher_n.to_numpy(float); n1 = g.asof_pitcher_n.shift(-1).to_numpy(float)
    ok = (n1 - n0) == 1
    out = {}
    for dim in ('reverse', 'middle'):
        c = f'asof_pitcher_{dim}_rate'
        v = np.round(g[c].shift(-1).to_numpy(float) * n1 - x[c].to_numpy(float) * n0)
        out[dim] = np.where(ok & np.isin(v, [0., 1.]), v, np.nan)
    return pd.DataFrame(out, index=x.index).reindex(df.index)


def build(d, tgt):
    """(cell표, hand표). tgt 는 0/1 또는 NaN."""
    e, cell, bk, bh, ph = keys(d)
    m = ~np.isnan(tgt)
    D = pd.DataFrame(dict(e=e[m], cell=cell[m], bk=bk[m], bh=bh[m], ph=ph[m], y=tgt[m]))
    la_ = D.y.mean()
    lc = D.groupby(['cell', 'bh']).y.mean(); lb = D.groupby('bk').y.mean()
    lh = D.groupby(['ph', 'bh']).y.mean(); lp = D.groupby('ph').y.mean()
    gp = D.groupby(['e', 'bk']).y.agg(['sum', 'count'])
    par = (gp['sum'] + K * lb.reindex(gp.index.get_level_values('bk')).to_numpy()) / (gp['count'] + K)
    g1 = D.groupby(['e', 'cell', 'bh', 'bk']).y.agg(['sum', 'count']).reset_index()
    p1 = par.reindex(pd.MultiIndex.from_arrays([g1['e'], g1['bk']])).to_numpy()
    r1 = (g1['sum'].to_numpy() + K * p1) / (g1['count'].to_numpy() + K)
    g1['dev'] = (r1 - p1) - (lc.reindex(pd.MultiIndex.from_arrays([g1['cell'], g1['bh']])).to_numpy()
                             - lb.reindex(g1['bk']).to_numpy())
    ge = D.groupby(['e', 'ph']).y.agg(['sum', 'count'])
    pe = (ge['sum'] + K * lp.reindex(ge.index.get_level_values('ph')).to_numpy()) / (ge['count'] + K)
    g2 = D.groupby(['e', 'ph', 'bh']).y.agg(['sum', 'count']).reset_index()
    p2 = pe.reindex(pd.MultiIndex.from_arrays([g2['e'], g2['ph']])).to_numpy()
    r2 = (g2['sum'].to_numpy() + K * p2) / (g2['count'].to_numpy() + K)
    g2['dev'] = (r2 - p2) - (lh.reindex(pd.MultiIndex.from_arrays([g2['ph'], g2['bh']])).to_numpy()
                             - lp.reindex(g2['ph']).to_numpy())
    return g1.set_index(['e', 'cell', 'bh'])['dev'], g2.set_index(['e', 'ph', 'bh'])['dev']


def feats(d, T, nmin):
    e, cell, _, bh, ph = keys(d)
    x1 = np.nan_to_num(T[0].reindex(pd.MultiIndex.from_arrays([e, cell, bh])).to_numpy(float))
    x2 = np.nan_to_num(T[1].reindex(pd.MultiIndex.from_arrays([e, ph, bh])).to_numpy(float))
    g = (d.asof_pitcher_n.fillna(0).to_numpy() >= nmin).astype(float)
    return g * x1, g * x2


tr = pd.read_csv(f'{LG}/open/data/train.csv')
DEC = decode(tr)
print('표 구축...', flush=True)
TB = {}
for up in (2021, 2023):
    sub = tr[tr.season <= up]
    TB[('pcxh', up)] = build(sub, sub.control_success.to_numpy(float))
    TB[('ctr', up)] = build(sub, (DEC.reverse - DEC.middle).loc[sub.index].to_numpy(float))
    print(f'  ≤{up}: pcxh {len(TB[("pcxh",up)][0]):,}셀 / ctr {len(TB[("ctr",up)][0]):,}셀', flush=True)

FOLD = {}
for yr, up in ((2022, 2021), (2024, 2023)):
    d = tr[tr.season == yr].reset_index(drop=True)
    y = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    wa = np.where(d.game_type.astype(str).values == 'F', 0.2, 0.55)
    base = np.clip(wa * prod(yr) + (1 - wa) * teamB(yr), EPS, 1 - EPS)
    a1, a2 = feats(d, TB[('pcxh', up)], NP)
    c1, c2 = feats(d, TB[('ctr', up)], NC)
    FOLD[yr] = dict(y=y, base=base, X=np.stack([a1, a2, c1, c2], 1))
    print(f'  fold {yr} (표 ≤{up}): base {skill(base, y):8.2f}', flush=True)


def fit(f, mode):
    y, p, X = f['y'], f['base'], f['X']
    w = np.clip(p * (1 - p), 1e-9, None)
    if mode == 'joint':
        return np.linalg.lstsq(X * w[:, None], y - p, rcond=None)[0]
    b1 = np.linalg.lstsq(X[:, :2] * w[:, None], y - p, rcond=None)[0]
    p1 = la(p, X[:, :2] @ b1); w1 = np.clip(p1 * (1 - p1), 1e-9, None)
    b2 = np.linalg.lstsq(X[:, 2:] * w1[:, None], y - p1, rcond=None)[0]
    return np.concatenate([b1, b2])


def ev(f, b, mode):
    base = skill(f['base'], f['y'])
    q = la(f['base'], f['X'] @ b) if mode == 'joint' else \
        la(la(f['base'], f['X'][:, :2] @ b[:2]), f['X'][:, 2:] @ b[2:])
    return skill(q, f['y']) - base


print('\n' + '=' * 70)
print(f'{"fit":>6}{"eval":>7}{"순차":>11}{"동시":>11}{"동시−순차":>12}')
print('-' * 70)
R = []
for fy, ey in ((2022, 2024), (2024, 2022)):
    r = {}
    for m in ('seq', 'joint'):
        b = fit(FOLD[fy], m) * 0.8          # 출하와 동일하게 0.8 수축
        r[m] = ev(FOLD[ey], b, m); r[m + 'b'] = b
    R.append(r)
    print(f'{fy:>6}{ey:>7}{r["seq"]:>+11.2f}{r["joint"]:>+11.2f}{r["joint"]-r["seq"]:>+12.2f}')
print('-' * 70)
B = np.stack([r['jointb'] for r in R])
print('동시적합 계수 (0.8 수축 후):')
for i, nm in enumerate(['pcxh_cell', 'pcxh_hand', 'ctr_cell', 'ctr_hand']):
    print(f'  {nm:<11} {np.round(B[:, i], 4)}  '
          f'{"✅" if len(set(np.sign(B[:, i]))) == 1 else "🔴 부호 뒤집힘"}')
g1 = all(r['joint'] > r['seq'] for r in R)
g2 = all(len(set(np.sign(B[:, i]))) == 1 for i in range(4))
avg = np.mean([r['joint'] - r['seq'] for r in R])
g3 = avg > 1.6
print(f'\nG1 양방향 모두 동시>순차 : {"PASS" if g1 else "FAIL"}')
print(f'G2 4계수 부호 일치       : {"PASS" if g2 else "FAIL"}')
print(f'G3 평균 이득 > +1.6      : {avg:+.2f} -> {"PASS" if g3 else "FAIL"}')
print(f'\n판정: {"✅ 채택 후보" if (g1 and g2 and g3) else "❌ 기각"}')
