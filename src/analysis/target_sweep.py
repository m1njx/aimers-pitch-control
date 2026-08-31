"""pcxh 구조 위의 **새 타겟** 스윕 — ctr 이 열어둔 문을 끝까지 연다.

팀의 ctr 은 (투수 × 카운트셀 × 타자손) 키에 **reverse−middle 대비**를 얹어 +11.00 을 냈다.
그런데 주최자가 추적하는 결과 차원은 5개다: success / reverse / middle / **ball** / **strike**.
ball·strike 는 **심판 콜 차원**으로 미스 위치 차원과 다른 정보다 — 아직 아무도 안 썼다.

디코딩: 연속 동일투수 행에서 asof 누적률의 차분으로 그 투구의 결과를 복원한다
  outcome = rate(다음행)·n(다음행) − rate(이번행)·n(이번행)
검산 완료: success 차원을 알려진 라벨과 대조해 **99.95% 일치**. 99.8% 의 행에서 복원 가능.
⚠️ 디코딩은 **train 에서만** 한다. test 행끼리 참조하지 않으므로 규정 4 안전
   (테이블은 train 적합 자산이고, 적용은 행 자신의 키로만 조회한다).

정직 프로토콜: 표 ≤2021 → β@2022 적합 / 표 ≤2023 → β@2022×0.8 로 2024 적용
측정: pcxh+ctr 체인 **위의 증분**
🎲 대조군: 디코딩 타겟을 투수 내에서 셔플(주변분포 보존, 셀 신호 파괴) — 위양성 바닥

사전 확정 기준: G1 2024 > max(+12, 대조군 최고) / G2 2022 > 0 / G3 β 부호 일치
"""
import glob, os, time
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
K, N_MIN, SCALE = 100.0, 1000, 0.8
PCXH_B = (0.508947879968906, 1.27948246622023, 1000)
CTR_B = (-0.5180599172482175, -0.9338837106747617, 2000)
DIMS = ['success', 'reverse', 'middle', 'ball', 'strike']


def prod(yr):
    ps = []
    for f in sorted(glob.glob(f'{CACHE}/pred_{yr}_*.npz')):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def skill(p, y):
    r = y.mean(); return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.where(z != 0., 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z))), p)


def cellbucket(d):
    b = d['balls_before'].fillna(0).to_numpy().astype(np.int64)
    s = d['strikes_before'].fillna(0).to_numpy().astype(np.int64)
    return np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2), np.sign(s - b).astype(np.int64) + 1


def shift_par(d, tdir, bc, bh_, nmin):
    cell = pd.read_csv(f'{tdir}/pcxh_cell.csv').set_index(['pitcher_id', 'cell', 'bh'])['dev_cell']
    hand = pd.read_csv(f'{tdir}/pcxh_hand.csv').set_index(['pitcher_id', 'ph', 'bh'])['dev_hand']
    c, _ = cellbucket(d)
    pid = d['pitcher_id'].to_numpy().astype(np.int64)
    bh = np.clip(d['batter_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    ph = np.clip(d['pitcher_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    n = d['asof_pitcher_n'].fillna(0).to_numpy().astype(float)
    dc = cell.reindex(pd.MultiIndex.from_arrays([pid, c, bh])).to_numpy(float)
    dh = hand.reindex(pd.MultiIndex.from_arrays([pid, ph, bh])).to_numpy(float)
    cov = ~np.isnan(dc)
    dc = np.where(cov, dc, 0.); dh = np.where(cov & ~np.isnan(dh), dh, 0.)
    return (n >= nmin) * (bc * SCALE * dc + bh_ * SCALE * dh)


def decode(df):
    """train 행마다 5차원 결과를 복원. 복원 불가 행은 NaN."""
    d = df.copy(); d['_o'] = np.arange(len(d))
    d = d.sort_values(['season', 'pitcher_id', 'asof_pitcher_n', '_o'])
    g = d.groupby(['season', 'pitcher_id'], sort=False)
    n0 = d.asof_pitcher_n.to_numpy(float); n1 = g.asof_pitcher_n.shift(-1).to_numpy(float)
    ok = (n1 - n0) == 1
    out = {}
    for dim in DIMS:
        c = f'asof_pitcher_{dim}_rate'
        v = g[c].shift(-1).to_numpy(float) * n1 - d[c].to_numpy(float) * n0
        v = np.round(v)
        out[dim] = np.where(ok & np.isin(v, [0., 1.]), v, np.nan)
    R = pd.DataFrame(out, index=d.index).reindex(df.index)
    return R


def build(tr, tgt, cell, bucket, bh):
    """이중 중심화 + EB 수축. tgt 는 0/1/NaN."""
    m = ~np.isnan(tgt)
    D = pd.DataFrame(dict(e=tr['pitcher_id'].to_numpy().astype(np.int64)[m],
                          cell=cell[m], bucket=bucket[m], bh=bh[m], y=tgt[m]))
    lg_all = D.y.mean()
    lg_cb = D.groupby(['cell', 'bh']).y.mean(); lg_b = D.groupby('bucket').y.mean()
    lg_h = D.groupby('bh').y.mean()
    gp = D.groupby(['e', 'bucket']).y.agg(['sum', 'count'])
    r_p = (gp['sum'] + K * lg_b.reindex(gp.index.get_level_values('bucket')).to_numpy()) / (gp['count'] + K)
    g = D.groupby(['e', 'cell', 'bh', 'bucket']).y.agg(['sum', 'count']).reset_index()
    par = r_p.reindex(pd.MultiIndex.from_arrays([g['e'], g['bucket']])).to_numpy()
    rc = (g['sum'].to_numpy() + K * par) / (g['count'].to_numpy() + K)
    g['dev'] = (rc - par) - (lg_cb.reindex(pd.MultiIndex.from_arrays([g['cell'], g['bh']])).to_numpy()
                             - lg_b.reindex(g['bucket']).to_numpy())
    ge = D.groupby('e').y.agg(['sum', 'count'])
    r_e = (ge['sum'] + K * lg_all) / (ge['count'] + K)
    g2 = D.groupby(['e', 'bh']).y.agg(['sum', 'count']).reset_index()
    p2 = r_e.reindex(g2['e']).to_numpy()
    rt = (g2['sum'].to_numpy() + K * p2) / (g2['count'].to_numpy() + K)
    g2['dev'] = (rt - p2) - (lg_h.reindex(g2['bh']).to_numpy() - lg_all)
    return g.set_index(['e', 'cell', 'bh'])['dev'], g2.set_index(['e', 'bh'])['dev']


def feats(d, T1, T2):
    cell, _ = cellbucket(d)
    e = d['pitcher_id'].to_numpy().astype(np.int64)
    bh = np.clip(d['batter_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    x1 = np.nan_to_num(T1.reindex(pd.MultiIndex.from_arrays([e, cell, bh])).to_numpy(float))
    x2 = np.nan_to_num(T2.reindex(pd.MultiIndex.from_arrays([e, bh])).to_numpy(float))
    g = (d['asof_pitcher_n'].fillna(0).to_numpy() >= N_MIN).astype(float)
    return g * x1, g * x2


def fit_beta(p, y, x1, x2):
    pc = np.clip(p, EPS, 1 - EPS); w = pc * (1 - pc)
    return np.linalg.lstsq(np.stack([x1 * w, x2 * w], 1), y - p, rcond=None)[0]


print('train.csv 로드 + 5차원 디코딩...', flush=True)
df = pd.read_csv(f'{LG}/open/data/train.csv')
DEC = decode(df)
print(f'  복원률 {100*DEC.success.notna().mean():.1f}%', flush=True)

base = {}
for yr in (2022, 2024):
    yv = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    s = df[df.season == yr].reset_index(drop=True)
    p = prod(yr)
    p = logit_add(p, shift_par(s, f'{PAR}/pcxh_parity_le2023', *PCXH_B))
    p = logit_add(p, shift_par(s, f'{PAR}/ctr_parity_le2023', *CTR_B))
    base[yr] = (s, yv, p)
    print(f'  베이스 {yr}: {skill(p, yv):.2f}', flush=True)

# 타겟 후보
rng = np.random.default_rng(5)
def targets(idx):
    D = DEC.loc[idx]
    t = {'ball': D.ball.to_numpy(), 'strike': D.strike.to_numpy(),
         'reverse': D.reverse.to_numpy(), 'middle': D.middle.to_numpy(),
         'ball-strike': D.ball.to_numpy() - D.strike.to_numpy(),
         'rev-mid': D.reverse.to_numpy() - D.middle.to_numpy(),
         'ball+rev': D.ball.to_numpy() + D.reverse.to_numpy()}
    for i in range(4):                                  # 🎲 대조군
        v = D.ball.to_numpy().copy(); ok = ~np.isnan(v)
        vv = v[ok].copy(); rng.shuffle(vv); v[ok] = vv
        t[f'RANDOM:{i}'] = v
    return t

tr21, tr23 = df[df.season <= 2021], df[df.season <= 2023]
T21, T23 = targets(tr21.index), targets(tr23.index)
c21, b21 = cellbucket(tr21); h21 = np.clip(tr21['batter_hand'].fillna(1).to_numpy().astype(np.int64)-1, 0, 1)
c23, b23 = cellbucket(tr23); h23 = np.clip(tr23['batter_hand'].fillna(1).to_numpy().astype(np.int64)-1, 0, 1)

rows, t0 = [], time.time()
for nm in T21:
    A1, A2 = build(tr21, T21[nm], c21, b21, h21)
    B1, B2 = build(tr23, T23[nm], c23, b23, h23)
    d22, y22, p22 = base[2022]
    x1, x2 = feats(d22, A1, A2)
    b = fit_beta(p22, y22, x1, x2)
    i22 = skill(logit_add(p22, b[0]*x1 + b[1]*x2), y22) - skill(p22, y22)
    d24, y24, p24 = base[2024]
    z1, z2 = feats(d24, B1, B2)
    i24 = skill(logit_add(p24, SCALE*(b[0]*z1 + b[1]*z2)), y24) - skill(p24, y24)
    b24 = fit_beta(p24, y24, z1, z2)
    rows.append(dict(name=nm, b1=b[0], b2=b[1], inc22=i22, inc24=i24,
                     sign=bool(b[0]*b24[0] > 0), rnd=nm.startswith('RANDOM')))
    print(f'  {nm:<14} 2022 {i22:+8.2f}   2024 {i24:+8.2f}   ({time.time()-t0:.0f}s)', flush=True)

R = pd.DataFrame(rows); R.to_csv(f'{LG}/outputs/532_target_sweep.csv', index=False)
real, rnd = R[~R.rnd], R[R.rnd]
thr = max(12.0, rnd.inc24.max())
print('\n' + '='*72)
print(f'[위양성 바닥] 대조군 {len(rnd)}개 2024 최고 {rnd.inc24.max():+.2f}  임계 {thr:+.2f}')
print('='*72)
print(f'{"타겟":<14}{"2022":>10}{"2024":>10}{"β부호":>7}  판정')
print('-'*72)
for _, x in real.sort_values('inc24', ascending=False).iterrows():
    ok = x.inc24 > thr and x.inc22 > 0 and x.sign
    print(f'{x["name"]:<14}{x.inc22:>+10.2f}{x.inc24:>+10.2f}{"O" if x.sign else "X":>7}  '
          f'{"✅ PASS" if ok else "—"}')
win = real[(real.inc24 > thr) & (real.inc22 > 0) & real.sign]
print('-'*72)
print(f'\n통과 {len(win)}개' + (f': {list(win.name)}' if len(win) else ' — 정직한 null'))
