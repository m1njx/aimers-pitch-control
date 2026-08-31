"""pcxh 제3축 스윕 — batter_hand 자리에 다른 문맥축을 넣어 증분을 잰다.

팀 파이프라인의 엔티티 잔차 룩업 는 (투수 × 카운트셀 × 타자손). ctr 은 같은 구조의 다른 타겟.
둘 다 실측으로 살아 있으므로(내 독립측정 +19.70 / +11.00) **제3축 교체**가 유일하게
열린 확장이다. 여기서 여러 후보를 한 번에 훑는다.

각 후보 T 마다 팀과 동일한 2단 구조를 만든다:
    dev_cellT = EB[r(투수, cell, T)] − r(투수, bucket) − [리그 r(cell,T) − 리그 r(bucket)]
    dev_T     = EB[r(투수, T)]       − r(투수)        − [리그 r(T)      − 리그 전체]
    shift = gate(asof_pitcher_n ≥ 1000) · (β1·dev_cellT + β2·dev_T)

정직 프로토콜 (팀과 동일):
    표 ≤2021 → 2022 에서 (β1,β2) 최소제곱 적합
    표 ≤2023 → **2022 에서 적합한 β × 0.8** 로 2024 에 적용
  즉 2024 는 표로도 β로도 라벨이 새지 않는다.

측정 대상은 **pcxh+ctr 체인 위의 증분**이다(이미 있는 것과 겹치면 0 이 나와야 정상).

🎲 무작위 대조군: 라벨과 무관한 난수 축을 같은 파이프라인에 통과시켜
   자유도가 만드는 위양성 바닥을 함께 잰다. 이게 없으면 다축 스윕은 자기기만이다.

사전 확정 기준 (실행 전 고정):
   G1  2024 전이 증분 > max(+12.0, 무작위 최고치)
   G2  2022 증분 > 0
   G3  β 부호가 두 폴드에서 일치
"""
import glob, os, sys, time
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
K = 100.0
N_MIN = 1000
SCALE = 0.8
PCXH_B = (0.508947879968906, 1.27948246622023, 1000)
CTR_B = (-0.5180599172482175, -0.9338837106747617, 2000)
N_RANDOM = 12


def prod(yr):
    ps = []
    for f in sorted(glob.glob(f'{CACHE}/pred_{yr}_*.npz')):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    q = 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z)))
    return np.where(z != 0., q, p)


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


# ---------- 제3축 후보 ----------
def axes(d, rng):
    """이름 -> 정수코드(0..L-1). 반드시 **행 자신의 컬럼**만 사용(규정 4)."""
    o = {}
    o['outs'] = d['outs_before'].fillna(0).to_numpy().astype(np.int64).clip(0, 2)
    o['scoring_pos'] = ((d['runner_on_2b'].fillna(0).to_numpy() > 0) |
                        (d['runner_on_3b'].fillna(0).to_numpy() > 0)).astype(np.int64)
    o['nrunners'] = d['num_runners_on'].fillna(0).to_numpy().astype(np.int64).clip(0, 3)
    o['base_state'] = pd.factorize(d['base_state'].astype(str))[0]
    o['inning3'] = np.digitize(d['inning'].fillna(1).to_numpy(), [4, 7])
    o['tb'] = (d['top_bottom'].astype(str).to_numpy() == 'T').astype(np.int64)
    sd = d['score_diff_pitcher_team'].fillna(0).to_numpy()
    o['scorediff3'] = np.digitize(sd, [-1.5, 1.5])
    li = d['li'].fillna(1.0).to_numpy()
    o['leverage3'] = np.digitize(li, [0.85, 1.5])
    o['month3'] = np.digitize(d['game_month'].fillna(6).to_numpy(), [6, 8])
    o['gtype'] = (d['game_type'].astype(str).to_numpy() == 'F').astype(np.int64)
    o['bh_x_out'] = (np.clip(d['batter_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1) * 3
                     + d['outs_before'].fillna(0).to_numpy().astype(np.int64).clip(0, 2))
    for i in range(N_RANDOM):                      # 🎲 위양성 바닥
        o[f'RANDOM:{i:02d}'] = rng.integers(0, 3, len(d))
    return o


def build(tr, code):
    """(dev_cellT 시리즈, dev_T 시리즈) — 이중 중심화 + EB 수축."""
    cell, bucket = cellbucket(tr)
    e = tr['pitcher_id'].to_numpy().astype(np.int64)
    y = tr['control_success'].to_numpy().astype(float)
    D = pd.DataFrame(dict(e=e, cell=cell, bucket=bucket, t=code, y=y))
    lg_all = y.mean()
    lg_ct = D.groupby(['cell', 't']).y.mean()
    lg_b = D.groupby('bucket').y.mean()
    lg_t = D.groupby('t').y.mean()
    # 부모: 투수×bucket
    gp = D.groupby(['e', 'bucket']).y.agg(['sum', 'count'])
    r_p = (gp['sum'] + K * lg_b.reindex(gp.index.get_level_values('bucket')).to_numpy()) / (gp['count'] + K)
    g = D.groupby(['e', 'cell', 't', 'bucket']).y.agg(['sum', 'count']).reset_index()
    par = r_p.reindex(pd.MultiIndex.from_arrays([g['e'], g['bucket']])).to_numpy()
    rc = (g['sum'].to_numpy() + K * par) / (g['count'].to_numpy() + K)
    g['dev'] = (rc - par) - (lg_ct.reindex(pd.MultiIndex.from_arrays([g['cell'], g['t']])).to_numpy()
                             - lg_b.reindex(g['bucket']).to_numpy())
    # 거친 표: 투수×T
    ge = D.groupby('e').y.agg(['sum', 'count'])
    r_e = (ge['sum'] + K * lg_all) / (ge['count'] + K)
    g2 = D.groupby(['e', 't']).y.agg(['sum', 'count']).reset_index()
    par2 = r_e.reindex(g2['e']).to_numpy()
    rt = (g2['sum'].to_numpy() + K * par2) / (g2['count'].to_numpy() + K)
    g2['dev'] = (rt - par2) - (lg_t.reindex(g2['t']).to_numpy() - lg_all)
    return (g.set_index(['e', 'cell', 't'])['dev'], g2.set_index(['e', 't'])['dev'])


def feats(d, code, T1, T2):
    cell, _ = cellbucket(d)
    e = d['pitcher_id'].to_numpy().astype(np.int64)
    x1 = np.nan_to_num(T1.reindex(pd.MultiIndex.from_arrays([e, cell, code])).to_numpy(float))
    x2 = np.nan_to_num(T2.reindex(pd.MultiIndex.from_arrays([e, code])).to_numpy(float))
    g = (d['asof_pitcher_n'].fillna(0).to_numpy() >= N_MIN).astype(float)
    return g * x1, g * x2


def fit_beta(p, y, x1, x2):
    """로짓 shift 의 선형계수를 잔차 최소제곱으로 근사 적합."""
    pc = np.clip(p, EPS, 1 - EPS)
    w = pc * (1 - pc)                       # dp/dz
    X = np.stack([x1 * w, x2 * w], 1)
    r = y - p
    b, *_ = np.linalg.lstsq(X, r, rcond=None)
    return b


print('train.csv 로드...', flush=True)
df = pd.read_csv(f'{LG}/open/data/train.csv')
tr21, tr23 = df[df.season <= 2021], df[df.season <= 2023]

base, DAT = {}, {}
for yr in (2022, 2024):
    yv = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    s = df[df.season == yr].reset_index(drop=True)
    m = np.ones(len(yv), bool)
    s = s[m].reset_index(drop=True)
    p = prod(yr)[m]
    p = logit_add(p, shift_par(s, f'{PAR}/pcxh_parity_le2023', *PCXH_B))
    p = logit_add(p, shift_par(s, f'{PAR}/ctr_parity_le2023', *CTR_B))
    base[yr] = (s, yv[m], p)
    print(f'  베이스 체인 {yr}: skill={skill(p, yv[m]):.2f}', flush=True)

A22 = axes(base[2022][0], np.random.default_rng(11))
A24 = axes(base[2024][0], np.random.default_rng(22))
names = [k for k in A22 if k in A24]
print(f'\n후보 {len(names)}개 (무작위 {N_RANDOM}개 포함)\n', flush=True)

rows, t0 = [], time.time()
for i, nm in enumerate(names, 1):
    c21 = axes(tr21, np.random.default_rng(33))[nm] if nm.startswith('RANDOM') else None
    c23 = axes(tr23, np.random.default_rng(44))[nm] if nm.startswith('RANDOM') else None
    if c21 is None:
        c21 = axes(tr21, np.random.default_rng(1))[nm]
        c23 = axes(tr23, np.random.default_rng(2))[nm]
    T1a, T2a = build(tr21, c21)
    T1b, T2b = build(tr23, c23)
    d22, y22, p22 = base[2022]
    x1, x2 = feats(d22, A22[nm], T1a, T2a)
    b = fit_beta(p22, y22, x1, x2)
    inc22 = skill(logit_add(p22, b[0] * x1 + b[1] * x2), y22) - skill(p22, y22)
    d24, y24, p24 = base[2024]
    z1, z2 = feats(d24, A24[nm], T1b, T2b)
    inc24 = skill(logit_add(p24, SCALE * (b[0] * z1 + b[1] * z2)), y24) - skill(p24, y24)
    b24 = fit_beta(p24, y24, z1, z2)
    rows.append(dict(name=nm, L=int(np.max(A24[nm]) + 1), b1=b[0], b2=b[1],
                     inc22=inc22, inc24=inc24,
                     sign=bool(b[0] * b24[0] > 0),
                     rnd=nm.startswith('RANDOM')))
    print(f'  [{i:>2}/{len(names)}] {nm:<14} 2022 {inc22:+8.2f}   2024 {inc24:+8.2f}'
          f'   ({time.time()-t0:.0f}s)', flush=True)

R = pd.DataFrame(rows)
R.to_csv(f'{LG}/outputs/531_axis3_sweep.csv', index=False)
real, rnd = R[~R.rnd], R[R.rnd]
thr = max(12.0, rnd.inc24.max() if len(rnd) else 0)
print('\n' + '=' * 78)
print(f'[위양성 바닥] 무작위 {len(rnd)}개: 2024 최고 {rnd.inc24.max():+.2f}, '
      f'평균 {rnd.inc24.mean():+.2f}, sd {rnd.inc24.std():.2f}')
print(f'[게이트 임계] max(12.0, 무작위최고) = {thr:+.2f}')
print('=' * 78)
print(f'\n{"후보":<14}{"L":>3}{"2022":>10}{"2024":>10}{"β부호":>7}  판정')
print('-' * 78)
for _, x in real.sort_values('inc24', ascending=False).iterrows():
    ok = x.inc24 > thr and x.inc22 > 0 and x.sign
    print(f'{x["name"]:<14}{x.L:>3}{x.inc22:>+10.2f}{x.inc24:>+10.2f}'
          f'{"O" if x.sign else "X":>7}  {"✅ PASS" if ok else "—"}')
print('-' * 78)
win = real[(real.inc24 > thr) & (real.inc22 > 0) & real.sign]
print(f'\n통과 {len(win)}개' + (f': {list(win.name)}' if len(win) else ' — 정직한 null')
      + f'\n-> {LG}/outputs/531_axis3_sweep.csv')
