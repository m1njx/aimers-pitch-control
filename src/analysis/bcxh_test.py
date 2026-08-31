"""bcxh — 타자 대칭 테이블 시험 (batter × count-cell × pitcher-hand).

팀 파이프라인의 엔티티 잔차 룩업 는 **투수 축만** 쓴다. 대칭인 타자 축은 미탐색이다.
같은 레시피(이중 중심화 + EB 수축 + asof 게이트 + 교차폴드 베타)를 타자에 적용해
**pcxh + ctr 을 이미 얹은 뒤의 증분**을 잰다. 증분이 0 이면 정직한 null 이다.

누출 차단 (팀 프로토콜 그대로):
  - 2022 판정: 테이블 ≤2021 로 적합 → 2022 에서 베타 적합
  - 2024 판정: 테이블 ≤2023 로 적합 → **2022 에서 적합한 베타**로 2024 에 적용
    즉 2024 는 테이블·베타 어느 쪽으로도 라벨이 새지 않는다.
  - 베이스는 내 산출물 + pcxh(≤2023) + ctr(≤2023) 체인

사전 확정 판정기준 (결과 보기 전 고정):
  G1  2024 증분 > +12.0  (내 노이즈 바닥 기준)
  G2  2022 증분 > 0      (양 폴드 부호 일치 — "2024만 REAL"은 전이 실패)
  G3  베타 부호가 두 폴드에서 일치

[기여 구분] 여기서 다루는 엔티티 잔차 룩업 기법 자체는 **팀 공동 파트에서 도입**됐다.
이 스크립트는 그 기법에 대해 내가 수행한 **독립 검증·확장 시도**다.
"""
import glob, os
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
K = 100.0
N_MIN = 1000
SCALE = 0.8
PCXH_B = (0.508947879968906, 1.27948246622023, 1000)
CTR_B = (-0.5180599172482175, -0.9338837106747617, 2000)


def prod(yr):
    ps = []
    for f in sorted(glob.glob(os.path.join(CACHE, f'pred_{yr}_*.npz'))):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)


def skill(p, y):
    r = y.mean()
    return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


def keys(d, ent, hand):
    b = d['balls_before'].fillna(0).to_numpy().astype(np.int64)
    s = d['strikes_before'].fillna(0).to_numpy().astype(np.int64)
    cell = np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2)
    bucket = np.sign(s - b).astype(np.int64) + 1
    h = np.clip(d[hand].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    return d[ent].to_numpy().astype(np.int64), cell, bucket, h


def build(tr, ent, hand):
    """dev = EB[r(ent,cell,h) | parent r(ent,bucket)] - r(ent,bucket)
             - (league r(cell,h) - league r(bucket))"""
    e, cell, bucket, h = keys(tr, ent, hand)
    y = tr['control_success'].to_numpy().astype(float)
    D = pd.DataFrame(dict(e=e, cell=cell, bucket=bucket, h=h, y=y))
    lg_cell = D.groupby(['cell', 'h']).y.mean()
    lg_buck = D.groupby('bucket').y.mean()
    g_eb = D.groupby(['e', 'bucket']).y.agg(['sum', 'count'])
    r_eb = ((g_eb['sum'] + K * lg_buck.reindex(g_eb.index.get_level_values('bucket')).to_numpy())
            / (g_eb['count'] + K))
    g = D.groupby(['e', 'cell', 'h', 'bucket']).y.agg(['sum', 'count']).reset_index()
    parent = r_eb.reindex(pd.MultiIndex.from_arrays(
        [g['e'], g['bucket']])).to_numpy()
    r_cell = (g['sum'].to_numpy() + K * parent) / (g['count'].to_numpy() + K)
    lgc = lg_cell.reindex(pd.MultiIndex.from_arrays([g['cell'], g['h']])).to_numpy()
    lgb = lg_buck.reindex(g['bucket']).to_numpy()
    g['dev'] = (r_cell - parent) - (lgc - lgb)
    return g.set_index(['e', 'cell', 'h'])['dev']


def dev_of(d, tbl, ent, hand):
    e, cell, _, h = keys(d, ent, hand)
    v = tbl.reindex(pd.MultiIndex.from_arrays([e, cell, h])).to_numpy(float)
    return np.nan_to_num(v, nan=0.0)


def shift_par(d, tdir, bc, bh_, nmin):
    cell = pd.read_csv(f'{tdir}/pcxh_cell.csv').set_index(['pitcher_id', 'cell', 'bh'])['dev_cell']
    hand = pd.read_csv(f'{tdir}/pcxh_hand.csv').set_index(['pitcher_id', 'ph', 'bh'])['dev_hand']
    pid, c, _, bh = keys(d, 'pitcher_id', 'batter_hand')
    ph = np.clip(d['pitcher_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    n = d['asof_pitcher_n'].fillna(0).to_numpy().astype(float)
    dc = cell.reindex(pd.MultiIndex.from_arrays([pid, c, bh])).to_numpy(float)
    dh = hand.reindex(pd.MultiIndex.from_arrays([pid, ph, bh])).to_numpy(float)
    cov = ~np.isnan(dc)
    dc = np.where(cov, dc, 0.); dh = np.where(cov & ~np.isnan(dh), dh, 0.)
    return (n >= nmin) * (bc * SCALE * dc + bh_ * SCALE * dh)


def logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    q = 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z)))
    return np.where(z != 0., q, p)


def fold(df, yr):
    y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(float)
    sub = df[df.season == yr].reset_index(drop=True)
    m = (sub.game_type.values == 'R') if yr == 2023 else np.ones(len(y), bool)
    return sub[m].reset_index(drop=True), y[m], prod(yr)[m]


print('train.csv 로드...', flush=True)
df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))

# 베이스 체인 (내 산출물 + pcxh + ctr, 둘 다 ≤2023 파리티 테이블)
base = {}
for yr in (2022, 2024):
    d, y, p = fold(df, yr)
    p = logit_add(p, shift_par(d, f'{PAR}/pcxh_parity_le2023', *PCXH_B))
    p = logit_add(p, shift_par(d, f'{PAR}/ctr_parity_le2023', *CTR_B))
    base[yr] = (d, y, p)
    print(f'  베이스 체인 {yr}: skill={skill(p, y):.2f}', flush=True)

print('\n타자 테이블 구축 (≤2021, ≤2023)...', flush=True)
T21 = build(df[df.season <= 2021], 'batter_id', 'pitcher_hand')
T23 = build(df[df.season <= 2023], 'batter_id', 'pitcher_hand')
print(f'  ≤2021 {len(T21):,}셀 / ≤2023 {len(T23):,}셀', flush=True)

# 2022 에서 베타 적합 (테이블 ≤2021)
d22, y22, p22 = base[2022]
dev22 = dev_of(d22, T21, 'batter_id', 'pitcher_hand')
g22 = (d22['asof_batter_n'].fillna(0).to_numpy() >= N_MIN).astype(float)
x22 = g22 * dev22
grid = np.arange(-3.0, 3.01, 0.05)
sc = [skill(logit_add(p22, b * x22), y22) for b in grid]
beta22 = float(grid[int(np.argmax(sc))])
inc22 = max(sc) - skill(p22, y22)

# 2024 에 적용 (테이블 ≤2023, 베타는 2022 적합 × 0.8)
d24, y24, p24 = base[2024]
dev24 = dev_of(d24, T23, 'batter_id', 'pitcher_hand')
g24 = (d24['asof_batter_n'].fillna(0).to_numpy() >= N_MIN).astype(float)
x24 = g24 * dev24
inc24 = skill(logit_add(p24, beta22 * SCALE * x24), y24) - skill(p24, y24)
# 참고: 2024 자체 최적 베타 (누출, 상한 확인용)
sc24 = [skill(logit_add(p24, b * x24), y24) for b in grid]
beta24, cap24 = float(grid[int(np.argmax(sc24))]), max(sc24) - skill(p24, y24)

print('\n' + '=' * 68)
print(f'{"항목":<34}{"2022":>12}{"2024":>12}')
print('-' * 68)
print(f'{"적용률 (asof_batter_n>=1000)":<34}{100*g22.mean():>11.1f}%{100*g24.mean():>11.1f}%')
print(f'{"폴드내 최적 베타":<34}{beta22:>12.2f}{beta24:>12.2f}')
print(f'{"증분 (폴드내 최적 = 상한)":<34}{inc22:>+12.2f}{cap24:>+12.2f}')
print(f'{"증분 (2022 베타 x0.8 전이) ★":<34}{"—":>12}{inc24:>+12.2f}')
print('=' * 68)
print('\n[사전 확정 기준]')
print(f'  G1 2024 전이 증분 > +12.0 : {inc24:+.2f}  ->  {"PASS" if inc24 > 12 else "FAIL"}')
print(f'  G2 2022 증분 > 0          : {inc22:+.2f}  ->  {"PASS" if inc22 > 0 else "FAIL"}')
print(f'  G3 베타 부호 일치         : {beta22:+.2f} / {beta24:+.2f}  ->  '
      f'{"PASS" if beta22 * beta24 > 0 else "FAIL"}')
ok = inc24 > 12 and inc22 > 0 and beta22 * beta24 > 0
print(f'\n판정: {"✅ 통과 — 실제 후보" if ok else "❌ 기각 (정직한 null)"}')
