"""pcxh 이득의 독립 재현 — 2024 를 전혀 보지 않은 조합으로.

팀 파이프라인의 주장: pcxh(투수×카운트셀×타자손 잔차 룩업)가 2024 에서 +20.2 REAL.
내 기억에는 "투수×카운트 상호작용 = 0" 이 적혀 있어 정면으로 충돌한다. 직접 잰다.

누출 차단:
  - 테이블은 `pcxh_parity_le2023` (2023 까지만으로 적합, 2024 라벨 미열람)
  - 베타는 `beta_fold2022` (2022 폴드에서 적합, 2024 미열람) × scale 0.8
  - 베이스 예측은 **내 하네스 캐시의 내 산출물**(팀 체인이 아님) — 완전 독립 재현
  - 평가는 2024 전체, 그리고 2023R (이쪽은 테이블이 2023 을 봤으므로 참고용으로만)

즉 여기서 나오는 2024 숫자는 어느 경로로도 2024 라벨을 보지 않았다.
"""
import glob, json, os
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
# 출하된 파라미터 (2024 를 보지 않고 적합된 것만 사용)
BETA = {'pcxh': (0.508947879968906, 1.27948246622023, 1000),      # beta_fold2022
        'ctr':  (-0.5180599172482175, -0.9338837106747617, 2000)}  # beta_fold2022
SCALE = 0.8


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


def shift(df, tdir, beta_cell, beta_hand, n_min):
    cell = pd.read_csv(os.path.join(tdir, 'pcxh_cell.csv')).set_index(
        ['pitcher_id', 'cell', 'bh'])['dev_cell']
    hand = pd.read_csv(os.path.join(tdir, 'pcxh_hand.csv')).set_index(
        ['pitcher_id', 'ph', 'bh'])['dev_hand']
    pid = df['pitcher_id'].to_numpy().astype(np.int64)
    b = df['balls_before'].fillna(0).to_numpy().astype(np.int64)
    s = df['strikes_before'].fillna(0).to_numpy().astype(np.int64)
    c = np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2)
    bh = np.clip(df['batter_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    ph = np.clip(df['pitcher_hand'].fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
    n = df['asof_pitcher_n'].fillna(0).to_numpy().astype(float)
    dc = cell.reindex(pd.MultiIndex.from_arrays([pid, c, bh])).to_numpy(float)
    dh = hand.reindex(pd.MultiIndex.from_arrays([pid, ph, bh])).to_numpy(float)
    cov = ~np.isnan(dc)
    dc = np.where(cov, dc, 0.); dh = np.where(cov & ~np.isnan(dh), dh, 0.)
    return (n >= n_min) * (beta_cell * dc + beta_hand * dh), cov, (n >= n_min)


def apply_shift(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    q = 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z)))
    return np.where(z != 0., q, p)


df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
print(f'{"폴드":<10}{"n":>10}{"적용률":>9}{"base":>11}{"+pcxh":>11}{"Δ":>9}{"+ctr":>11}{"Δ":>8}')
print('-' * 80)
for yr in (2024, 2023):
    y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(float)
    sub = df[df.season == yr].reset_index(drop=True)
    m = (sub.game_type.values == 'R') if yr == 2023 else np.ones(len(y), bool)
    sub = sub[m].reset_index(drop=True)
    yy, p0 = y[m], prod(yr)[m]

    bc, bh_, nm = BETA['pcxh']
    z1, cov, gate = shift(sub, f'{PAR}/pcxh_parity_le2023', bc * SCALE, bh_ * SCALE, nm)
    p1 = apply_shift(p0, z1)
    bc2, bh2, nm2 = BETA['ctr']
    z2, _, _ = shift(sub, f'{PAR}/ctr_parity_le2023', bc2 * SCALE, bh2 * SCALE, nm2)
    p2 = apply_shift(p1, z2)

    s0, s1, s2 = skill(p0, yy), skill(p1, yy), skill(p2, yy)
    tag = f'{yr}R' if yr == 2023 else str(yr)
    print(f'{tag:<10}{len(yy):>10,}{100*(cov&gate).mean():>8.1f}%'
          f'{s0:>11.2f}{s1:>11.2f}{s1-s0:>+9.2f}{s2:>11.2f}{s2-s1:>+8.2f}')

print('\n2024 는 테이블(≤2023)·베타(fold2022) 어느 쪽도 2024 라벨을 보지 않음 -> 정직한 전방 측정')
print('2023R 은 테이블이 2023 을 포함하므로 낙관 편향 -> 참고용')
