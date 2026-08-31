"""출하 전 실측 — β 조정과 잔차 후처리를 v37 의 실제 베이스에서 정직하게 잰다.

이전 측정(+5.20 / β 여지 +4.1)은 **B arm 이 빠진 약한 베이스**(skill 857) 위였다.
v37 의 실제 베이스는 A/B 블렌드 + pcxh + ctr (skill 921)이다. 베이스가 강해지면
남는 여지는 줄어든다 — 그래서 출하 전에 제대로 잰다.

정직 규칙:
  β      : 2022 폴드에서 고른 값을 2024 에 적용(2024 격자최적은 **상한 표시용**으로만)
  잔차모델: 2021+2022 학습 → 2023 에서 계수 적합 → 2024 평가
"""
import glob, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, skill  # noqa: E402

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
# 출하값 = pooled x 0.8
PCXH_SHIP = (0.5113952298199526, 1.0317445073629654, 1000)
CTR_SHIP = (-0.59814543382707, -0.7189782550594095, 2000)
PCXH_POOL = (0.5113952298199526 / .8, 1.0317445073629654 / .8)
CTR_POOL = (-0.59814543382707 / .8, -0.7189782550594095 / .8)


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


def logit_add(p, z):
    pc = np.clip(p, EPS, 1 - EPS)
    return np.where(z != 0., 1. / (1. + np.exp(-(np.log(pc / (1 - pc)) + z))), p)


def raw_dev(d, tdir, nmin):
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
    g = (n >= nmin).astype(float)
    return g * np.where(cov, dc, 0.), g * np.where(cov & ~np.isnan(dh), dh, 0.)


tr = pd.read_csv(f'{LG}/open/data/train.csv')
Y, BASE, DEV, D = {}, {}, {}, {}
for yr in (2021, 2022, 2023, 2024):
    d = tr[tr.season == yr].reset_index(drop=True)
    y = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    wa = np.where(d.game_type.astype(str).values == 'F', 0.2, 0.55)
    ab = np.clip(wa * prod(yr) + (1 - wa) * teamB(yr), EPS, 1 - EPS)
    DEV[yr] = (raw_dev(d, f'{PAR}/pcxh_parity_le2023', 1000),
               raw_dev(d, f'{PAR}/ctr_parity_le2023', 2000))
    Y[yr], BASE[yr], D[yr] = y, ab, d


def chain(yr, pb, cb):
    (c1, h1), (c2, h2) = DEV[yr]
    p = logit_add(BASE[yr], pb[0] * c1 + pb[1] * h1)
    return logit_add(p, cb[0] * c2 + cb[1] * h2)


print('=' * 70)
print('[A] β 스케일 — 출하 0.8 이 최적인가? (2022 로 고르고 2024 로 확인)')
print(f'{"scale":>7}{"2022":>11}{"2024":>11}{"2024 Δ":>10}')
print('-' * 70)
s24_ship = skill(chain(2024, PCXH_SHIP, CTR_SHIP), Y[2024])
s22_ship = skill(chain(2022, PCXH_SHIP, CTR_SHIP), Y[2022])
rows = []
for sc in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    pb = (PCXH_POOL[0] * sc, PCXH_POOL[1] * sc, 1000)
    cb = (CTR_POOL[0] * sc, CTR_POOL[1] * sc, 2000)
    a22 = skill(chain(2022, pb, cb), Y[2022]) - s22_ship
    a24 = skill(chain(2024, pb, cb), Y[2024]) - s24_ship
    rows.append((sc, a22, a24))
    print(f'{sc:>7.1f}{a22:>+11.2f}{a24:>+11.2f}{a24:>+10.2f}'
          + ('   <- 출하값' if abs(sc - 0.8) < 1e-9 else ''))
best22 = max(rows, key=lambda r: r[1])
print('-' * 70)
print(f'2022 가 고르는 scale = {best22[0]:.1f}  ->  그 값의 2024 실현 = {best22[2]:+.2f} 폴드'
      f' ≈ {best22[2]*0.76:+.1f} LB')

print('\n' + '=' * 70)
print('[B] 잔차 후처리 — 2021+2022 학습 → 2023 계수 → 2024 평가')
FE = pd.read_parquet(f'{LG}/work/autonomous_arm_c_search/train_features.parquet')
season = tr.season.to_numpy()
ctx = [c for c in FE.columns if c in tr.columns and FE[c].dtype.kind in 'ifb'
       and c not in ('control_success', 'row_id', 'season')]
print(f'  피처 {len(ctx)}개 (test.csv 원본 컬럼만)')
r_tr = np.concatenate([Y[2021] - chain(2021, PCXH_SHIP, CTR_SHIP),
                       Y[2022] - chain(2022, PCXH_SHIP, CTR_SHIP)])
m_tr = np.isin(season, [2021, 2022])
mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                        min_child_samples=200, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=10.0, n_jobs=4, verbose=-1)
mdl.fit(FE.loc[m_tr, ctx], r_tr)
h23 = mdl.predict(FE.loc[season == 2023, ctx])
r23 = Y[2023] - chain(2023, PCXH_SHIP, CTR_SHIP)
beta = float(np.dot(h23, r23) / max(np.dot(h23, h23), 1e-12))
h24 = mdl.predict(FE.loc[season == 2024, ctx])
c24 = chain(2024, PCXH_SHIP, CTR_SHIP)
d24 = skill(np.clip(c24 + beta * h24, EPS, 1 - EPS), Y[2024]) - s24_ship
print(f'  2023 적합 β = {beta:.4f}')
print(f'  2024 실현 Δ = {d24:+.2f} 폴드 ≈ {d24*0.76:+.1f} LB')
print('=' * 70)
print(f'\n합산 기대 = {(best22[2]+d24)*0.76:+.1f} LB   (1113 -> {1113+(best22[2]+d24)*0.76:.0f})')
np.save('resid_beta.npy', np.array([beta]))
mdl.booster_.save_model('resid_model.txt')
print('\n모델 저장: resid_model.txt / resid_beta.npy')
