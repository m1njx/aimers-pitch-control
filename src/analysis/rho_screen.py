"""ρ 후보 스크린 — arm 을 학습하지 않고 "그 정보원에 남은 것이 있는가" 를 직접 잰다.

블렌드 이득은 ρ = corr(체인 잔차, 새 arm 고유방향) 하나로 결정된다(08-30 확인).
그러면 arm 을 만들 필요가 없다. **정보원 X 로 체인 잔차를 직접 예측**해 보면
X 위에 세울 수 있는 어떤 arm 도 그 이상은 못 준다 — 상한을 바로 얻는다.

3분할 중첩(누출 차단):
    2021+2022  → 잔차 모델 학습
    2023       → β 적합 (모델에 out-of-sample)
    2024       → 평가 (모델·β 어느 쪽도 안 본 해)

베이스 잔차는 `prod(yr)`(A 계열 산출물)로 정의한다 — pcxh/ctr 파리티 표는 ≤2023 적합이라
학습연도에 넣으면 누출이 된다. 최종 평가만 **전체 체인(prod+pcxh+ctr) 위에서** 한다.

블록 비교:
  all144   : Gemini 트랙맨 파켓 전체
  phys     : 물리/트랙맨 계열만 (투수 구종별 구속·회전·무브먼트·익스텐션·터널링)
  ctx      : test.csv 원본 문맥 컬럼만 (베이스 모델이 이미 쓰는 것 = 대조군, ~0 이어야 정상)
"""
import glob, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb

LG = os.path.expanduser('~/LG_data')
CACHE = f'{LG}/harness/cache'
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out'
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


def skill(p, y):
    r = y.mean(); return 1e5 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))


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


print('로드...', flush=True)
tr = pd.read_csv(f'{LG}/open/data/train.csv')
FE = pd.read_parquet(f'{LG}/work/autonomous_arm_c_search/train_features.parquet')
assert len(FE) == len(tr), f'{len(FE)} != {len(tr)}'
season = tr.season.to_numpy()

Y, P = {}, {}
for yr in (2021, 2022, 2023, 2024):
    Y[yr] = np.load(f'{CACHE}/y_{yr}.npy').astype(float)
    P[yr] = prod(yr)
    assert len(Y[yr]) == (season == yr).sum()
print('  베이스 skill:', {y: round(skill(P[y], Y[y]), 1) for y in P}, flush=True)

# 블록 정의
drop = {'control_success', 'row_id', 'season'}
allc = [c for c in FE.columns if c not in drop and FE[c].dtype.kind in 'ifb']
PHYS_KEY = ('rel_speed', 'spin', 'ivb', 'hb', 'extension', 'rel_height', 'rel_side',
            'speed_drop', 'tunnel', 'diff_', 'movement', 'vert', 'horz', 'break')
phys = [c for c in allc if any(k in c.lower() for k in PHYS_KEY)]
ctx = [c for c in allc if c in tr.columns]
BLOCKS = {'all144': allc, 'phys': phys, 'ctx(대조군)': ctx}
print({k: len(v) for k, v in BLOCKS.items()}, flush=True)

m_tr = np.isin(season, [2021, 2022]); m_b = season == 2023; m_te = season == 2024
r_tr = np.concatenate([Y[2021] - P[2021], Y[2022] - P[2022]])

# 전체 체인(2024) — 평가 기준선
d24 = tr[m_te].reset_index(drop=True)
chain24 = logit_add(P[2024], pcxh_shift(d24, f'{PAR}/pcxh_parity_le2023', *PCXH_B))
chain24 = logit_add(chain24, pcxh_shift(d24, f'{PAR}/ctr_parity_le2023', *CTR_B))
base24 = skill(chain24, Y[2024])
print(f'\n2024 전체 체인 기준선 = {base24:.2f}\n', flush=True)

print(f'{"블록":<14}{"피처":>6}{"2023 β":>9}{"2024 Δ":>10}{"ρ 추정":>10}  판정')
print('-' * 66)
res = []
for nm, cols in BLOCKS.items():
    t0 = time.time()
    X = FE[cols]
    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                            min_child_samples=200, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=10.0, n_jobs=4, verbose=-1)
    mdl.fit(X[m_tr], r_tr)
    # β 를 2023 에서 적합 (모델에 out-of-sample)
    h23 = mdl.predict(X[m_b])
    r23 = Y[2023] - P[2023]
    beta = float(np.dot(h23, r23) / max(np.dot(h23, h23), 1e-12))
    # 2024 평가: 전체 체인 위에 얹는다
    h24 = mdl.predict(X[m_te])
    got = skill(np.clip(chain24 + beta * h24, EPS, 1 - EPS), Y[2024])
    d = got - base24
    rho = np.sqrt(max(d, 0) / (1e5 - base24))
    ok = d > 12
    res.append((nm, d, rho))
    print(f'{nm:<14}{len(cols):>6}{beta:>9.3f}{d:>+10.2f}{100*rho:>9.2f}%  '
          f'{"✅ PASS" if ok else "—"}   ({time.time()-t0:.0f}s)', flush=True)
print('-' * 66)
best = max(res, key=lambda x: x[1])
print(f'\n최고: {best[0]}  Δ {best[1]:+.2f}  ρ {100*best[2]:.2f}%')
print(f'게이트 +12 에 필요한 ρ = {100*np.sqrt(12/(1e5-base24)):.2f}%')
print(f'\n※ 이 값은 **상한**이다 — 잔차를 직접 회귀했으므로, 같은 피처로 만든 어떤 arm 도 이보다 못하다.')
