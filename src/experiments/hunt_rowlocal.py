"""
hunt_rowlocal.py — 행-로컬 방향 대량 탐색 (규정 안전) + 위양성 바닥 동시 측정

배경: `527` 의 닫힌형 `Δ = (1e5 − s_A)·ρ²` 덕분에 후보 방향 하나를 **1초 이내**로 판정할 수
있다. 그래서 수천 개를 훑을 수 있다. 문제는 **다중검정**이다 — 후보가 2000개면 우연히
게이트를 통과하는 것이 반드시 나온다.

그래서 이 스크립트는 **무작위 대조군(random control)을 같은 파이프라인에 섞어 넣고**,
"진짜 후보 중 최고" 를 "무작위 중 최고" 와 비교한다. 무작위가 같은 수준을 내면 그 발견은
탐색 잡음이다. 이것이 없으면 대량 탐색은 자기기만 장치가 된다.

규정: 후보는 전부 **행 자신의 컬럼에 대한 행 내 산술**이다(`test.csv` 48컬럼).
다른 test 행을 참조하지 않는다. 셀 보정 상수는 **train 폴드에서만** 적합한다.

판정(사전 확정):
  1) 캘리브레이션(1, p−0.5)에 직교화한 뒤의 증분만 인정
  2) 계수는 이전 폴드에서 적합 → 다음 폴드에서 실현된 값만 결과
  3) 3개 전이(2021→2022, 2022→2023R, 2023R→2024) **전부 양수**
  4) LB 최근사 폴드(2022→2023R)에서 **> +12**
  5) 위 4를 통과한 것이 **무작위 대조군 최고치를 넘어설 것**

    python3 harness/hunt_rowlocal.py --out outputs/529_hunt.csv
"""
import argparse, glob, itertools, os, time
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
EPS = 1e-6
SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
FOLDS = [2021, 2022, 2023, 2024]


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


def transfer(fa, fb, va, vb):
    """fa 에서 적합 → fb 에서 실현. 캘리브레이션 직교화 후의 증분을 반환."""
    Xa = np.stack([np.ones_like(fa['p']), fa['p'] - 0.5, va], 1)
    Xb = np.stack([np.ones_like(fb['p']), fb['p'] - 0.5, vb], 1)
    ra, rb = fa['y'] - fa['p'], fb['y'] - fb['p']
    try:
        b3 = np.linalg.lstsq(Xa, ra, rcond=None)[0]
        b2 = np.linalg.lstsq(Xa[:, :2], ra, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    full = skill(np.clip(fb['p'] + Xb @ b3, EPS, 1 - EPS), fb['y'])
    cal = skill(np.clip(fb['p'] + Xb[:, :2] @ b2, EPS, 1 - EPS), fb['y'])
    return full - cal


def build_candidates(sub, rng, n_random):
    """행-로컬 파생량. sub 는 한 폴드의 원본 컬럼 프레임."""
    d = sub
    num = [c for c in d.columns if d[c].dtype.kind in 'if'
           and c not in ('row_id', 'season', 'control_success')]
    out = {}
    b = d.balls_before.fillna(0).values
    s = d.strikes_before.fillna(0).values

    # (a) 카운트 구조 — 명시 피처가 아닌 것
    out['cnt:b+s'] = b + s
    out['cnt:b*s'] = b * s
    out['cnt:2strike'] = (s >= 2).astype(float)
    out['cnt:3ball'] = (b >= 3).astype(float)
    out['cnt:full'] = ((b >= 3) & (s >= 2)).astype(float)
    out['cnt:first'] = ((b == 0) & (s == 0)).astype(float)

    # (b) 범주 상호작용 (행 내 비교만)
    out['cat:hand_same'] = (d.pitcher_hand == d.batter_hand).astype(float).values
    out['cat:same_team'] = (d.pitcher_team_id == d.batter_team_id).astype(float).values
    out['cat:F'] = (d.game_type == 'F').astype(float).values
    out['cat:top'] = (d.top_bottom == 'T').astype(float).values

    # (c) 단항 변환
    for c in num:
        v = d[c].values.astype(float)
        if np.nanstd(v) < 1e-12:
            continue
        out[f'log:{c}'] = np.log1p(np.clip(v, 0, None))
        out[f'sq:{c}'] = v * v

    # (d) 쌍별 곱/차 — 수치 컬럼 전수
    for c1, c2 in itertools.combinations(num, 2):
        v1 = d[c1].values.astype(float); v2 = d[c2].values.astype(float)
        if np.nanstd(v1) < 1e-12 or np.nanstd(v2) < 1e-12:
            continue
        out[f'mul:{c1}*{c2}'] = v1 * v2
        out[f'sub:{c1}-{c2}'] = v1 - v2

    # (e) 🎲 무작위 대조군 — 위양성 바닥 측정용 (같은 파이프라인을 통과시킨다)
    for i in range(n_random):
        out[f'RANDOM:{i:03d}'] = rng.standard_normal(len(d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(LG, 'outputs/529_hunt_rowlocal.csv'))
    ap.add_argument('--random', type=int, default=200, help='무작위 대조군 개수')
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    F, C = {}, {}
    rng = np.random.default_rng(7)
    for yr in FOLDS:
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(float)
        sub = df[df.season == yr].reset_index(drop=True)
        m = (sub.game_type.values == 'R') if yr == 2023 else np.ones(len(y), bool)
        F[yr] = dict(y=y[m], p=prod(yr)[m], m=m)
        # 같은 시드로 폴드마다 생성하되, 무작위 대조군은 폴드별 독립이어야 의미가 있다
        C[yr] = build_candidates(sub, np.random.default_rng(1000 + yr), a.random)
        print(f'fold {yr}: n={m.sum():,}  후보 {len(C[yr]):,}개', flush=True)

    keys = [k for k in C[FOLDS[0]] if all(k in C[y] for y in FOLDS)]
    print(f'\n공통 후보 {len(keys):,}개 (무작위 대조군 {a.random}개 포함) — 판정 시작', flush=True)

    rows, t0 = [], time.time()
    for i, k in enumerate(keys, 1):
        d = []
        ok = True
        for j in range(1, 4):
            ya, yb = FOLDS[j - 1], FOLDS[j]
            va = np.nan_to_num(C[ya][k][F[ya]['m']], nan=0., posinf=0., neginf=0.)
            vb = np.nan_to_num(C[yb][k][F[yb]['m']], nan=0., posinf=0., neginf=0.)
            if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
                ok = False; break
            d.append(transfer(F[ya], F[yb], va, vb))
        if not ok or any(np.isnan(x) for x in d):
            continue
        rows.append(dict(name=k, t21_22=d[0], t22_23R=d[1], t23R_24=d[2],
                         mean=float(np.mean(d)), all_pos=bool(all(x > 0 for x in d)),
                         is_random=k.startswith('RANDOM')))
        if i % 200 == 0:
            print(f'  {i:,}/{len(keys):,}  ({time.time()-t0:.0f}s)', flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    real, rand = R[~R.is_random], R[R.is_random]

    print(f'\n=== 완료: 실후보 {len(real):,} / 무작위 {len(rand):,}  ({time.time()-t0:.0f}s) ===')
    print(f'\n[위양성 바닥] 무작위 대조군 {len(rand)}개의 분포')
    if len(rand):
        print(f'  3전이 전부 양수 비율 : {rand.all_pos.mean()*100:.1f}%  (기대 ~12.5%)')
        print(f'  2022→2023R 최대치    : {rand.t22_23R.max():+.2f}')
        print(f'  3전이 모두 양수인 것 중 2022→2023R 최대: '
              f'{rand[rand.all_pos].t22_23R.max() if rand.all_pos.any() else float("nan"):+.2f}')
    thr = rand[rand.all_pos].t22_23R.max() if len(rand) and rand.all_pos.any() else 0.0

    P = real[real.all_pos].sort_values('t22_23R', ascending=False)
    print(f'\n[실후보] 3전이 전부 양수 = {len(P):,}개 / {len(real):,}')
    print(P.head(15).to_string(index=False))
    win = P[P.t22_23R > max(12.0, thr)]
    print(f'\n최종 게이트(3전이 양수 & 2022→2023R > max(12, 무작위최고={thr:+.2f})) 통과: {len(win)}개')
    if len(win):
        print(win.to_string(index=False))
    else:
        print('  없음 — 정직한 null.')
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
