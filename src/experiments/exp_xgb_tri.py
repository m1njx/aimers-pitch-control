"""
exp_xgb_tri.py — "XGBoost 재합류로 Tri-GBDT" 제안의 캐시 기반 상한 계산.

제안: LGBM/CatBoost만 쓰고 있으니 XGBoost(tree_method='hist')를 추가하고
      GBDT 서브가중치를 LGBM .50 / CatBoost .30 / XGB .20 으로 탐색하자.

전제 검증 결과: XGBoost 는 이미 프로덕션 arm 의 3번째 GBDT 성분이다
      (build_cache.py L239 tree_method='hist', 5시드; script.py W_XGB_BIN=0.08).
      따라서 "추가"할 것은 없고, 남은 자유도는 **서브가중치 재탐색뿐**이다.
      이 스크립트는 그 재탐색의 상한을 재학습 0 으로 계산한다.

착수 전 확정 기준 (pre-registered):
  C1 프로덕션 arm 에 tree_method='hist' XGBClassifier 5시드가 이미 있으면
     제안의 "추가" 전제는 거짓으로 확정한다.
  C2 GBDT 2-심플렉스 위에서 **폴드별 in-sample 최적**(즉 오라클 선택,
     실제 달성 불가능한 낙관적 상한)을 잡아도 블렌드 Δu 가 inner 3폴드
     평균 +12점(LB 노이즈 바닥)을 못 넘으면 REJECT, 재학습 금지.
  C3 5성분 전체 심플렉스의 블렌드 최적 xgb 유효가중치가 inner 3폴드 중
     2폴드 이상에서 PROD 0.032 보다 낮으면 xgb 확대는 방향 자체가 반대다.
  C4 제안 사양(.50/.30/.20)의 paired 15셀(3폴드×5시드) 델타 부호가
     일관되게 음수면 그것만으로 REJECT 근거로 충분하다.

주의: 로컬 절대 skill 은 LB 와 역상관이다(dacon-local-harness-invalid).
      절대값이 아니라 **부호·방향·상한의 크기**만 읽는다.

실행: venv311/bin/python3 harness/exp_xgb_tri.py
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize
from scipy import stats

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
from evaluate import PROD, predict  # noqa: E402

CACHE = os.path.join(LG, 'harness/cache')
INNER = [2021, 2022, 2023]
OUTER = 2024
SEEDS = [7, 123, 2025, 31415, 8675309]
COMPS = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
EPS = 1e-6

V_PROD = np.array([0.40 * 0.20, 0.40 * 0.72, 0.40 * 0.08, 0.20, 0.40])
S_PROD = np.array([-0.007, -0.008, -0.006, 0.0, 0.0])

# 제안 사양: GBDT 서브가중치 .50/.30/.20 (top-level .40/.40/.20 은 유지)
PROPOSED = dict(w_lgb=0.50, w_cb=0.30, w_xgb=0.20)


def sk(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


def load(y):
    yv = np.load(os.path.join(CACHE, f'y_{y}.npy'))
    bag = []
    for s in SEEDS:
        f = os.path.join(CACHE, f'pred_{y}_{s}.npz')
        if os.path.exists(f):
            bag.append(dict(np.load(f)))
    return yv, bag


def arm5(v, bag, scale=1.10, shift=-0.0045192086):
    """5성분 유효가중치 v -> 배깅 arm. v=V_PROD 면 predict(PROD,.) 와 동일."""
    out = []
    for P in bag:
        raw = sum(w * np.clip(P[c] + s, EPS, 1 - EPS)
                  for w, s, c in zip(v, S_PROD, COMPS))
        out.append(np.clip(0.5 + scale * (raw - 0.5) + shift, EPS, 1 - EPS))
    return np.mean(out, axis=0)


def pseudo_b(bag):
    """B축 CatBoost arm 대역폭 스탠드인 (exp_partner_dedup.py 와 동일)."""
    out = []
    for P in bag:
        raw = np.clip(P['cb_bin'] - 0.008, EPS, 1 - EPS)
        out.append(np.clip(0.5 + 1.10 * (raw - 0.5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(out, axis=0)


def bagged_cfg(cfg, bag):
    c = dict(PROD)
    c.update(cfg)
    return np.mean([predict(c, P) for P in bag], axis=0)


def opt_gbdt(bag, yv, V, pB=None):
    """GBDT 2-심플렉스(w_lgb,w_cb,w_xgb) 만 최적화. top-level/캘리브는 PROD 고정.
    이것이 제안이 제시한 탐색 공간의 정확한 상위집합이다."""
    def neg(z):
        w = np.abs(z)
        w = w / w.sum()
        p = bagged_cfg(dict(w_lgb=w[0], w_cb=w[1], w_xgb=w[2]), bag)
        if pB is not None:
            p = 0.5 * (p + pB)
        return -sk(p, yv, V)
    r = minimize(neg, np.array([0.20, 0.72, 0.08]), method='Nelder-Mead',
                 options=dict(maxiter=3000, xatol=1e-6, fatol=1e-5))
    w = np.abs(r.x)
    return w / w.sum(), -r.fun


def opt5(bag, yv, V, pB=None):
    def neg(v):
        v = np.abs(v)
        v = v / v.sum() * V_PROD.sum()
        p = arm5(v, bag)
        if pB is not None:
            p = 0.5 * (p + pB)
        return -sk(p, yv, V)
    r = minimize(neg, V_PROD.copy(), method='Nelder-Mead',
                 options=dict(maxiter=6000, xatol=1e-6, fatol=1e-5))
    v = np.abs(r.x)
    return v / v.sum() * V_PROD.sum(), -r.fun


def main():
    print('=' * 92)
    print('519  XGB Tri-GBDT 제안 — 캐시 기반 상한 (재학습 0)')
    print('=' * 92)

    # ---------- S1. 성분 상관 (중복도) ----------
    print('\n[S1] GBDT 3성분 상관 — "3대 축 시너지" 전제의 사실 확인 (fold 2023, seed 7)')
    P = dict(np.load(os.path.join(CACHE, 'pred_2023_7.npz')))
    for a in range(3):
        for b in range(a + 1, 3):
            r = np.corrcoef(P[COMPS[a]].astype(np.float64),
                            P[COMPS[b]].astype(np.float64))[0, 1]
            print(f'     corr({COMPS[a]:>8}, {COMPS[b]:>8}) = {r:.4f}')

    # ---------- S2. 제안 사양 직접 채점 (paired 15셀) ----------
    print('\n[S2] 제안 사양 .50/.30/.20  vs  PROD .20/.72/.08   — paired 폴드×시드 원시 셀')
    print(f'     {"fold":>6} {"seed":>9} {"arm PROD":>10} {"arm PROP":>10} {"Δs":>9} '
          f'{"blend PROD":>11} {"blend PROP":>11} {"Δu":>9}')
    cells_ds, cells_du = [], []
    for fold in INNER:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB_all = pseudo_b(bag)
        for i, s in enumerate(SEEDS):
            one = [bag[i]]
            pBs = pseudo_b(one)
            a0 = bagged_cfg({}, one)
            a1 = bagged_cfg(PROPOSED, one)
            s0, s1 = sk(a0, yv, V), sk(a1, yv, V)
            u0 = sk(0.5 * (a0 + pBs), yv, V)
            u1 = sk(0.5 * (a1 + pBs), yv, V)
            cells_ds.append(s1 - s0)
            cells_du.append(u1 - u0)
            print(f'     {fold:>6} {s:>9} {s0:>10.1f} {s1:>10.1f} {s1-s0:>9.2f} '
                  f'{u0:>11.1f} {u1:>11.1f} {u1-u0:>9.2f}')
        _ = pB_all
    ds = np.array(cells_ds)
    du = np.array(cells_du)
    t_ds = stats.ttest_1samp(ds, 0)
    t_du = stats.ttest_1samp(du, 0)
    print(f'\n     Δs  mean={ds.mean():+8.2f}  sd={ds.std(ddof=1):6.2f}  '
          f't={t_ds.statistic:+6.2f}  p={t_ds.pvalue:.2e}  '
          f'음수셀 {int((ds<0).sum())}/{len(ds)}')
    print(f'     Δu  mean={du.mean():+8.2f}  sd={du.std(ddof=1):6.2f}  '
          f't={t_du.statistic:+6.2f}  p={t_du.pvalue:.2e}  '
          f'음수셀 {int((du<0).sum())}/{len(du)}')

    # 프로덕션 동일 배깅(5시드)으로도 한 번
    print('\n     [프로덕션 동일 5시드 배깅]')
    bag_ds, bag_du = [], []
    for fold in INNER + [OUTER]:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB = pseudo_b(bag)
        a0, a1 = bagged_cfg({}, bag), bagged_cfg(PROPOSED, bag)
        s0, s1 = sk(a0, yv, V), sk(a1, yv, V)
        u0 = sk(0.5 * (a0 + pB), yv, V)
        u1 = sk(0.5 * (a1 + pB), yv, V)
        tag = 'inner' if fold in INNER else 'OUTER'
        print(f'     {fold} ({tag})  Δs={s1-s0:+8.2f}   Δu={u1-u0:+8.2f}')
        if fold in INNER:
            bag_ds.append(s1 - s0)
            bag_du.append(u1 - u0)
    print(f'     inner 3폴드 평균  Δs={np.mean(bag_ds):+8.2f}   Δu={np.mean(bag_du):+8.2f}')

    # ---------- S3. GBDT 서브가중치 오라클 상한 ----------
    print('\n[S3] GBDT 2-심플렉스 in-sample 오라클 상한 (폴드별 최적 = 달성 불가능한 낙관 상한)')
    print(f'     {"fold":>6} {"목적":>7} {"w_lgb":>7} {"w_cb":>7} {"w_xgb":>7} '
          f'{"유효w_xgb":>10} {"ceil":>10} {"Δ vs PROD":>10}')
    solo_ceil, team_ceil, xgb_w_team = [], [], []
    for fold in INNER:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB = pseudo_b(bag)
        a0 = bagged_cfg({}, bag)
        s0 = sk(a0, yv, V)
        u0 = sk(0.5 * (a0 + pB), yv, V)

        w_s, c_s = opt_gbdt(bag, yv, V, None)
        w_t, c_t = opt_gbdt(bag, yv, V, pB)
        solo_ceil.append(c_s - s0)
        team_ceil.append(c_t - u0)
        xgb_w_team.append(0.40 * w_t[2])
        print(f'     {fold:>6} {"solo s":>7} {w_s[0]:>7.3f} {w_s[1]:>7.3f} {w_s[2]:>7.3f} '
              f'{0.40*w_s[2]:>10.4f} {c_s:>10.1f} {c_s-s0:>+10.2f}')
        print(f'     {fold:>6} {"blend u":>7} {w_t[0]:>7.3f} {w_t[1]:>7.3f} {w_t[2]:>7.3f} '
              f'{0.40*w_t[2]:>10.4f} {c_t:>10.1f} {c_t-u0:>+10.2f}')
    print(f'\n     PROD 유효 w_xgb = {V_PROD[2]:.4f}')
    print(f'     inner 평균 오라클 상한   solo Δs = {np.mean(solo_ceil):+8.2f}')
    print(f'     inner 평균 오라클 상한  blend Δu = {np.mean(team_ceil):+8.2f}   <-- C2 판정 대상')

    # ---------- S4. 5성분 전체 심플렉스 (exp_partner_dedup 재현·확장) ----------
    print('\n[S4] 5성분 전체 심플렉스 블렌드 최적 — xgb 유효가중치의 방향 (C3)')
    print(f'     {"fold":>6} {"목적":>7} ' + ' '.join(f'{c:>9}' for c in COMPS) + f' {"Δ":>9}')
    xgb5_solo, xgb5_team, ceil5 = [], [], []
    for fold in INNER:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB = pseudo_b(bag)
        a0 = bagged_cfg({}, bag)
        s0 = sk(a0, yv, V)
        u0 = sk(0.5 * (a0 + pB), yv, V)
        v_s, c_s = opt5(bag, yv, V, None)
        v_t, c_t = opt5(bag, yv, V, pB)
        xgb5_solo.append(v_s[2])
        xgb5_team.append(v_t[2])
        ceil5.append(c_t - u0)
        print(f'     {fold:>6} {"solo s":>7} ' + ' '.join(f'{x:>9.4f}' for x in v_s) +
              f' {c_s-s0:>+9.2f}')
        print(f'     {fold:>6} {"blend u":>7} ' + ' '.join(f'{x:>9.4f}' for x in v_t) +
              f' {c_t-u0:>+9.2f}')
    print(f'\n     PROD 유효가중치      ' + ' '.join(f'{x:>9.4f}' for x in V_PROD))
    print(f'     xgb 유효w  solo 평균={np.mean(xgb5_solo):.4f}  '
          f'blend 평균={np.mean(xgb5_team):.4f}  (PROD {V_PROD[2]:.4f})')
    print(f'     blend 목적 하에서 xgb < PROD 인 폴드: '
          f'{sum(1 for x in xgb5_team if x < V_PROD[2])}/3')
    print(f'     5성분 오라클 블렌드 상한 Δu = {np.mean(ceil5):+8.2f}')

    # ---------- S5. xgb 제거 vs 배증 감도 ----------
    print('\n[S5] xgb 감도 — 제거(0) / PROD(0.08) / 배증(0.16) / 제안(0.20), 나머지는 비율 유지')
    print(f'     {"fold":>6} {"w_xgb=0":>10} {"0.08(PROD)":>12} {"0.16":>10} {"0.20":>10}   (블렌드 u)')
    for fold in INNER:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB = pseudo_b(bag)
        row = []
        for wx in [0.0, 0.08, 0.16, 0.20]:
            rest = 1.0 - wx
            wl, wc = 0.20 / 0.92 * rest, 0.72 / 0.92 * rest
            p = bagged_cfg(dict(w_lgb=wl, w_cb=wc, w_xgb=wx), bag)
            row.append(sk(0.5 * (p + pB), yv, V))
        print(f'     {fold:>6} {row[0]:>10.1f} {row[1]:>12.1f} {row[2]:>10.1f} {row[3]:>10.1f}')

    # ---------- S6. "더 좋은 xgb 를 재학습하면?" 의 대체 상한 ----------
    # xgb_bin 슬롯에 점점 더 강한 대체물을 꽂아 블렌드 Δu 를 잰다.
    #   cb    : 우리 최고 GBDT 성분 수준의 xgb (현실적 최상)
    #   arm   : arm 전체와 동급 품질의 xgb (비현실적)
    #   oracle: 완벽한 예측 (물리적 상한)
    print('\n[S6] xgb 슬롯 대체 상한 — "재학습해서 더 좋은 xgb 를 얻으면?" (블렌드 Δu)')
    print(f'     {"fold":>6} {"w_xgb":>7} {"cb급":>10} {"arm급":>10} {"oracle":>12}')
    for fold in INNER:
        yv, bag = load(fold)
        V = yv.mean() * (1 - yv.mean())
        pB = pseudo_b(bag)
        for wx, wl, wc in [(0.08, 0.20, 0.72), (0.20, 0.50, 0.30)]:
            base = bagged_cfg(dict(w_lgb=wl, w_cb=wc, w_xgb=wx), bag)
            u_base = sk(0.5 * (base + pB), yv, V)
            row = []
            for repl in ['cb', 'arm', 'oracle']:
                subs = []
                for Pm in bag:
                    Q = dict(Pm)
                    if repl == 'cb':
                        Q['xgb_bin'] = Pm['cb_bin']
                    elif repl == 'arm':
                        Q['xgb_bin'] = predict(PROD, Pm)
                    else:
                        Q['xgb_bin'] = np.clip(yv, EPS, 1 - EPS)
                    subs.append(predict(dict(PROD, w_lgb=wl, w_cb=wc, w_xgb=wx), Q))
                p = np.mean(subs, axis=0)
                row.append(sk(0.5 * (p + pB), yv, V) - u_base)
            print(f'     {fold:>6} {wx:>7.2f} {row[0]:>+10.2f} {row[1]:>+10.2f} '
                  f'{row[2]:>+12.1f}')

    print('\n' + '=' * 92)


if __name__ == '__main__':
    main()
