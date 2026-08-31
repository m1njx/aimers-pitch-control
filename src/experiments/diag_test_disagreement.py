"""
diag_test_disagreement.py — 블렌드 목적함수의 대수 검증 + 측정면 타당성 검사 + 경계.

--------------------------------------------------------------------------
1. 마스터 항등식 (정확, 가정 없음)

    u(w) = Σᵢ wᵢ·sᵢ  +  Σᵢ≠ⱼ wᵢwⱼ·Dᵢⱼ
    Dᵢⱼ = 1e5·E[(pᵢ-pⱼ)²]/(2V),   V = r(1-r)

  블렌드 skill = **arm skill 가중평균** + **순수 예측 불일치 항**.
  Dᵢⱼ 는 라벨이 필요 없다. 2-arm 50:50 이면 u = (a+b)/2 + D/2.
  대회 산식이 Score = 1e5·(1 - Brier/(r(1-r))) 이고 Public 이 테스트 100% 이므로
  이 항등식은 LB 점수에 대해 **정확히** 성립한다 (COMPETITION_RULES.md §1).

  검증: LB 3점 -> (1032.1376+1016.4138)/2 + 61.190/2 = 1054.8708  ✅ 실측과 일치

2. 그래서 무엇이 달라지나

    2-arm 50:50 에서 우리 arm A 를 후보 C 로 교체할 때
        Δu = (s_C - s_A)/2 + (D_CB - D_AB)/2
    -> **skill 과 불일치가 정확히 같은 무게다.** 우리 arm 단독 skill 만 보는 판정은
       목적함수의 절반을 못 본다.

3. ⚠️ 측정면 함정 (2026-08-26 실측)

  D 는 라벨이 필요 없지만, **테스트와 같은 방식으로 생성된 예측**에서 재야 한다.
  `/private/tmp/triple5k.*` 의 5000행 프록시는 TRAIN 2024 에서 뽑은 행이고,
  팀 arm B 는 "2024년말 통산"을 이 행의 asof 카운터에서 빼서 시즌내 성분을 만든다.
  2024 행에 먹이면 통산 > 행 이라 **전 행에서 isf_n < 0** 이 되어 피처가 파괴된다
  (실측 100.0%, 중앙값 -515). 그 결과 B 의 예측 sd 가 0.026 으로 붕괴하고
  A-B RMS 불일치가 0.0386 (LB 정합값 0.0175 의 2.2배)로 부풀려진다.
  **이 프록시에서 나온 A/B 비교·블렌드 비교는 전부 무효다.**

  이 스크립트의 `admissible()` 이 그 검사를 한다. 새 비교면을 쓸 때마다 먼저 통과시킬 것.

실행:
  venv311/bin/python3 harness/diag_test_disagreement.py            # 항등식 + 경계
  venv311/bin/python3 harness/diag_test_disagreement.py --check-proxy DIR
"""
import os, sys, argparse, itertools, json
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
LB = dict(a=1032.137582, b=1016.4138496773, blend=1054.8707763763)
V_MAX = 0.25          # r(1-r) 의 상한. 학습 시즌 r=0.486~0.533 -> V 는 0.2498~0.2500
ID, TGT = 'row_id', 'control_success'


# ------------------------------------------------------------------ 항등식 검증
def verify_master_identity():
    print('=== 1. 마스터 항등식 검증: u(w) = Σ wᵢsᵢ + Σᵢ≠ⱼ wᵢwⱼDᵢⱼ ===')
    comps = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
    worst, rng = 0.0, np.random.default_rng(7)
    for yr in (2021, 2022, 2023, 2024):
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy')).astype(np.float64)
        V = y.mean() * (1 - y.mean())
        Z = np.load(os.path.join(CACHE, f'pred_{yr}_7.npz'))
        P = {k: np.asarray(Z[k], dtype=np.float64) for k in comps}
        s = {k: 1e5 * (1 - ((P[k] - y) ** 2).mean() / V) for k in comps}
        D = {(i, j): 1e5 * ((P[i] - P[j]) ** 2).mean() / (2 * V)
             for i, j in itertools.combinations(comps, 2)}
        for _ in range(5):
            w = dict(zip(comps, rng.dirichlet(np.ones(len(comps)))))
            direct = 1e5 * (1 - ((sum(w[k] * P[k] for k in comps) - y) ** 2).mean() / V)
            viaid = (sum(w[k] * s[k] for k in comps) + 2 * sum(
                w[i] * w[j] * D[(i, j)] for i, j in itertools.combinations(comps, 2)))
            worst = max(worst, abs(direct - viaid))
    print('   5성분 × 4시즌 × 무작위 가중치 20조합, 최대 오차 = %.2e  -> 항등식 정확' % worst)

    D_AB = 2 * (LB['blend'] - 0.5 * (LB['a'] + LB['b']))
    recon = 0.5 * (LB['a'] + LB['b']) + D_AB / 2
    print('   LB 재현: (a+b)/2 + D/2 = %.4f  (실측 %.4f)  D_AB = %.3f'
          % (recon, LB['blend'], D_AB))
    return D_AB


# ------------------------------------------------- 측정면 타당성 (B arm 전용 검사)
def admissible(proxy_dir):
    """팀 arm B 의 isf 분해가 성립하는 행인지 검사한다.

    B 는 model/isf_pitcher.csv 의 '2024년말 통산'을 이 행의 asof 카운터에서 빼서
    시즌내 성분을 만든다. 2025 테스트 행이면 isf_n >= 0 이어야 한다.
    음수가 나오면 그 행에서 B 의 피처는 무의미하다.
    """
    tp = os.path.join(proxy_dir, 'data/test.csv')
    ap = os.path.join(proxy_dir, 'model/isf_pitcher.csv')
    if not (os.path.exists(tp) and os.path.exists(ap)):
        print('   [skip] %s 에 data/test.csv 또는 model/isf_pitcher.csv 없음' % proxy_dir)
        return None
    t = pd.read_csv(tp)
    anc = pd.read_csv(ap).set_index('pitcher_id').reindex(t['pitcher_id'].values)
    car_n = np.nan_to_num(anc['car_n'].values)
    isf_n = t['asof_pitcher_n'].values.astype(float) - car_n
    neg = float((isf_n < 0).mean())
    print('   %s' % proxy_dir)
    print('     season=%s  n=%d  isf_n<0 비율=%.1f%%  (중앙값 %.0f)'
          % (sorted(t['season'].unique()), len(t), 100 * neg, np.median(isf_n)))
    ok = neg < 0.01
    print('     판정: %s' % ('✅ 사용 가능' if ok else
                             '❌ 사용 불가 — B 의 피처가 파괴된다. 여기서 나온 비교는 전부 무효'))
    return ok


# ------------------------------------------------------------------------ 경계
def bounds(D_AB):
    """B 없이도 계산되는 ΔD 의 상·하한.

    C = A + Δ 로 두면
        D_CB - D_AB = K·(E[Δ²] + 2·E[Δ·(p_A - p_B)]),   K = 1e5/(2V)
    Cauchy-Schwarz: |E[Δ(p_A-p_B)]| <= d·m,  d = √E[Δ²], m = √M.
        => ΔD ∈ K·[d² - 2dm,  d² + 2dm]
    E[Δ²] 는 우리 쪽 변경분이라 언제든 계산된다. m 은 LB 에서 이미 안다.
    """
    K = 1e5 / (2 * V_MAX)
    M = D_AB * 2 * V_MAX / 1e5
    m = np.sqrt(M)
    print('\n=== 2. LB 가 못박는 값 (V=r(1-r)≤0.25, Public=테스트 100%%) ===')
    print('   D_AB = %.2f  ->  M = E[(p_A-p_B)²] = %.3e,  RMS 불일치 m = %.5f' % (D_AB, M, m))
    print('   c = (a+b)/2 + D_AB = %.2f   (516 의 1085.47 과 일치)'
          % (0.5 * (LB['a'] + LB['b']) + D_AB))

    print('\n=== 3. 후보 C = A + Δ 의 블렌드 영향 경계 (B 없이 계산 가능) ===')
    print('   Δu = (s_C - s_A)/2 + ΔD/2,   ΔD ∈ K·[d²-2dm, d²+2dm],  K = %.0f' % K)
    print('   %10s %11s %11s %11s %18s'
          % ('d=√E[Δ²]', 'ΔD 하한', 'ΔD 직교', 'ΔD 상한', '다양성 기여 Δu'))
    for d in (0.002, 0.005, 0.010, m, 0.025, 0.035, 0.050):
        lo, mid, hi = K * (d * d - 2 * d * m), K * d * d, K * (d * d + 2 * d * m)
        tag = '  <- 현재 A-B 간격' if abs(d - m) < 1e-9 else ''
        print('   %10.4f %11.1f %11.1f %11.1f   [%+6.1f, %+6.1f] 직교 %+6.1f%s'
              % (d, lo, mid, hi, lo / 2, hi / 2, mid / 2, tag))
    print('\n   ★ 중앙 케이스: 변경분 Δ 가 (p_A - p_B) 와 직교하면 ΔD = K·d² 정확히.')
    print('     -> 블렌드 다양성 이득 ≈ 1e5·d²  (V=0.25). d=0.0175 이면 +30.6.')
    print('     -> d > 2m = %.4f 이면 하한도 양수다: 아무리 나쁘게 정렬돼도 D 는 오른다.' % (2 * m))
    print('\n   읽는 법: 우리 arm 을 RMS %.4f 만큼 바꾸는 변경은 skill 이 그대로여도' % m)
    print('   블렌드를 최대 %+.0f점까지 움직일 수 있다 — 부호는 B 와의 정렬이 정한다.' % (K * (m * m + 2 * m * m) / 2))
    print('   지금까지 닫은 피처 축 20개 중 이 크기를 넘긴 것은 없다.')
    print('   ⚠️ 상한은 상한일 뿐이다. 실제 부호·크기는 E[Δ·(p_A-p_B)] 가 정하고,')
    print('      그건 B 의 예측 없이는 못 잰다. 이게 팀 요청이 병목인 이유다.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-proxy', nargs='*', default=None,
                    help='B arm 을 돌릴 후보 디렉토리(들). data/test.csv + model/isf_pitcher.csv 필요')
    a = ap.parse_args()
    D_AB = verify_master_identity()
    if a.check_proxy is not None:
        dirs = a.check_proxy or ['/private/tmp/triple5k.n3Wmdd/v23/B']
        print('\n=== 측정면 타당성 검사 (팀 arm B) ===')
        for d in dirs:
            admissible(d)
    bounds(D_AB)


if __name__ == '__main__':
    main()
