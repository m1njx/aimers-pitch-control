"""Arm C 개선 전제 검산 — "단독 573→750 이면 ΔG2 가 +15~25 로 폭발한다" 가 참인가?

블렌드 스킬은 오차벡터의 정확한 2차형식이므로 계산으로 답이 나온다. 튜닝에 시간을 쓰기 전에
**필요한 단독 스킬이 얼마인지** 먼저 못박는다.

반사실 모형: e_C(c) = c · e_C   (오차를 비례 축소 = 같은 모델이 균일하게 더 정확해진 경우)
  - 상관구조를 보존하면서 s_C 를 올린다
  - c→0 이면 p_C→y 라 d_AC 가 오히려 **커진다** → 다양성 손실이 전혀 없는 **낙관적** 모형
  - 즉 여기서 나오는 필요 스킬은 **하한**이다. 실제로는 이보다 더 필요하다.

비교용으로 두 번째 모형도 본다:
  직교성분 보존 축소 — e_C 를 span{e_A,e_B} 성분만 줄이고 직교성분은 유지 (다양성 최대 보존)
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, team_arm, skill, gram, opt  # noqa: E402

LG = os.path.expanduser('~/LG_data')
y = np.load(os.path.join(LG, 'harness/cache/y_2024.npy')).astype(float)
pA, pB = our_arm(), team_arm()
pC = np.load(os.path.join(LG, 'work/arm_c_exports/arm_c_phys_lgb_2024.npy')).astype(float).ravel()
# 게이트와 동일하게 아핀 재보정된 C 를 기준으로 (raw 422 -> 573)
A = np.stack([np.ones_like(pC), pC], 1)
pC = A @ np.linalg.lstsq(A, y, rcond=None)[0]

r = y.mean(); V = r * (1 - r)
eA, eB, eC = pA - y, pB - y, pC - y


def gain(ec):
    """주어진 C 오차벡터에서 3-arm 최적 − 2-arm 최적."""
    E = [eA, eB, ec]
    M3 = np.array([[1e5 * (1 - (E[i] * E[j]).mean() / V) for j in range(3)] for i in range(3)])
    w3, u3 = opt(M3)
    w2, u2 = opt(M3[:2, :2])
    return u3 - u2, w3[2], 1e5 * (1 - (ec * ec).mean() / V)


base_d, base_w, s_now = gain(eC)
print(f'현재 Arm C: 단독 skill {s_now:.2f}   ΔG2 {base_d:+.2f}   w_C {base_w:.3f}')
print(f'대조: A {skill(pA,y):.2f}  B {skill(pB,y):.2f}\n')

# 모형 1 — 비례 축소 (상관구조 보존, 낙관적)
print('=' * 74)
print('모형 1: 오차 비례 축소 (같은 모델이 균일하게 정확해짐)')
print(f'{"c":>6}{"단독 skill":>13}{"d_AC":>9}{"d_BC":>9}{"w_C":>8}{"ΔG2":>10}')
print('-' * 74)
need1 = None
for c in [1.0, .95, .9, .85, .8, .7, .6, .5, .4, .3]:
    ec = c * eC
    d, w, s = gain(ec)
    dAC = np.sqrt(((pA - (y + ec)) ** 2).mean()); dBC = np.sqrt(((pB - (y + ec)) ** 2).mean())
    print(f'{c:>6.2f}{s:>13.1f}{dAC:>9.4f}{dBC:>9.4f}{w:>8.3f}{d:>+10.2f}')
    if need1 is None and d > 12:
        need1 = s

# 모형 2 — 직교성분 보존 (다양성 최대 보존, 더 낙관적)
S = np.stack([eA, eB], 1)
proj = S @ np.linalg.lstsq(S, eC, rcond=None)[0]
orth = eC - proj
print('=' * 74)
print('모형 2: span{e_A,e_B} 성분만 축소, 직교성분 유지 (다양성 최대 보존)')
print(f'  |e_C| 분해: 공유성분 {np.sqrt((proj**2).mean()):.5f}  '
      f'직교성분 {np.sqrt((orth**2).mean()):.5f}  '
      f'(직교 비중 {100*(orth**2).mean()/(eC**2).mean():.1f}%)')
print(f'{"k":>6}{"단독 skill":>13}{"d_AC":>9}{"d_BC":>9}{"w_C":>8}{"ΔG2":>10}')
print('-' * 74)
need2 = None
for k in [1.0, .8, .6, .4, .2, .0]:
    ec = k * proj + orth
    d, w, s = gain(ec)
    dAC = np.sqrt(((pA - (y + ec)) ** 2).mean()); dBC = np.sqrt(((pB - (y + ec)) ** 2).mean())
    print(f'{k:>6.2f}{s:>13.1f}{dAC:>9.4f}{dBC:>9.4f}{w:>8.3f}{d:>+10.2f}')
    if need2 is None and d > 12:
        need2 = s
print('=' * 74)

# ΔG2 = +12 에 필요한 단독 스킬 (모형 1 정밀 탐색)
lo, hi = 0.05, 1.0
for _ in range(40):
    mid = (lo + hi) / 2
    if gain(mid * eC)[0] > 12: lo = mid
    else: hi = mid
d_at, w_at, s_at = gain(lo * eC)
print(f'\n★ ΔG2 = +12.0 에 필요한 Arm C 단독 스킬 (모형 1, 낙관적) : {s_at:.0f}')
print(f'   그때 w_C {w_at:.3f}, 현재 573 대비 필요 상승폭 +{s_at-s_now:.0f}')
print(f'\n   참고: A={skill(pA,y):.0f}  B={skill(pB,y):.0f}  '
      f'-> 필요 스킬이 A/B 를 넘으면 "제3의 arm" 이 아니라 "더 좋은 주모델" 을 요구하는 것')
print(f'\n   모형2(직교성분 100% 보존, 비현실적 상한)에서의 필요 스킬: '
      f'{need2 if need2 else "구간 내 미달성"}')
