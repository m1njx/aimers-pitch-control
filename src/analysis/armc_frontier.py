"""Arm C 변형 11종의 실측 프론티어 — 단독 스킬 ↔ 다양성 ↔ ΔG2 의 진짜 관계.

이론 반사실(오차 비례축소)은 다양성이 오히려 커지는 낙관 모형이다. 실제 모델을 개선하면
예측이 A/B 합의 쪽으로 끌려가 다양성이 깎인다. 그 교환비는 **이미 만들어진 변형들**이
말해 준다 — arm_c_exports/ 에 2024 예측이 11종 있다.

각 변형에 대해: 단독 skill(아핀 재보정), d_AC, d_BC, 3-arm ΔG2, w_C 를 재고
(skill, ΔG2) 산점을 만들어 "스킬을 올리면 ΔG2 가 따라 오르는가" 를 직접 본다.
"""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/LG_data/harness'))
from gate_newarm import our_arm, team_arm, skill, opt  # noqa: E402

LG = os.path.expanduser('~/LG_data')
EXP = os.path.join(LG, 'work/arm_c_exports')
y = np.load(os.path.join(LG, 'harness/cache/y_2024.npy')).astype(float)
pA, pB = our_arm(), team_arm()
r = y.mean(); V = r * (1 - r)
eA, eB = pA - y, pB - y


def load(path):
    if path.endswith('.npy'):
        v = np.load(path).astype(float).ravel()
    else:
        d = pd.read_csv(path)
        col = [c for c in d.columns if d[c].dtype.kind == 'f']
        if not col:
            return None
        v = d[col[-1]].to_numpy(float).ravel()
    return v if len(v) == len(y) else None


def evaluate(p):
    A = np.stack([np.ones_like(p), p], 1)
    pc = A @ np.linalg.lstsq(A, y, rcond=None)[0]        # 아핀 재보정
    ec = pc - y
    E = [eA, eB, ec]
    M = np.array([[1e5 * (1 - (E[i] * E[j]).mean() / V) for j in range(3)] for i in range(3)])
    w3, u3 = opt(M); _, u2 = opt(M[:2, :2])
    return dict(s_raw=skill(p, y), s=1e5 * (1 - (ec * ec).mean() / V),
                d_AC=float(np.sqrt(((pA - pc) ** 2).mean())),
                d_BC=float(np.sqrt(((pB - pc) ** 2).mean())),
                dG2=u3 - u2, w_C=w3[2])


rows = []
for f in sorted(glob.glob(os.path.join(EXP, '*2024*'))):
    if f.endswith('.npy') and os.path.exists(f[:-4] + '.csv'):
        continue                                          # csv 와 중복
    p = load(f)
    if p is None:
        continue
    rows.append(dict(name=os.path.basename(f).replace('_2024.csv', '').replace('_2024.npy', ''),
                     **evaluate(p)))

R = pd.DataFrame(rows).sort_values('s', ascending=False).reset_index(drop=True)
print(f'A={skill(pA,y):.1f}  B={skill(pB,y):.1f}  d_AB={np.sqrt(((pA-pB)**2).mean()):.4f}   '
      f'2-arm 베이스=879.09\n')
print(f'{"변형":<18}{"raw":>8}{"아핀skill":>11}{"d_AC":>8}{"d_BC":>8}{"w_C":>7}{"ΔG2":>9}{"G1":>5}{"G2":>5}')
print('-' * 80)
for _, x in R.iterrows():
    g1 = 'O' if (x.d_AC >= .02 and x.d_BC >= .02) else 'X'
    g2 = 'O' if x.dG2 > 12 else 'X'
    print(f'{x["name"]:<18}{x.s_raw:>8.1f}{x.s:>11.1f}{x.d_AC:>8.4f}{x.d_BC:>8.4f}'
          f'{x.w_C:>7.3f}{x.dG2:>+9.2f}{g1:>5}{g2:>5}')
print('-' * 80)

# 프론티어 관계
c1 = np.corrcoef(R.s, R.dG2)[0, 1]
c2 = np.corrcoef(R.s, R.d_AC)[0, 1]
c3 = np.corrcoef(R.d_AC, R.dG2)[0, 1]
print(f'\n[실측 상관] skill↔ΔG2 {c1:+.3f}   skill↔d_AC {c2:+.3f}   d_AC↔ΔG2 {c3:+.3f}')
print(f'  최고 ΔG2 = {R.dG2.max():+.2f} ({R.loc[R.dG2.idxmax(),"name"]}), '
      f'최고 skill = {R.s.max():.1f} ({R.loc[R.s.idxmax(),"name"]})')

# 2-arm 조합: 변형끼리 서로 다르면 C 를 여러 개 넣을 수 있다
print('\n[4-arm 시험] 상위 변형 2개를 동시에 넣으면?')
best = R.nlargest(4, 'dG2')['name'].tolist()
P = {}
for nm in best:
    for ext in ('.csv', '.npy'):
        f = os.path.join(EXP, nm + '_2024' + ext)
        if os.path.exists(f):
            p = load(f); A = np.stack([np.ones_like(p), p], 1)
            P[nm] = A @ np.linalg.lstsq(A, y, rcond=None)[0]; break
from itertools import combinations
for a, b in combinations(best, 2):
    E = [eA, eB, P[a] - y, P[b] - y]
    M = np.array([[1e5 * (1 - (E[i] * E[j]).mean() / V) for j in range(4)] for i in range(4)])
    w4, u4 = opt(M); _, u2 = opt(M[:2, :2])
    print(f'  A,B,{a},{b}: ΔG2 {u4-u2:+.2f}  w=({w4[2]:.3f},{w4[3]:.3f})  '
          f'd_CC\'={np.sqrt(((P[a]-P[b])**2).mean()):.4f}')
