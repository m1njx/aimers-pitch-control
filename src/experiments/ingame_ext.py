"""ingame_ext.py — 팀 ingame.py 의 h=5/6 수용 게이트만 완화해 커버리지를 넓힌다.

배경 (`outputs/524`)
--------------------
팀 모듈의 2024 실측: 커버리지 22.83%, 복원 정확도 j/s 98.95%, 이득 +31.06 로컬.
식별 가능 상한(h<=6)은 +49.81 로컬이므로 **미회수 잔여 +18.76 로컬 (~+10 LB)**.
그 잔여의 87% 가 h=5, h=6 에 있고 거기서 커버리지가 43% 로 떨어진다.

원인은 `ingame.py:402` 의 게이트다:

    ok[h] = g & (role_T != 0) & (L5 >= L5_MIN) & w3

  * `role_T != 0`  : 구원투수(train 이력 있고 평균 경기길이 < STARTER_N) 전면 제외
  * `L5 >= 150`    : prev5 분모들의 lcm 하한
  * `w3`           : prev3 창 존재 필터

이 파일은 **팀 저장소 를 수정하지 않는다.** 소스를 읽어 그 한 줄만 파라미터화한 뒤
별도 모듈 네임스페이스로 exec 한다. 나머지 로직은 바이트 그대로다.

⚠️ 완화하면 커버리지는 오르지만 정확도가 떨어진다. 틀린 (j,s) 는 **잘못된 방향으로**
보정하므로 순이득이 상쇄될 수 있다. 반드시 아래 게이트로 채점할 것.

⚠️ 규정: 이 축 전체가 `outputs/524` 의 미해결 규정 리스크 위에 있다.
`outputs/525` 질의 답이 "불가" 면 이 파일과 v26 계열 전체가 무효다.

실행: venv311/bin/python3 harness/ingame_ext.py
"""
import os
import sys
import types

import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
TEAM = '<팀 저장소 경로>/experiments/v11_cli'
ING = os.path.join(TEAM, 'ingame.py')

_ORIG_GATE = "ok[h] = g & (role_T != 0) & (L5 >= L5_MIN) & w3"
_NEW_GATE = ("ok[h] = g & ((role_T != 0) | bool(ALLOW_ROLE0)) "
             "& (L5 >= L5_MIN) & (w3 | bool(SKIP_W3))")


def load(allow_role0=False, l5_min=150, skip_w3=False):
    """게이트만 바꾼 ingame 모듈 사본을 만든다. 원본 파일은 건드리지 않는다."""
    src = open(ING, encoding='utf-8').read()
    if _ORIG_GATE not in src:
        raise RuntimeError('게이트 문자열을 못 찾았다 — 팀 쪽 ingame.py 가 바뀌었다. '
                           '확장을 적용하기 전에 524 를 다시 확인할 것.')
    src = src.replace(_ORIG_GATE, _NEW_GATE)
    mod = types.ModuleType(f'ingame_r{int(allow_role0)}_{l5_min}_{int(skip_w3)}')
    mod.__dict__['__file__'] = ING
    exec(compile(src, ING, 'exec'), mod.__dict__)
    mod.L5_MIN = l5_min
    mod.ALLOW_ROLE0 = allow_role0
    mod.SKIP_W3 = skip_w3
    return mod


def main():
    sys.path.insert(0, os.path.join(LG, 'harness'))
    from evaluate import PROD, predict

    base_mod = load()
    tr = pd.read_csv(os.path.join(LG, 'open/data/train.csv'),
                     encoding='utf-8-sig', usecols=base_mod.FIT_COLS)
    fit = tr[tr.season < 2024]
    va = tr[tr.season == 2024].reset_index(drop=True)

    y = np.load(os.path.join(LG, 'harness/cache/y_2024.npy'))
    assert (va.control_success.values == y).all(), '행 정렬 불일치'
    p = np.mean([predict(PROD, dict(np.load(
        os.path.join(LG, f'harness/cache/pred_2024_{s}.npz'))))
        for s in [7, 123, 2025, 31415, 8675309]], axis=0)
    V = y.mean() * (1 - y.mean())

    def skill(q):
        return 1e5 * (1.0 - ((q - y) ** 2).mean() / V)

    truth = base_mod.reconstruct_truth(va)
    tj, ts, th = truth.j_true.values, truth.s_true.values, truth.h_true.values
    s0 = skill(p)
    logit = np.log(p / (1 - p))

    def gain(jj, ss, mask, b):
        z = logit.copy()
        z[mask] = logit[mask] + b * (ss[mask] - jj[mask] * p[mask]) / (jj[mask] + 20)
        return skill(1.0 / (1.0 + np.exp(-z))) - s0

    ceil = gain(tj, ts, th <= 5, 1.5)
    print(f'베이스 {s0:.2f}   h<=6 오라클 상한 {ceil:+.2f}   전 행 오라클 {gain(tj,ts,np.ones(len(y),bool),1.5):+.2f}\n')
    print(f'{"설정":>26} {"커버":>7} {"정확도":>7} {"이득b1.2":>9} {"이득b1.5":>9} {"상한대비":>8}')

    cfgs = [(False, 150, False, '기준선 (팀 그대로)'),
            (True,  150, False, 'role0 허용'),
            (False, 100, False, 'L5>=100'),
            (False,  60, False, 'L5>=60'),
            (True,  100, False, 'role0 + L5>=100'),
            (True,   60, False, 'role0 + L5>=60'),
            (True,   60, True,  'role0 + L5>=60 + w3해제'),
            (True,    1, True,  '전부 해제')]
    for r0, l5, sw, name in cfgs:
        mod = load(allow_role0=r0, l5_min=l5, skip_w3=sw)
        tab = mod.fit_tables(fit)
        out = mod.transform(va, tab)
        v = out.ig_valid.values.astype(bool)
        pj, ps = out.ig_j.values, out.ig_s.values
        acc = np.mean((pj[v] == tj[v]) & (ps[v] == ts[v])) if v.any() else np.nan
        g12, g15 = gain(pj, ps, v, 1.2), gain(pj, ps, v, 1.5)
        print(f'{name:>26} {v.mean():7.2%} {acc:7.2%} {g12:+9.2f} {g15:+9.2f} '
              f'{g15/ceil:8.1%}')

    print('\n판정 기준: 기준선(+31.06) 대비 이득이 유의하게 커야 채택.')
    print('정확도가 떨어지면 틀린 행이 반대 방향으로 보정해 순이득을 깎는다.')


if __name__ == '__main__':
    main()
