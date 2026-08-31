"""
diag_cross_skill.py — 교차 skill c 의 구조 분해와 우리 5성분 쌍별 측정.

outputs/516 은 LB 3점에서 c = 1085.47 을 유도했다. 여기서는 c 가 무엇으로
이루어져 있는지를 대수로 쪼개고, 그 항들을 캐시에서 직접 측정한다.

  skill(p) = 1e5 * (1 - E[(p-y)^2] / V),  V = r(1-r)
  c(A,B)   = 1e5 * (1 - E[e_A e_B] / V)

항등식 1 (분산 항등식):
  c = (a+b)/2 + D,   D = 1e5 * E[(p_A - p_B)^2] / (2V)
  -> D 는 오차 품질과 무관하고 **예측 벡터의 불일치량**만으로 결정된다.
  -> 50:50 블렌드 이득(평균 arm 대비) = D/2.

항등식 2 (독립 노이즈 불변성):
  p_A' = p_A + eps, eps ⟂ (e_B), E[eps]=0  =>  c 불변 (a 는 내려가고 D 는 같은 양만큼 오른다)
  -> "다양성"을 노이즈로 사는 것은 c 를 못 올린다. 시드/HP 확장이 c 를 못 올리는 이유.

항등식 3 (베이즈 분해):  e = (p - q) + (q - y), q = 진짜 조건부확률
  E[e_A e_B] = E[(p_A-q)(p_B-q)] + E[q(1-q)]
  c = 1e5*(1 - E[q(1-q)]/V) - 1e5*E[(p_A-q)(p_B-q)]/V
    = (베이즈 skill) - (두 arm 근사오차의 공분산 항)
  -> c 의 상한은 베이즈 skill 이고, 두 arm 의 근사오차가 직교할 때 달성된다.
  -> c 를 올리는 유일한 길: 근사오차의 공분산을 줄이는 것.

실행: venv311/bin/python3 harness/diag_cross_skill.py
"""
import os, glob, itertools
import numpy as np

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
COMPS = ['lgb_bin', 'cb_bin', 'xgb_bin', 'lgb_mse', 'mlp']
SEEDS = [7, 123, 2025, 31415, 8675309]


def load():
    folds = {}
    for f in sorted(glob.glob(os.path.join(CACHE, 'pred_*.npz'))):
        _, yr, sd = os.path.basename(f)[:-4].split('_')
        yr, sd = int(yr), int(sd)
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy'))
        folds.setdefault(yr, {'y': y, 'seeds': {}})['seeds'][sd] = {
            k: np.asarray(v, dtype=np.float64) for k, v in np.load(f).items()}
    return folds


def skill(p, y, V):
    return 1e5 * (1.0 - ((p - y) ** 2).mean() / V)


def cross(pa, pb, y, V):
    return 1e5 * (1.0 - ((pa - y) * (pb - y)).mean() / V)


def affine_fit(p, y):
    """arm 이 자기 캘리브레이션을 싣고 나가는 상황의 아날로그.
    p' = 0.5 + s*(p-0.5) + t 를 MSE 최소화로 적합 -> 닫힌형 최소제곱."""
    X = np.column_stack([p - 0.5, np.ones_like(p)])
    coef, *_ = np.linalg.lstsq(X, y - 0.5, rcond=None)
    return float(coef[0]), float(coef[1])


def affine_apply(p, s, t, e=1e-6):
    return np.clip(0.5 + s * (p - 0.5) + t, e, 1 - e)


def main():
    folds = load()
    years = sorted(folds)
    print('cache: ' + ', '.join('%d(n=%d, seeds=%d)' % (y, len(folds[y]['y']), len(folds[y]['seeds']))
                                for y in years))

    # ---------- 0. 팀 블렌드 숫자 재현 + D 의 의미 ----------
    a_t, b_t, bl_t = 1032.137582, 1016.4138496773, 1054.8707763763
    c_t = 2 * (bl_t - 0.25 * a_t - 0.25 * b_t)
    D_t = c_t - 0.5 * (a_t + b_t)
    print('\n=== 팀 v23 (LB 실측) ===')
    print('  a(우리v42)=%.1f  b(B축)=%.1f  블렌드=%.1f' % (a_t, b_t, bl_t))
    print('  c=%.2f   arm평균=%.2f   D = c - 평균 = %.2f' % (c_t, 0.5 * (a_t + b_t), D_t))
    print('  -> 50:50 블렌드가 arm평균 대비 얻은 이득 = D/2 = %.2f (최고 arm 대비 +%.1f)'
          % (D_t / 2, bl_t - a_t))
    for r in (0.25, 0.35, 0.50):
        V = r * (1 - r)
        rms = np.sqrt(D_t * 2 * V / 1e5)
        print('     r=%.2f 가정 -> 두 arm 예측 RMS 불일치 = %.4f (확률 %.2f%%p)' % (r, rms, rms * 100))

    # ---------- 1. 우리 5성분 쌍별 ----------
    print('\n=== 우리 5성분 쌍별 (성분별 out-of-fold 아핀 캘리브레이션 적용) ===')
    print('    캘리브레이션은 평가 시즌을 제외한 나머지 시즌에서 적합 -> 오라클 아님')

    # 성분별 캘리브레이션 계수: 평가연도 Y 를 뺀 나머지 연도 풀로 적합 (시드 평균 예측 사용)
    cal = {}
    for Y in years:
        for comp in COMPS:
            ps, ys = [], []
            for Y2 in years:
                if Y2 == Y:
                    continue
                pm = np.mean([folds[Y2]['seeds'][s][comp] for s in sorted(folds[Y2]['seeds'])], axis=0)
                ps.append(pm); ys.append(folds[Y2]['y'])
            cal[(Y, comp)] = affine_fit(np.concatenate(ps), np.concatenate(ys))

    rows = {}
    for Y in years:
        y = folds[Y]['y']; V = y.mean() * (1 - y.mean())
        seeds = sorted(folds[Y]['seeds'])
        # 프로덕션과 동일한 예측 배깅: 시드 평균 예측을 먼저 만들고 그 다음 채점
        P = {comp: affine_apply(np.mean([folds[Y]['seeds'][s][comp] for s in seeds], axis=0),
                                *cal[(Y, comp)]) for comp in COMPS}
        sk = {comp: skill(P[comp], y, V) for comp in COMPS}
        rows[Y] = dict(V=V, r=float(y.mean()), skill=sk, pair={})
        for i, j in itertools.combinations(COMPS, 2):
            c = cross(P[i], P[j], y, V)
            D = c - 0.5 * (sk[i] + sk[j])
            D_id = 1e5 * ((P[i] - P[j]) ** 2).mean() / (2 * V)   # 항등식 1 검증
            bl = skill(0.5 * (P[i] + P[j]), y, V)
            bl_pred = 0.25 * sk[i] + 0.25 * sk[j] + 0.5 * c       # 2차형식 검증
            rows[Y]['pair'][(i, j)] = dict(a=sk[i], b=sk[j], c=c, D=D, D_id=D_id,
                                           blend=bl, blend_pred=bl_pred)

    # 항등식 검증
    e1 = max(abs(v['D'] - v['D_id']) for Y in years for v in rows[Y]['pair'].values())
    e2 = max(abs(v['blend'] - v['blend_pred']) for Y in years for v in rows[Y]['pair'].values())
    print('\n  항등식 1 (D = 정규화 불일치)   최대 오차 = %.3e' % e1)
    print('  항등식 2 (블렌드 = 2차형식)    최대 오차 = %.3e' % e2)

    print('\n  성분 단독 skill (out-of-fold 캘리브레이션 후)')
    print('  %-9s' % 'season' + ''.join('%11s' % c for c in COMPS) + '%11s' % 'r')
    for Y in years:
        print('  %-9d' % Y + ''.join('%11.1f' % rows[Y]['skill'][c] for c in COMPS)
              + '%11.3f' % rows[Y]['r'])

    print('\n  쌍별 D = c - (a+b)/2  [예측 불일치량, 시즌별]')
    print('  %-22s' % 'pair' + ''.join('%10d' % Y for Y in years) + '%10s' % 'mean')
    pairs = list(itertools.combinations(COMPS, 2))
    Dmean = {}
    for p in pairs:
        vals = [rows[Y]['pair'][p]['D'] for Y in years]
        Dmean[p] = float(np.mean(vals))
        print('  %-22s' % ('%s|%s' % p) + ''.join('%10.1f' % v for v in vals) + '%10.1f' % Dmean[p])

    print('\n  쌍별 c  [교차 skill, 시즌별]  — 팀 LB c=1085.5 와 비교할 대상')
    print('  %-22s' % 'pair' + ''.join('%10d' % Y for Y in years) + '%10s' % 'mean')
    for p in pairs:
        vals = [rows[Y]['pair'][p]['c'] for Y in years]
        print('  %-22s' % ('%s|%s' % p) + ''.join('%10.1f' % v for v in vals) + '%10.1f' % float(np.mean(vals)))

    # ---------- 2. 시드 다양성이 c 를 올리는가 (항등식 2 실증) ----------
    print('\n=== 시드 다양성의 c 기여 (같은 성분, 다른 시드) ===')
    print('    "시드/HP 확장은 c 를 못 올린다"의 직접 실증')
    print('  %-12s %10s %10s %10s %10s' % ('comp/season', 'a(seed1)', 'b(seed2)', 'c', 'D'))
    for comp in COMPS:
        for Y in years:
            y = folds[Y]['y']; V = y.mean() * (1 - y.mean())
            s0, s1 = sorted(folds[Y]['seeds'])[:2]
            sA, tA = cal[(Y, comp)]
            pA = affine_apply(folds[Y]['seeds'][s0][comp], sA, tA)
            pB = affine_apply(folds[Y]['seeds'][s1][comp], sA, tA)
            a = skill(pA, y, V); b = skill(pB, y, V); c = cross(pA, pB, y, V)
            print('  %-12s %10.1f %10.1f %10.1f %10.1f'
                  % ('%s/%d' % (comp, Y), a, b, c, c - 0.5 * (a + b)))
        break  # 대표로 첫 성분만 (나머지는 --all 로)

    # ---------- 3. 프로덕션 arm 자체와 각 성분의 c ----------
    print('\n=== 프로덕션 블렌드(우리 arm) vs 각 성분의 c ===')
    print('    현재 arm 에 그 성분을 더 섞을 여지가 남았는지 = c 가 arm skill 보다 높은가')
    PRODW = dict(lgb_bin=0.40 * 0.20, cb_bin=0.40 * 0.72, xgb_bin=0.40 * 0.08,
                 mlp=0.40, lgb_mse=0.20)
    print('  %-10s' % 'season' + '%10s' % 'arm' + ''.join('%10s' % c for c in COMPS))
    for Y in years:
        y = folds[Y]['y']; V = y.mean() * (1 - y.mean())
        seeds = sorted(folds[Y]['seeds'])
        raw = {comp: np.mean([folds[Y]['seeds'][s][comp] for s in seeds], axis=0) for comp in COMPS}
        arm = sum(PRODW[c] * raw[c] for c in COMPS)
        # arm 도 동일하게 out-of-fold 아핀 캘리브레이션
        ps, ys = [], []
        for Y2 in years:
            if Y2 == Y:
                continue
            r2 = {comp: np.mean([folds[Y2]['seeds'][s][comp] for s in sorted(folds[Y2]['seeds'])], axis=0)
                  for comp in COMPS}
            ps.append(sum(PRODW[c] * r2[c] for c in COMPS)); ys.append(folds[Y2]['y'])
        sA, tA = affine_fit(np.concatenate(ps), np.concatenate(ys))
        armc = affine_apply(arm, sA, tA)
        a = skill(armc, y, V)
        line = '  %-10d%10.1f' % (Y, a)
        for comp in COMPS:
            pc = affine_apply(raw[comp], *cal[(Y, comp)])
            line += '%10.1f' % cross(armc, pc, y, V)
        print(line)

    print('\n주의: 로컬 절대 skill 은 LB 와 역상관이다(dacon-local-harness-invalid).')
    print('      위 표는 절대 수준이 아니라 **구조(불일치량·공분산)**를 읽는 용도다.')


if __name__ == '__main__':
    main()
