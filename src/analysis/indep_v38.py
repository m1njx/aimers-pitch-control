"""v36g 행 독립성 독립 감사 (팀 도구를 쓰지 않고 따로 구현).

규정 2-4: test 행은 서로 독립적으로 예측되어야 한다. 즉 파이프라인 전체가
  p_i = f(row_i, 학습시점 자산)
이어야 하며, 같은 행이 어떤 프레임에 담겨 들어가든 같은 값이 나와야 한다.

세 변형을 돌려 행별로 비교한다:
  FULL    : 프로브 N행 그대로
  SHUFFLE : 행 순서만 뒤섞음      -> 순서 의존(rolling/shift/정렬) 검출
  SUBSET  : 앞 절반만            -> 프레임 통계 의존(groupby/평균/분위수) 검출
  SOLO    : 무작위 8행을 1행씩    -> 가장 강한 검사

진짜 누출은 1e-3 안팎, 부동소수점 잡음은 1e-9 수준으로 갈린다.
"""
import os, shutil, subprocess, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(BASE, 'v38')
TRAIN = os.path.expanduser('~/LG_data/open/data/train.csv')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'KMP_DUPLICATE_LIB_OK': 'TRUE'}


def run(frame):
    """프레임을 data/test.csv 로 넣고 파이프라인을 돌려 row_id -> p 시리즈를 받는다."""
    frame.to_csv(os.path.join(RUN, 'data', 'test.csv'), index=False, encoding='utf-8')
    pd.DataFrame({'row_id': frame['row_id'], 'control_success': 0.5}).to_csv(
        os.path.join(RUN, 'data', 'sample_submission.csv'), index=False, encoding='utf-8')
    r = subprocess.run([sys.executable, 'script.py'], cwd=RUN, env=ENV,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f'파이프라인 실패 ({r.returncode})')
    out = pd.read_csv(os.path.join(RUN, 'output', 'submission.csv'), encoding='utf-8-sig')
    return out.set_index('row_id')['control_success']


cols = list(pd.read_csv(os.path.join(RUN, 'data', 'test.csv'), nrows=1,
                        encoding='utf-8-sig').columns)
print(f'프로브 제작: train.csv 에서 {N}행 (test 48컬럼 형태)', flush=True)
tr = pd.read_csv(TRAIN, encoding='utf-8-sig')
sub = tr[tr.season == 2024] if 'season' in tr.columns else tr
probe = sub.sample(n=min(N, len(sub)), random_state=20260830)[cols].reset_index(drop=True)
del tr, sub
# 같은 투수가 여러 행 들어가야 프레임 통계 의존이 드러난다
print(f'  n={len(probe):,}  고유 투수={probe.pitcher_id.nunique():,}  '
      f'최다 투수 행수={probe.pitcher_id.value_counts().iloc[0]}', flush=True)

print('\n[1/4] FULL', flush=True);    p_full = run(probe)
print('[2/4] SHUFFLE', flush=True)
perm = probe.sample(frac=1.0, random_state=7).reset_index(drop=True)
p_shuf = run(perm)
print('[3/4] SUBSET (앞 절반)', flush=True)
half = probe.iloc[:len(probe) // 2].reset_index(drop=True)
p_half = run(half)
print('[4/4] SOLO (무작위 8행을 1행씩)', flush=True)
rng = np.random.default_rng(0)
solo = {}
for i in rng.choice(len(probe), size=8, replace=False):
    row = probe.iloc[[i]].reset_index(drop=True)
    solo[row.row_id.iloc[0]] = run(row).iloc[0]

print('\n' + '=' * 62)
print(f'{"검사":<26}{"max|dp|":>14}{"판정":>10}')
print('-' * 62)
res = {}
for name, s in (('SHUFFLE 불변성', p_shuf), ('SUBSET 불변성', p_half)):
    common = p_full.index.intersection(s.index)
    d = float(np.abs(p_full.loc[common].values - s.loc[common].values).max())
    res[name] = d
    print(f'{name+f" (n={len(common):,})":<26}{d:>14.3e}{"PASS" if d < 1e-6 else "🔴FAIL":>10}')
d_solo = max(abs(p_full.loc[k] - v) for k, v in solo.items())
res['SOLO 불변성'] = d_solo
print(f'{"SOLO 불변성 (n=8)":<26}{d_solo:>14.3e}{"PASS" if d_solo < 1e-6 else "🔴FAIL":>10}')
print('=' * 62)
worst = max(res.values())
print(f'\n최대 편차 {worst:.3e}  ->  ' +
      ('✅ 행 독립 (부동소수점 잡음 수준)' if worst < 1e-6 else
       '🔴 프레임 의존 존재 — 규정 2-4 위반 소지'))
print(f'\n참고: 예측 분포 mean={p_full.mean():.6f} min={p_full.min():.6f} max={p_full.max():.6f}')
