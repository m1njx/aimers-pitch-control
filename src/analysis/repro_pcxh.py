"""출하 pcxh 테이블 재현 시도 — 성공하면 임의 폴드의 테이블을 만들 수 있다.

왜 필요한가: 내게 있는 파리티 테이블은 ≤2023 하나뿐이라 **β 를 정직하게 적합할 수 있는
폴드가 2024 밖에 없다**. 그래서 제안 #3(동시 재적합)의 전이를 측정할 수 없었다.
테이블 생성 레시피를 재현하면 ≤2021 테이블을 만들어 2022 에서 β 를 적합할 수 있다.

레시피 (pcxh_apply.py 독스트링):
  dev_cell = r(투수, cell, bh) − r(투수, bucket(cell)) − [리그 r(cell,bh) − 리그 r(bucket)]
  rates 는 EB 수축 (K 의사관측을 부모 수준으로)
미지수는 **bucket(cell)** 하나. 후보를 훑어 출하 테이블과 대조한다.

판정: dev_cell 의 상관 > 0.999 이고 최대절대오차 < 1e-3 이면 재현 성공.

[기여 구분] 여기서 다루는 엔티티 잔차 룩업 기법 자체는 **팀 공동 파트에서 도입**됐다.
이 스크립트는 그 기법에 대해 내가 수행한 **독립 검증·확장 시도**다.
"""
import itertools, os, sys
import numpy as np
import pandas as pd

LG = os.path.expanduser('~/LG_data')
PAR = os.path.dirname(os.path.abspath(__file__)) + '/parity/out/pcxh_parity_le2023'
K = 100.0

tr = pd.read_csv(f'{LG}/open/data/train.csv')
d = tr[tr.season <= 2023]
b = d.balls_before.fillna(0).to_numpy().astype(np.int64)
s = d.strikes_before.fillna(0).to_numpy().astype(np.int64)
cell = np.clip(b, 0, 3) * 3 + np.clip(s, 0, 2)
bh = np.clip(d.batter_hand.fillna(1).to_numpy().astype(np.int64) - 1, 0, 1)
pid = d.pitcher_id.to_numpy().astype(np.int64)
y = d.control_success.to_numpy().astype(float)

REF = pd.read_csv(f'{PAR}/pcxh_cell.csv')
print(f'출하 테이블: {len(REF):,}행, dev_cell sd={REF.dev_cell.std():.5f}')

BUCKETS = {
    'sign(s-b)': np.sign(s - b).astype(np.int64) + 1,
    'balls': np.clip(b, 0, 3),
    'strikes': np.clip(s, 0, 2),
    'pooled': np.zeros(len(d), np.int64),
    'ahead/behind/even x2s': (np.sign(s - b).astype(np.int64) + 1) * 2 + (s >= 2),
    'cell자체': cell,
}


def build(bucket, k):
    D = pd.DataFrame(dict(e=pid, cell=cell, bh=bh, bk=bucket, y=y))
    lg_all = y.mean()
    lg_cb = D.groupby(['cell', 'bh']).y.mean()
    lg_bk = D.groupby('bk').y.mean()
    gp = D.groupby(['e', 'bk']).y.agg(['sum', 'count'])
    par = (gp['sum'] + k * lg_bk.reindex(gp.index.get_level_values('bk')).to_numpy()) / (gp['count'] + k)
    g = D.groupby(['e', 'cell', 'bh', 'bk']).y.agg(['sum', 'count']).reset_index()
    p = par.reindex(pd.MultiIndex.from_arrays([g['e'], g['bk']])).to_numpy()
    rc = (g['sum'].to_numpy() + k * p) / (g['count'].to_numpy() + k)
    g['dev'] = (rc - p) - (lg_cb.reindex(pd.MultiIndex.from_arrays([g['cell'], g['bh']])).to_numpy()
                           - lg_bk.reindex(g['bk']).to_numpy())
    return g.set_index(['e', 'cell', 'bh'])['dev']


print(f'\n{"bucket":<24}{"K":>6}{"행수":>9}{"corr":>10}{"maxdiff":>11}')
print('-' * 62)
best = None
idx = pd.MultiIndex.from_arrays([REF.pitcher_id, REF.cell, REF.bh])
for name, bk in BUCKETS.items():
    for k in (50., 100., 200.):
        t = build(bk, k)
        v = t.reindex(idx).to_numpy(float)
        m = ~np.isnan(v)
        if m.sum() < len(REF) * 0.9:
            print(f'{name:<24}{k:>6.0f}{len(t):>9,}{"키 불일치":>10}{m.mean()*100:>10.0f}%')
            continue
        c = np.corrcoef(v[m], REF.dev_cell.to_numpy()[m])[0, 1]
        md = np.abs(v[m] - REF.dev_cell.to_numpy()[m]).max()
        print(f'{name:<24}{k:>6.0f}{len(t):>9,}{c:>10.5f}{md:>11.2e}')
        if best is None or c > best[0]:
            best = (c, name, k, md)
print('-' * 62)
print(f'\n최고 일치: bucket={best[1]}  K={best[2]:.0f}  corr={best[0]:.5f}  maxdiff={best[3]:.2e}')
print('->', '✅ 재현 성공 — 임의 폴드 테이블 생성 가능'
      if best[0] > 0.999 and best[3] < 1e-3 else
      '❌ 재현 실패 — 레시피의 다른 부분이 다르다(부모 수준·수축 방식 등)')
