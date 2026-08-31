#!/usr/bin/env python3
"""ceiling_probe.py — 천장이 '정보'인지 '시간 일반화'인지 가른다.

gap_analysis 결과: 캘리브레이션 여지는 +10.5점뿐이고, 우리 모델의 해상도(846.6)는
투수/타자/카운트/이닝/월 어떤 저차 셀 오라클보다도 이미 높다. 그렇다면 남은 격차는
어디에 있나? 두 가설이 남는다.

  H-정보천장 : 피처 안에 더 짜낼 신호가 없다. 누가 해도 이 근방이 끝.
  H-시간격차 : 신호는 있는데 '과거로 학습해 미래를 맞추는' 데서 잃는다.

가르는 법: **피처 행렬을 고정한 채** 라벨의 출처만 바꾼다.
  조건 A  과거(<=2023) 라벨로 학습 → 2024 평가   [실전 조건, 캐시에 이미 있음]
  조건 B  2024 라벨로 교차적합(5-fold) → 2024 평가 [동시대 라벨을 가진 가상 조건]

B >> A 이면 병목은 시간 일반화이고, 최근성·적응 쪽에 남은 여지가 있다는 뜻.
B ~= A 이면 정보천장이고, 같은 피처로는 누구도 크게 못 넘는다는 뜻이 된다.

주의: B는 진단 전용이다. 동시대 라벨은 실전에서 존재하지 않으므로 제출에 쓸 수 없다.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

CACHE = os.path.join(LG, 'harness/cache')
YEAR = 2024
NFOLD = 5


def skill(p, y):
    base = y.mean() * (1 - y.mean())
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / base)


def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    tr = df[df.season < YEAR]
    va = df[df.season == YEAR].reset_index(drop=True)
    y = va.control_success.values.astype(np.float64)

    print('[A] 과거 라벨 학습 (캐시된 lgb_bin)', flush=True)
    seeds = [s for s in (7, 123, 2025) if os.path.exists(os.path.join(CACHE, f'pred_{YEAR}_{s}.npz'))]
    a_lgb = np.mean([np.load(os.path.join(CACHE, f'pred_{YEAR}_{s}.npz'))['lgb_bin']
                     for s in seeds], axis=0)
    print(f'    lgb_bin  skill = {skill(a_lgb, y):.1f}  (시드 {len(seeds)}개 평균)', flush=True)

    print('\n[B] 동시대 라벨 교차적합 — 피처는 A와 완전히 동일하게 생성', flush=True)
    prep = bc.PitchPreprocessor()
    prep.fit(tr, as_of_season=YEAR - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    base_str = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              tr['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cc_str + '_' + base_str).unique())}
    dec = bc.AsofDecomposer2()
    dec.fit(tr, val_season=YEAR)
    Xva, _ = bc.build_features(va, prep, dec, cat_map)
    print(f'    피처 {Xva.shape}  ({time.time()-t0:.0f}s)', flush=True)

    rng = np.random.RandomState(0)
    fold = rng.randint(0, NFOLD, len(va))
    oof = np.zeros(len(va))
    p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
             seed=7, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8)
    for k in range(NFOLD):
        m = fold != k
        mdl = lgb.train(p, lgb.Dataset(Xva[m], label=y[m]))
        oof[~m] = mdl.predict(Xva[~m])
        print(f'    fold {k+1}/{NFOLD} 완료 ({time.time()-t0:.0f}s)', flush=True)

    def recal(pr):
        """레벨·스케일 차이를 2모수로 흡수 — 프로덕션의 SHIFT/SCALE에 해당.
        이걸 안 하면 A만 base-rate 이동 손해를 뒤집어써서 격차가 부풀려진다."""
        b = np.cov(pr, y)[0, 1] / pr.var()
        return skill(np.clip(y.mean() + b * (pr - pr.mean()), 1e-6, 1 - 1e-6), y)

    sa, sb = skill(a_lgb, y), skill(oof, y)
    ra, rb = recal(a_lgb), recal(oof)
    print('\n' + '=' * 64)
    print(f'  {"":22s} {"raw":>9s} {"선형재보정후":>12s}')
    print(f'  A 과거라벨 학습        {sa:9.1f} {ra:12.1f}')
    print(f'  B 동시대라벨 교차적합   {sb:9.1f} {rb:12.1f}')
    print(f'  차이                  {sb-sa:+9.1f} {rb-ra:+12.1f}')
    print('=' * 64)
    print(f'\n  raw 격차 중 {sb-sa-(rb-ra):.0f}점은 평균이동(프로덕션이 이미 보정하는 채널),')
    print(f'  순수 해상도 격차는 {rb-ra:+.1f}점이다.')

    np.savez_compressed(os.path.join(LG, 'harness/ceiling_preds.npz'),
                        a=a_lgb, b=oof, y=y,
                        pitcher_id=va.pitcher_id.values,
                        asof_n=va.asof_pitcher_n.fillna(0).values)

    # ---- 격차의 출처: 시즌간 드리프트인가, 투수 모집단 교체인가 ----
    print('\n[C] 격차 분해 — 학습기간에 등장한 투수 vs 신규 투수', flush=True)
    seen = set(tr.pitcher_id.unique())
    is_seen = va.pitcher_id.isin(seen).values
    for name, mask in [('학습기 등장 투수', is_seen), ('신규 투수', ~is_seen)]:
        if mask.sum() < 1000:
            print(f'    {name:18s} n={mask.sum():,} (표본부족)')
            continue
        yy = y[mask]
        ba = np.cov(a_lgb[mask], yy)[0, 1] / a_lgb[mask].var()
        bb = np.cov(oof[mask], yy)[0, 1] / oof[mask].var()
        ka = skill(np.clip(yy.mean() + ba * (a_lgb[mask] - a_lgb[mask].mean()), 1e-6, 1-1e-6), yy)
        kb = skill(np.clip(yy.mean() + bb * (oof[mask] - oof[mask].mean()), 1e-6, 1-1e-6), yy)
        print(f'    {name:18s} n={mask.sum():>7,}  A={ka:7.1f}  B={kb:7.1f}  격차={kb-ka:+7.1f}')

    print('\n[D] 격차 분해 — 투수 시즌내 누적 투구수 구간별', flush=True)
    n_asof = va.asof_pitcher_n.fillna(0).values
    for lo, hi in [(0, 100), (100, 500), (500, 1500), (1500, 10 ** 9)]:
        mask = (n_asof >= lo) & (n_asof < hi)
        if mask.sum() < 1000:
            continue
        yy = y[mask]
        ba = np.cov(a_lgb[mask], yy)[0, 1] / a_lgb[mask].var()
        bb = np.cov(oof[mask], yy)[0, 1] / oof[mask].var()
        ka = skill(np.clip(yy.mean() + ba * (a_lgb[mask] - a_lgb[mask].mean()), 1e-6, 1-1e-6), yy)
        kb = skill(np.clip(yy.mean() + bb * (oof[mask] - oof[mask].mean()), 1e-6, 1-1e-6), yy)
        lab = f'asof_n {lo}~{hi if hi < 10**8 else ""}'
        print(f'    {lab:18s} n={mask.sum():>7,}  A={ka:7.1f}  B={kb:7.1f}  격차={kb-ka:+7.1f}')
    print('\n해석 기준: 차이가 100점 이상이면 병목은 시간 일반화(최근성/적응),')
    print('           50점 미만이면 같은 피처 위에서는 정보천장에 근접한 것.')
    print(f'총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
