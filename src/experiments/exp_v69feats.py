#!/usr/bin/env python3
"""exp_v69feats.py — v69 의 신규 15피처를 프로덕션 파이프라인에 이식해 검증.

배경
----
`work/build_v69_features.py` 가 만든 v69 는 자체 CV 에서 폴드별 skill
2094.0 / 722.0 / 749.7 (평균 1188.6) 로, 같은 폴드의 현행 프로덕션
2124.3 / 699.5 / 826.7 (평균 1216.8) 보다 **−28.2 낮다**. 특히 2024 에서 −77.

다만 v69 는 LGB+CB 2성분(0.6/0.4)뿐이고 프로덕션은 5성분 블렌드 + 5시드 배깅이다.
**피처가 좋은데 아키텍처가 약해서 진 것일 수 있으므로**, 피처만 떼어 프로덕션에
이식해 단독으로 검증한다. 조작 변수는 15피처 추가 하나뿐.

이식하는 15개 (`build_v69_features.py:154-232` 원문 그대로 옮김)
--------------------------------------------------------------
new_eff_velocity, new_speed_loss, new_bauer_units, new_spin_efficiency,
tkm_fastball_pct, tkm_breaking_pct, tkm_offspeed_pct, new_pitcher_win_exp,
new_form_acceleration, new_form_volatility, new_experience_ratio,
new_count_advantage, new_pitcher_ahead, new_full_count, new_command_quality

⚠️ 이 중 일부는 프로덕션에 이미 있는 것과 사실상 중복이다:
  new_eff_velocity   ~ phys_effective_velocity
  new_spin_efficiency ~ phys_spin_efficiency
  new_count_advantage ~ feat_count_advantage
  new_pitcher_ahead   ~ feat_pitcher_ahead
  new_full_count      ~ feat_full_count
중복 피처는 순수 용량비용이며, 이 문제는 용량 추가를 강하게 처벌한다(`outputs/505`).
그래도 v69 주장을 충실히 재현하기 위해 15개 전부 넣는다.

⚠️ `tkm_pitch_mix_table.pkl` 은 v69 가 저장한 것을 그대로 쓴다. 이 표는 전 시즌
트랙맨으로 만들어져 val 시즌 정보가 섞여 있어 **v69 쪽에 미세하게 유리하다**.
14개 카운트 상태의 리그 평균이라 영향은 작지만, 유리한 쪽으로 기운 조건에서도
통과 못하면 결론은 더 확실해진다.

리키지 점검 결과: form_acceleration/volatility 는 주최측 제공 `prev1/3/5_game_*`
컬럼 기반이고 이 컬럼들은 현재 경기를 구조적으로 제외한다 → 경기내 리키지 없음.
각 행이 자기 입력만 쓰므로 규정4 무관.

판정 (사전 확정, 결과 보고 변경 금지)
-----------------------------------
inner 3폴드(2021/2022/2023) x 5시드, 프로덕션과 동일한 예측 배깅 채점,
짝지은 15셀. **3폴드 전부 양수 + t > 2.5.**
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc

BASE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
MIX_PATH = os.path.join(LG, 'work/submit_v69_newfeats/model/tkm_pitch_mix_table.pkl')
MIX = joblib.load(MIX_PATH) if os.path.exists(MIX_PATH) else None

_orig = bc.build_features


def add_v69(X, df_raw):
    """build_v69_features.py 의 add_new_features 를 그대로 옮긴 것."""
    X = X.copy()
    rel_speed = X['tkm_rel_speed_mean']
    ext = X['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
    zone_speed = X['tkm_zone_speed_mean']
    X['new_eff_velocity'] = (rel_speed * 60.5 / (60.5 - ext)).astype(np.float32)
    X['new_speed_loss'] = (rel_speed - zone_speed).astype(np.float32)

    spin = X['tkm_spin_rate_mean']
    ivb = X['tkm_induced_vert_break_mean']
    hb = X['tkm_horz_break_mean']
    X['new_bauer_units'] = (spin / rel_speed.clip(lower=60.0)).astype(np.float32)
    total_break = np.sqrt(ivb ** 2 + hb ** 2)
    X['new_spin_efficiency'] = (total_break / spin.clip(lower=100.0)).astype(np.float32)

    balls_i = df_raw['balls_before'].fillna(0).astype(int)
    strikes_i = df_raw['strikes_before'].fillna(0).astype(int)
    key = balls_i.astype(str) + '_' + strikes_i.astype(str)
    for col in ['tkm_fastball_pct', 'tkm_breaking_pct', 'tkm_offspeed_pct']:
        if MIX is not None:
            X[col] = key.map(MIX.get(col, {})).fillna(
                MIX.get(col + '_global', 0.33)).astype(np.float32).values
        else:
            X[col] = np.float32(0.33)

    tb = df_raw['top_bottom'].values
    hwe = df_raw['home_win_expectancy'].fillna(50.0).values
    awe = df_raw['away_win_expectancy'].fillna(50.0).values
    X['new_pitcher_win_exp'] = np.where(np.isin(tb, ['T', 'Top']), hwe, awe).astype(np.float32)

    prev1 = df_raw['asof_pitcher_prev1_game_success_rate'].fillna(0.5).values
    prev3 = df_raw['asof_pitcher_prev3_game_success_rate'].fillna(0.5).values
    prev5 = df_raw['asof_pitcher_prev5_game_success_rate'].fillna(0.5).values
    X['new_form_acceleration'] = ((prev1 - prev3) - (prev3 - prev5)).astype(np.float32)
    X['new_form_volatility'] = np.nanstd(np.stack([prev1, prev3, prev5], axis=1),
                                         axis=1).astype(np.float32)

    pn = df_raw['asof_pitcher_n'].fillna(1).values.astype(float)
    bn = df_raw['asof_batter_n'].fillna(1).values.astype(float)
    X['new_experience_ratio'] = (np.log1p(pn) / np.log1p(bn).clip(min=0.01)).astype(np.float32)

    b = df_raw['balls_before'].fillna(0).values.astype(float)
    s = df_raw['strikes_before'].fillna(0).values.astype(float)
    X['new_count_advantage'] = (s - 1.5 * b).astype(np.float32)
    X['new_pitcher_ahead'] = (s > b).astype(int).astype(np.float32)
    X['new_full_count'] = ((b == 3) & (s == 2)).astype(int).astype(np.float32)

    sr = df_raw['asof_pitcher_strike_rate'].fillna(0.5).values
    br = df_raw['asof_pitcher_ball_rate'].fillna(0.3).values
    mr = df_raw['asof_pitcher_middle_rate'].fillna(0.15).values
    X['new_command_quality'] = (sr - br - mr).astype(np.float32)
    return X


def patched(df, prep, dec, cat_map):
    X, X133 = _orig(df, prep, dec, cat_map)
    Xn = add_v69(X, df)
    new = [c for c in Xn.columns if c not in X.columns]
    add = Xn[new]
    add.index = X133.index
    return Xn, pd.concat([X133, add], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v69feats')
    a = ap.parse_args()
    t0 = time.time()
    print(f'pitch mix 표 로드: {"OK" if MIX is not None else "없음(0.33 대체)"}', flush=True)

    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    cdir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cdir, exist_ok=True)
    bc.build_features = patched
    bc.CACHE = cdir
    for y in FOLDS:
        bc.run_fold(df, y, SEEDS)

    from evaluate import PROD, predict, skill
    print('\n[채점] 프로덕션 배깅 + 짝지은 15셀', flush=True)
    cells, per_fold, bag = [], {}, {}
    for y in FOLDS:
        yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
        d, pa, pb = [], [], []
        for s in SEEDS:
            A = predict(dict(PROD), dict(np.load(os.path.join(BASE, f'pred_{y}_{s}.npz'))))
            B = predict(dict(PROD), dict(np.load(os.path.join(cdir, f'pred_{y}_{s}.npz'))))
            ka, kb = skill(A, yv), skill(B, yv)
            d.append(kb - ka); pa.append(A); pb.append(B)
            print(f'  {y} {s:>9}: {ka:8.1f} -> {kb:8.1f}  ({kb-ka:+7.1f})')
        per_fold[y] = d; cells += d
        bag[y] = skill(np.mean(pb, 0), yv) - skill(np.mean(pa, 0), yv)

    dd = np.array(cells)
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    t = dd.mean() / se
    print('\n' + '=' * 62)
    for y in FOLDS:
        v = np.array(per_fold[y])
        print(f'  {y}: 시드평균 {v.mean():+8.1f} 양수 {(v>0).sum()}/5   배깅 {bag[y]:+8.1f}')
    bm = float(np.mean(list(bag.values())))
    print(f'\n  15셀 평균 {dd.mean():+.1f}  sd {dd.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(dd>0).sum()}/15')
    print(f'  배깅 3폴드 평균 {bm:+.1f}  전부 양수 {all(v > 0 for v in bag.values())}')
    ok = all(np.mean(per_fold[y]) > 0 for y in FOLDS) and t > 2.5
    print(f'\n  → 사전기준(3폴드 전부 양수 + t>2.5) {"충족 ✅" if ok else "미달"}')
    print(f'  → LB 노이즈 바닥(12) 대비 배깅 {bm:+.1f}')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
