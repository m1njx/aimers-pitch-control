#!/usr/bin/env python3
"""exp_distill.py — 인게임 시퀀스 교사 -> 행독립 학생 증류 (single-variable test)

가설
----
경기 내(in-game) 시퀀스 문맥(이 경기에서 이 투수가 방금 무엇을 했는가)은 실재
신호이지만(본 프로젝트 실측 ~+98점) 규정4(행 독립성) 때문에 추론 시 사용할 수
없다. 따라서 학습 시에만 쓴다:
  교사(teacher) = 프로덕션 피처 + 인과적 인게임 피처로 학습, OOF soft label 생성
  학생(student) = 프로덕션 피처만 보고 교사의 soft label 을 회귀
학생은 추론 시 행 독립이므로 규정4 준수.

조작 변수는 오직 '학습 타깃' 하나다.
  baseline : LightGBM(X119) <- hard label control_success
  student  : LightGBM(X119) <- teacher OOF soft label
피처/모델/시드/폴드 전부 동일.

설계 제약
--------
1. 엄격 인과: 인게임 피처는 같은 경기의 '앞선' 투구만 사용. 행 i 자신의 결과는
   절대 안 쓴다. causality_check() 로 명시 검증.
2. train.csv 에 game_id 가 없다 -> (pitcher_id, asof_pitcher_n) 순서로 정렬한 뒤
   (season, game_month, game_dayofweek, batter_team_id) 변화 | inning 감소 |
   asof_pitcher_prev1_game_success_rate 변화 를 경기 경계로 본다.
3. 교사 soft label 은 pitcher_id 그룹 5-fold 교차적합(OOF).
   교사 시드는 폴드당 고정이다 — hard label 이 5시드에 걸쳐 동일하듯,
   soft label 도 5시드에 걸쳐 동일해야 대칭적인 짝지은 비교가 된다.
4. 학생 피처 = 프로덕션 피처셋(X119)만. 인게임 피처는 학생에 절대 안 들어간다.

프로토콜(사전 확정, 결과 보고 변경 금지)
  forward folds: <=2020->2021, <=2021->2022, <=2022->2023
  seeds 7 123 2025 31415 8675309, 짝지은 15셀
  metric skill = 100000*(1 - mean((p-y)^2)/(ybar*(1-ybar)))
  통과 기준: 3폴드 전부 평균 양수 AND 15셀 t > 2.5

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE venv311/bin/python3 -u harness/exp_distill.py
"""
import os, sys, time, argparse, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import build_cache as bc
from exp_template import fold_data          # 폴드 정의는 템플릿 것을 그대로 재사용
from evaluate import skill

FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
K_TEACHER = 5
TEACHER_SEED = 777

LGB_P = dict(objective='regression', metric='rmse', learning_rate=0.05,
             num_leaves=31, verbose=-1, n_estimators=300, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8,
             num_threads=4, deterministic=True, force_row_wise=True)


# ===========================================================================
# 인게임(경기 내) 인과 피처
# ===========================================================================
def _game_boundary(d):
    """d 는 (pitcher_id, asof_pitcher_n) 로 정렬된 프레임. 경기 시작 행 = True."""
    p = d['pitcher_id'].values
    newp = np.r_[True, p[1:] != p[:-1]]
    b = newp.copy()
    for c in ('season', 'game_month', 'game_dayofweek', 'batter_team_id'):
        v = d[c].values
        b |= np.r_[True, v[1:] != v[:-1]]
    inn = d['inning'].values
    b |= np.r_[True, inn[1:] < inn[:-1]]                     # 이닝 역행 = 새 경기
    pg = np.nan_to_num(d['asof_pitcher_prev1_game_success_rate'].values, nan=-999.0)
    b |= np.r_[True, pg[1:] != pg[:-1]]                      # 직전경기 지표 변화
    return b


def _seg_start(boundary):
    idx = np.arange(len(boundary))
    return np.maximum.accumulate(np.where(boundary, idx, -1))


def build_ingame(df):
    """df 의 index 순서에 맞춘 인게임 인과 피처 DataFrame 을 반환.

    행 i 의 모든 값은 '같은 경기에서 i 보다 앞선 투구'만 사용한다.
    """
    order = np.lexsort((df['asof_pitcher_n'].values, df['pitcher_id'].values))
    d = df.iloc[order]
    b = _game_boundary(d)
    ss = _seg_start(b)
    pos = (np.arange(len(d)) - ss).astype(np.float64)        # 경기 내 0-based 위치
    y = d['control_success'].values.astype(np.float64)
    n = len(d)

    csum = np.cumsum(y)
    P = np.r_[0.0, csum[:-1]]                                # P[i] = sum(y[0..i-1])
    cum_prior = P - P[ss]                                    # 같은 경기 앞선 성공 합

    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.where(pos > 0, cum_prior / np.maximum(pos, 1), np.nan)

    def roll(k):
        lo = np.maximum(pos - k, 0)
        lo_abs = (ss + lo).astype(np.int64)
        s = P - P[lo_abs]
        cnt = pos - lo
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    last1 = np.where(pos > 0, np.r_[np.nan, y[:-1]], np.nan)
    career_at_start = d['asof_pitcher_success_rate'].values[ss]

    # EWMA / streak: 명시적 루프 (인과성이 눈으로 보이게: 쓰고 나서 갱신)
    ew = np.full(n, np.nan)
    st = np.zeros(n)
    alpha, cur_e, cur_s = 0.3, np.nan, 0.0
    for i in range(n):
        if b[i]:
            cur_e, cur_s = np.nan, 0.0
        ew[i] = cur_e                       # <- 읽기: 앞선 투구들만 반영된 상태
        st[i] = cur_s
        yi = y[i]                           # <- 갱신은 읽은 '뒤'에만 일어난다
        cur_e = yi if np.isnan(cur_e) else alpha * yi + (1 - alpha) * cur_e
        if cur_s == 0 or (cur_s > 0) != (yi > 0.5):
            cur_s = 1.0 if yi > 0.5 else -1.0
        else:
            cur_s += 1.0 if yi > 0.5 else -1.0

    bat = d['batter_id'].values
    newbat = (np.r_[True, bat[1:] != bat[:-1]] | b).astype(np.float64)
    cnb = np.cumsum(newbat)
    pa = cnb - cnb[ss]

    inn = d['inning'].values
    innb = b | np.r_[True, inn[1:] != inn[:-1]]
    pii = (np.arange(n) - _seg_start(innb)).astype(np.float64)

    out = pd.DataFrame({
        'ig_n_prior': pos,
        'ig_succ_rate': rate,
        'ig_succ_vs_career': rate - career_at_start,
        'ig_last1': last1,
        'ig_last3': roll(3),
        'ig_last5': roll(5),
        'ig_last10': roll(10),
        'ig_ewma': ew,
        'ig_streak': st,
        'ig_pa_idx': pa,
        'ig_pitch_in_inning': pii,
        'ig_inning_rel': (inn - inn[ss]).astype(np.float64),
    }, index=d.index)
    return out.loc[df.index], int(b.sum())


IG_COLS = ['ig_n_prior', 'ig_succ_rate', 'ig_succ_vs_career', 'ig_last1',
           'ig_last3', 'ig_last5', 'ig_last10', 'ig_ewma', 'ig_streak',
           'ig_pa_idx', 'ig_pitch_in_inning', 'ig_inning_rel']


# ===========================================================================
# 인과성 검증 (제약 #1)
# ===========================================================================
def _eqmat(A, B):
    a, b = A.values, B.values
    return (a == b) | (np.isnan(a) & np.isnan(b))


def causality_check(df):
    """행 i 의 인게임 피처가 행 i 자신의 라벨에 의존하지 않음을 증명한다.

    테스트 A (off-by-one 탐지): 각 경기의 '마지막' 행 라벨만 뒤집는다. 어떤 행도
      뒤집힌 행보다 뒤에 있지 않으므로, 인과적이라면 인게임 행렬 전체가 완전히
      동일해야 한다. 한 칸이라도 밀렸으면(자기 라벨을 썼으면) 바로 차이가 난다.
    테스트 B (양성 대조): 각 경기의 '첫' 행 라벨을 뒤집는다. 첫 행 자체의 피처는
      불변이어야 하고, 그 뒤 행들의 피처는 실제로 바뀌어야 한다
      (= 피처가 정말 앞선 결과를 담고 있다는 증거).
    """
    print('\n' + '=' * 64)
    print('인과성 검증 (제약 #1)')
    A, ng = build_ingame(df)
    order = np.lexsort((df['asof_pitcher_n'].values, df['pitcher_id'].values))
    d = df.iloc[order]
    b = _game_boundary(d)
    last_idx = d.index[np.r_[b[1:], True]]                   # 경기 마지막 행
    first_idx = d.index[b]
    print(f'  행 {len(df):,}  경기 {ng:,}  경기당 중앙값 '
          f'{np.median(np.diff(np.r_[np.flatnonzero(b), len(b)])):.0f} 투구')

    df2 = df.copy()
    df2.loc[last_idx, 'control_success'] = 1 - df2.loc[last_idx, 'control_success']
    B, _ = build_ingame(df2)
    ndiff = int((~_eqmat(A, B)).sum())
    print(f'  테스트 A (경기 마지막 행 {len(last_idx):,}개 라벨 flip): '
          f'인게임 행렬이 다른 셀 = {ndiff}  (0 이어야 통과)')

    df3 = df.copy()
    df3.loc[first_idx, 'control_success'] = 1 - df3.loc[first_idx, 'control_success']
    C, _ = build_ingame(df3)
    neq = ~_eqmat(A, C)
    fmask = df.index.isin(first_idx)
    n_first = int(neq[fmask].any(axis=1).sum())
    n_other = int(neq[~fmask].any(axis=1).sum())
    print(f'  테스트 B (경기 첫 행 {len(first_idx):,}개 라벨 flip): '
          f'첫 행 중 피처 바뀐 행 {n_first} (0 이어야 함) / '
          f'나머지 행 중 바뀐 행 {n_other:,} (>0 이어야 함)')
    ok = (ndiff == 0) and n_first == 0 and n_other > 0
    print(f'  -> 인과성 {"통과 OK" if ok else "실패 FAIL"}')
    return ok


# ===========================================================================
def fit_predict(Xtr, ytr, Xva, seed):
    p = dict(LGB_P); p['seed'] = seed
    m = lgb.train(p, lgb.Dataset(Xtr, label=ytr))
    return m.predict(Xva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=FOLDS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--no-diag', action='store_true')
    ap.add_argument('--control', action='store_true',
                    help='사후 대조군: 교사에서 인게임 피처를 빼고(자기증류) 동일 절차. '
                         '이득이 특권 신호에서 온 것인지 soft-label 평활 자체에서 온 '
                         '것인지 분리한다. 사전 확정 기준의 일부가 아니다.')
    a = ap.parse_args()
    sfx = '_ctrl' if a.control else ''

    t0 = time.time()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    print(f'train {df.shape}  ({time.time()-t0:.0f}s)', flush=True)

    if not causality_check(df[df.season <= 2020].reset_index(drop=True)):
        print('인과성 검증 실패 -> 중단'); return
    if a.check_only:
        return

    outdir = os.path.join(LG, 'harness/cache_distill')
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for y in a.years:
        past, va, prep, dec, cat_map = fold_data(df, y)
        past = past.reset_index(drop=True)
        Xpa_df, _ = bc.build_features(past, prep, dec, cat_map)
        Xva_df, _ = bc.build_features(va, prep, dec, cat_map)
        cols = list(Xpa_df.columns)
        Xpa = Xpa_df.values.astype(np.float32)
        Xva = Xva_df.values.astype(np.float32)
        del Xpa_df, Xva_df; gc.collect()

        ypa = past['control_success'].values.astype(np.float64)
        yva = va['control_success'].values.astype(np.float64)
        IG, ng = build_ingame(past)
        # --control 이면 교사도 X119 만 본다(자기증류 대조군).
        Xt = Xpa if a.control else np.hstack(
            [Xpa, IG[IG_COLS].values.astype(np.float32)])
        groups = past['pitcher_id'].values
        print(f'\n=== eval {y}: past {len(past):,} ({ng:,} games)  va {len(va):,}  '
              f'X {Xpa.shape[1]} / teacher {Xt.shape[1]}  ({time.time()-t0:.0f}s) ===',
              flush=True)
        del IG; gc.collect()

        # ---- 교사 OOF soft label (pitcher_id 그룹 5-fold 교차적합) ----
        fo = os.path.join(outdir, f'oof{sfx}_{y}.npy')
        if os.path.exists(fo):
            oof = np.load(fo).astype(np.float64)
            print(f'  teacher OOF cached', flush=True)
        else:
            t1 = time.time()
            oof = np.zeros(len(past))
            for ki, (tr, te) in enumerate(GroupKFold(n_splits=K_TEACHER)
                                          .split(Xt, ypa, groups)):
                oof[te] = fit_predict(Xt[tr], ypa[tr], Xt[te], TEACHER_SEED + ki)
                print(f'    teacher k{ki}: train {len(tr):,} -> oof {len(te):,} '
                      f'({time.time()-t1:.0f}s)', flush=True)
            oof = np.clip(oof, 1e-6, 1 - 1e-6)
            np.save(fo, oof.astype(np.float64))
        print(f'  OOF soft label: mean {oof.mean():.4f} sd {oof.std():.4f} '
              f'(hard mean {ypa.mean():.4f});  OOF skill(train rows) '
              f'{skill(oof, ypa):.1f}', flush=True)

        # ---- 진단(참고용, 규정4 위반이므로 제출 불가): 교사를 va 에 직접 적용 ----
        if not a.no_diag and not a.control:
            fd = os.path.join(outdir, f'tdiag_{y}.npy')
            if os.path.exists(fd):
                td = float(np.load(fd)[0])
            else:
                IGv, _ = build_ingame(va)
                Xtv = np.hstack([Xva, IGv[IG_COLS].values.astype(np.float32)])
                td = skill(fit_predict(Xt, ypa, Xtv, TEACHER_SEED), yva)
                np.save(fd, np.array([td]))
                del IGv, Xtv; gc.collect()
            print(f'  [진단] teacher(인게임 포함) {y} 직접 skill = {td:.1f}', flush=True)

        # ---- baseline vs student: 타깃만 다르다 ----
        for s in a.seeds:
            f = os.path.join(outdir, f'preds{sfx}_{y}_{s}.npz')
            if os.path.exists(f):
                z = np.load(f); pb, ps = z['base'], z['stud']
            else:
                t1 = time.time()
                fbase = os.path.join(outdir, f'preds_{y}_{s}.npz')
                # baseline 은 조작 변수와 무관하므로(동일 피처/시드/타깃) 재사용한다
                pb = (np.load(fbase)['base'] if os.path.exists(fbase)
                      else fit_predict(Xpa, ypa, Xva, s))
                ps = fit_predict(Xpa, oof, Xva, s)
                np.savez_compressed(f, base=pb, stud=ps)
                print(f'    seed {s} fit ({time.time()-t1:.0f}s)', flush=True)
            kb, ks = skill(pb, yva), skill(ps, yva)
            rows.append((y, s, kb, ks, ks - kb))
            print(f'  {y} seed {s:>8}:  baseline {kb:9.1f}  student {ks:9.1f}  '
                  f'delta {ks-kb:+9.1f}', flush=True)

        del Xt, Xpa, Xva, past, va, oof; gc.collect()

    # ------------------------------------------------------------------
    R = pd.DataFrame(rows, columns=['fold', 'seed', 'base', 'stud', 'delta'])
    print('\n' + '=' * 72)
    print(R.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    print('\n폴드별')
    for y in sorted(R.fold.unique()):
        v = R[R.fold == y]['delta'].values
        print(f'  {y}: 평균 {v.mean():+9.1f}   양수 {(v > 0).sum()}/{len(v)}   '
              f'sd {v.std(ddof=1):.1f}')
    d = R['delta'].values
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se
    print(f'\n{len(d)}셀: 평균 {d.mean():+.1f}  sd {d.std(ddof=1):.1f}  SE {se:.1f}  '
          f't={t:.2f}  양수 {(d > 0).sum()}/{len(d)}')
    allpos = all(R[R.fold == y]['delta'].mean() > 0 for y in R.fold.unique())
    ok = allpos and t > 2.5 and len(R.fold.unique()) == 3
    print(f'3폴드 전부 양수 = {allpos},  t>2.5 = {t > 2.5}')
    print(f'\n=> 사전 확정 기준 판정: {"통과 PASS" if ok else "미달 FAIL"}')
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
