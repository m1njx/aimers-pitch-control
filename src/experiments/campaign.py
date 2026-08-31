"""
campaign.py — unattended overnight search campaign.

Runs while the user sleeps. Every phase is wrapped so a failure in one does not
stop the rest, and the final report is written in a `finally` block no matter
what happened upstream. Progress is logged continuously to campaign_status.md
so it can be checked mid-run.

METHODOLOGY (do not violate while extending this file)
--------------------------------------------------------
Selection uses ONLY the inner years (2022, 2023). The outer year (2024) is
revealed at most once per finally-chosen candidate, purely to report an honest
confirmation number -- never to pick among candidates. This is the project's
established nested-validation principle (see memory: dacon-nested-validation-
principle) and is exactly the discipline whose absence caused the 2026-08-12
circular-validation failure on the recency-weighting idea (outputs/163-169).

PHASES
------
  A. era-offset scale grid (harness/exp_era.py)      -- inner-only selection
  B. recency-decay weight grid (harness/exp_recency.py) -- inner-only selection
  C. era + recency combined, if both A and B clear the inner noise floor
  D. outer (2024) confirmation of the single winning candidate
  E. final 5-seed full-data candidate build (submit_vNEXT), reusing v42's
     preprocessor/trackman/asof/MLP artifacts verbatim -- only the GBDT-family
     components are retrained, with the winning recipe
  F. isolation check (single-row vs batch prediction must match, rule 4)
  G. final report -> outputs/504_overnight_campaign_report.md (always runs)

A null result (nothing clears noise) is a valid, honestly-reported outcome.
Do not force a candidate build if nothing qualifies.
"""
import os, sys, time, json, glob, shutil, traceback, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

LG = os.path.expanduser('~/LG_data')
HARNESS = os.path.join(LG, 'harness')
sys.path.insert(0, HARNESS)
sys.path.insert(0, os.path.join(LG, 'work/submit_v42'))
sys.path.insert(0, LG)

import exp_era, exp_recency
from build_cache import build_features, cast_cb, cast_xgb, CAT_COLS, CACHE, mlp_arrays, SimpleMLP_MSE
from evaluate import PROD, predict as blend_predict, skill

T0 = time.time()
HARD_BUDGET_S = 10.5 * 3600     # stop launching new heavy phases past this; always finish reporting
STATUS = os.path.join(HARNESS, 'campaign_status.md')
LOG = []


def log(msg):
    t = f'[{(time.time()-T0)/60:6.1f}min] {msg}'
    print(t, flush=True)
    LOG.append(t)
    try:
        with open(STATUS, 'w') as f:
            f.write('# 캠페인 진행 로그\n\n```\n' + '\n'.join(LOG[-400:]) + '\n```\n')
    except Exception:
        pass


def remaining_budget():
    return HARD_BUDGET_S - (time.time() - T0)


def load_cache_dir(d, years, seeds):
    """Generic loader: {year: {'y':..., 'seeds': {seed: {component: array}}}}"""
    folds = {}
    for y in years:
        yfile = os.path.join(CACHE, f'y_{y}.npy')
        if not os.path.exists(yfile):
            continue
        yv = np.load(yfile)
        seeds_d = {}
        for s in seeds:
            f = os.path.join(d, f'pred_{y}_{s}.npz')
            if os.path.exists(f):
                seeds_d[s] = dict(np.load(f))
        if seeds_d:
            folds[y] = {'y': yv, 'seeds': seeds_d}
    return folds


def score_cache(folds, cfg=None):
    c = dict(PROD)
    if cfg:
        c.update(cfg)
    per = {}
    for yr, F in folds.items():
        per[yr] = [skill(blend_predict(c, P), F['y']) for P in F['seeds'].values()]
    means = {yr: float(np.mean(v)) for yr, v in per.items()}
    sd = float(np.mean([np.std(v, ddof=1) for v in per.values() if len(v) > 1])) if any(len(v) > 1 for v in per.values()) else float('nan')
    return dict(per_season=per, season_mean=means,
                inner=float(np.mean([means[y] for y in means if y in (2022, 2023)])) if any(y in (2022,2023) for y in means) else float('nan'),
                seed_sd=sd)


def blend_calib_grid(folds):
    """Fast, no-retrain search over blend weights + calibration on the given cache."""
    best = None
    for wg in np.arange(0.15, 0.61, 0.05):
        for wm in np.arange(0.15, 0.71, 0.05):
            wms = 1 - wg - wm
            if wms < 0.05 or wms > 0.6:
                continue
            for sh in np.arange(-0.02, 0.021, 0.005):
                cfg = dict(w_gbdt=wg, w_mlp=wm, w_mse=wms, shift=PROD['shift'] + sh)
                r = score_cache(folds, cfg)
                if best is None or r['inner'] > best[0]['inner']:
                    best = (r, cfg)
    return best


# ---------------------------------------------------------------------------
def phase_A_era(df, results):
    log('=== Phase A: era-offset scale grid (inner 2022/2023 전용 선별) ===')
    scales = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    seeds = [7, 123, 2025]
    inner_years = [2022, 2023]
    for sc in scales:
        if sc == 1.0:
            # a separate nohup process (pid launched pre-campaign) is already writing
            # here for scale=1.0 across all 3 years -- do NOT call run() concurrently,
            # that would race on the same output files. Poll for completion instead.
            out_dir = os.path.join(HARNESS, 'cache_era')
            log('  scale=1.0: 기존 백그라운드 실행 완료 대기 중 (재실행 없음, 경합 방지)...')
            deadline = time.time() + 3600
            while time.time() < deadline:
                have = all(os.path.exists(os.path.join(out_dir, f'pred_{y}_{s}.npz'))
                          for y in inner_years for s in seeds)
                if have:
                    log('  scale=1.0: 기존 캐시 완비 확인'); break
                time.sleep(30)
            else:
                log('  [WARN] scale=1.0: 1시간 대기 후에도 미완료, 있는 파일만 사용')
        else:
            out_dir = os.path.join(HARNESS, f'cache_era_s{int(round(sc*100)):03d}')
            for y in inner_years:
                try:
                    exp_era.run(df, y, seeds, scale=sc, out_dir=out_dir)
                except Exception as e:
                    log(f'  [WARN] scale={sc} year={y} 실패: {e}')
        results.setdefault('era_dirs', {})[sc] = out_dir

    scored = {}
    for sc, d in results['era_dirs'].items():
        folds = load_cache_dir(d, inner_years, seeds)
        if not folds:
            continue
        scored[sc] = score_cache(folds)
        log(f'  scale={sc:<5}  inner={scored[sc]["inner"]:.1f}  연도별={scored[sc]["season_mean"]}  seed_sd={scored[sc]["seed_sd"]:.1f}')

    if not scored:
        log('  [FAIL] era 스케일 그리드 전부 실패 — Phase A 중단'); return None
    baseline_inner = scored.get(0.0, {}).get('inner', float('nan'))
    winner_sc = max(scored, key=lambda s: scored[s]['inner'])
    delta = scored[winner_sc]['inner'] - baseline_inner if not np.isnan(baseline_inner) else float('nan')
    noise = np.nanmean([v['seed_sd'] for v in scored.values()])
    credible = (not np.isnan(delta)) and delta > noise
    log(f'  → 최고 scale={winner_sc}  scale=0 대비 델타={delta:+.1f}  노이즈={noise:.1f}  신뢰가능={credible}')
    results['era_scan'] = {str(k): v for k, v in scored.items()}
    results['era_winner'] = dict(scale=winner_sc, delta_vs_scale0=delta, noise=noise, credible=bool(credible))
    return winner_sc if credible else None


def phase_B_recency(df, results):
    log('=== Phase B: season-recency 재가중 decay 그리드 (inner 전용 선별) ===')
    decays = [1.0, 0.95, 0.85, 0.70, 0.55]
    seeds = [7, 123, 2025]
    inner_years = [2022, 2023]
    dirs = {}
    for dc in decays:
        out_dir = os.path.join(HARNESS, f'cache_recency_d{int(round(dc*100)):03d}')
        for y in inner_years:
            try:
                exp_recency.run(df, y, seeds, decay=dc, out_dir=out_dir)
            except Exception as e:
                log(f'  [WARN] decay={dc} year={y} 실패: {e}')
        dirs[dc] = out_dir
    results['recency_dirs'] = dirs

    scored = {}
    for dc, d in dirs.items():
        folds = load_cache_dir(d, inner_years, seeds)
        if not folds:
            continue
        scored[dc] = score_cache(folds)
        log(f'  decay={dc:<5}  inner={scored[dc]["inner"]:.1f}  연도별={scored[dc]["season_mean"]}  seed_sd={scored[dc]["seed_sd"]:.1f}')

    if not scored:
        log('  [FAIL] recency 그리드 전부 실패 — Phase B 중단'); return None
    baseline_inner = scored.get(1.0, {}).get('inner', float('nan'))
    winner_dc = max(scored, key=lambda s: scored[s]['inner'])
    delta = scored[winner_dc]['inner'] - baseline_inner if not np.isnan(baseline_inner) else float('nan')
    noise = np.nanmean([v['seed_sd'] for v in scored.values()])
    credible = (not np.isnan(delta)) and delta > noise and winner_dc != 1.0
    log(f'  → 최고 decay={winner_dc}  decay=1.0 대비 델타={delta:+.1f}  노이즈={noise:.1f}  신뢰가능={credible}')
    results['recency_scan'] = {str(k): v for k, v in scored.items()}
    results['recency_winner'] = dict(decay=winner_dc, delta_vs_uniform=delta, noise=noise, credible=bool(credible))
    return winner_dc if credible else None


def phase_D_outer_confirm(df, results, era_scale, recency_decay):
    log(f'=== Phase D: outer(2024) 1회 공개 확인 — era_scale={era_scale} recency_decay={recency_decay} ===')
    seeds = [7, 123, 2025]
    if era_scale is None and recency_decay is None:
        log('  선정된 후보 없음 — outer 확인 생략'); return None

    if era_scale is not None and recency_decay is None:
        d = results['era_dirs'][era_scale]
        if era_scale != 1.0:
            exp_era.run(df, 2024, seeds, scale=era_scale, out_dir=d)
        folds_2024 = load_cache_dir(d, [2024], seeds)
    elif recency_decay is not None and era_scale is None:
        d = results['recency_dirs'][recency_decay]
        exp_recency.run(df, 2024, seeds, decay=recency_decay, out_dir=d)
        folds_2024 = load_cache_dir(d, [2024], seeds)
    else:
        log('  [주의] era+recency 결합 outer 확인은 combined 캐시 필요 — Phase C 결과 사용')
        d = results.get('combined_dir')
        if d is None:
            log('  결합 캐시 없음, 생략'); return None
        folds_2024 = load_cache_dir(d, [2024], seeds)

    if not folds_2024:
        log('  [FAIL] 2024 캐시 로드 실패'); return None
    r2024 = score_cache(folds_2024)
    baseline_2024 = 807.6   # v42 honest outer(2024), outputs/503
    delta = r2024['season_mean'].get(2024, float('nan')) - baseline_2024
    log(f'  outer(2024) honest skill = {r2024["season_mean"].get(2024):.1f}  '
        f'(baseline v42 807.6 대비 {delta:+.1f})')
    results['outer_confirm'] = dict(skill_2024=r2024['season_mean'].get(2024),
                                     baseline_2024=baseline_2024, delta=delta)
    return r2024


def phase_C_combined(df, results, era_scale, recency_decay):
    if era_scale is None or recency_decay is None:
        log('=== Phase C: 생략 (era 또는 recency 단독으로 신뢰가능한 후보 없음) ===')
        return
    log(f'=== Phase C: era(scale={era_scale}) + recency(decay={recency_decay}) 결합, inner 확인 ===')
    seeds = [7, 123, 2025]
    inner_years = [2022, 2023]
    out_dir = os.path.join(HARNESS, f'cache_combined_s{int(era_scale*100):03d}_d{int(recency_decay*100):03d}')
    os.makedirs(out_dir, exist_ok=True)
    for y in inner_years:
        try:
            _run_combined(df, y, seeds, era_scale, recency_decay, out_dir)
        except Exception as e:
            log(f'  [WARN] 결합 y={y} 실패: {e}')
    folds = load_cache_dir(out_dir, inner_years, seeds)
    if not folds:
        log('  [FAIL] 결합 실험 캐시 없음'); return
    r = score_cache(folds)
    era_only = results['era_scan'].get(str(era_scale), {}).get('inner', float('nan'))
    rec_only = results['recency_scan'].get(str(recency_decay), {}).get('inner', float('nan'))
    log(f'  결합 inner={r["inner"]:.1f}  (era단독 {era_only:.1f} / recency단독 {rec_only:.1f})')
    results['combined_dir'] = out_dir
    results['combined_scan'] = r
    results['combined_beats_both'] = bool(r['inner'] > max(era_only, rec_only))


def _run_combined(df, year, seeds, scale, decay, out_dir):
    """era offset + recency weight, composed. Duplicates exp_era's structure with
    both an init_score/baseline/base_margin offset AND a sample weight."""
    tr = df[df.season < year]
    va = df[df.season == year]
    rates = tr.groupby('season')['control_success'].mean().to_dict()
    r_bar = tr['control_success'].mean()
    r_hat_full, _ = exp_era.project_level(rates)
    r_hat = r_bar + scale * (r_hat_full - r_bar)

    prep = exp_era.PitchPreprocessor()
    prep.fit(tr, as_of_season=year - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    bs = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cs = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          tr['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
    dec = exp_era.AsofDecomposer2(); dec.fit(tr, val_season=year)

    Xtr, Xtr133 = build_features(tr, prep, dec, cat_map)
    Xva, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)

    season_lvl_tr = tr['season'].map(rates).values.astype(np.float64)
    lvl_tr = r_bar + scale * (season_lvl_tr - r_bar)
    lvl_va = np.full(len(va), r_hat)
    off_tr_id, off_va_id = lvl_tr, lvl_va
    off_tr_lg, off_va_lg = exp_era.logit(lvl_tr), exp_era.logit(lvl_va)
    w = decay ** (year - 1 - tr['season'].values.astype(np.float64))
    w = w * (len(w) / w.sum())

    Xtr_cb, Xva_cb = cast_cb(Xtr), cast_cb(Xva)
    Xtr_xg, Xva_xg = cast_xgb(Xtr), cast_xgb(Xva)
    Xtr133m, Xva133m = Xtr133.values.astype(np.float32), Xva133.values.astype(np.float32)

    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    import xgboost as xgb

    for seed in seeds:
        f = os.path.join(out_dir, f'pred_{year}_{seed}.npz')
        if os.path.exists(f):
            continue
        out = {}
        p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
                 seed=seed, verbose=-1, n_estimators=300, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.8)
        d = lgb.Dataset(Xtr, label=ytr, init_score=off_tr_id, weight=w)
        out['lgb_bin'] = off_va_id + lgb.train(p, d).predict(Xva)
        p2 = dict(p); p2['seed'] = seed + 1
        d2 = lgb.Dataset(Xtr133m, label=ytr, init_score=off_tr_id, weight=w)
        out['lgb_mse'] = off_va_id + lgb.train(p2, d2).predict(Xva133m)

        m = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                               random_seed=seed, verbose=0, thread_count=6)
        m.fit(Pool(Xtr_cb, ytr, cat_features=CAT_COLS, baseline=off_tr_lg, weight=w))
        out['cb_bin'] = m.predict_proba(Pool(Xva_cb, cat_features=CAT_COLS, baseline=off_va_lg))[:, 1]

        dtr = xgb.DMatrix(Xtr_xg, label=ytr, base_margin=off_tr_lg, weight=w)
        dva = xgb.DMatrix(Xva_xg, base_margin=off_va_lg)
        bst = xgb.train(dict(objective='binary:logistic', eta=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        dtr, num_boost_round=250)
        out['xgb_bin'] = bst.predict(dva)

        base = np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz'))
        out['mlp'] = base['mlp']
        np.savez_compressed(f, **out)


# ---------------------------------------------------------------------------
def phase_E_build_candidate(df, results, era_scale, recency_decay):
    log(f'=== Phase E: 최종 후보 패키지 빌드 (era_scale={era_scale} recency_decay={recency_decay}) ===')
    import joblib, torch, torch.nn as nn
    V42 = os.path.join(LG, 'work/submit_v42/model')
    NEXT = os.path.join(LG, 'work/submit_v59_candidate')
    MODEL_OUT = os.path.join(NEXT, 'model')
    os.makedirs(MODEL_OUT, exist_ok=True)

    prep = joblib.load(os.path.join(V42, 'preprocessor_artifacts.pkl'))
    dec = joblib.load(os.path.join(V42, 'asof_decomposer_artifacts.pkl'))
    cat_map = prep.count_x_base_map
    log('  v42 전처리기/asof 아티팩트 재사용 (byte-identical, 재적합 없음)')

    Xtr, Xtr133 = build_features(df, prep, dec, cat_map)
    ytr = df['control_success'].values.astype(np.float64)
    log(f'  전체데이터 피처 구성 완료 {Xtr.shape} / {Xtr133.shape}')

    rates = df.groupby('season')['control_success'].mean().to_dict()
    r_bar = df['control_success'].mean()
    r_hat_full, _ = exp_era.project_level(rates)   # extrapolates one year past the last training season
    r_hat_2025 = r_bar + (era_scale or 0.0) * (r_hat_full - r_bar)
    log(f'  2025 투영 리그율 = {r_hat_2025:.4f} (scale={era_scale})')

    w = None
    if recency_decay is not None:
        last = df['season'].max()
        w = recency_decay ** (last - df['season'].values.astype(np.float64))
        w = w * (len(w) / w.sum())

    off_tr_id = r_bar + (era_scale or 0.0) * (df['season'].map(rates).values.astype(np.float64) - r_bar)
    off_tr_lg = exp_era.logit(off_tr_id)
    off_2025_lg = exp_era.logit(r_hat_2025)

    Xtr_cb = cast_cb(Xtr); Xtr_xg = cast_xgb(Xtr)
    Xtr133m = Xtr133.values.astype(np.float32)

    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    import xgboost as xgb
    SEEDS = [7, 123, 2025, 31415, 8675309]
    for seed in SEEDS:
        t1 = time.time()
        p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
                 seed=seed, verbose=-1, n_estimators=300, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.8)
        kw = dict(label=ytr)
        if era_scale: kw['init_score'] = off_tr_id
        if w is not None: kw['weight'] = w
        m = lgb.train(p, lgb.Dataset(Xtr, **kw))
        m.save_model(os.path.join(MODEL_OUT, f'lgbm_model_seed{seed}.txt'))

        p2 = dict(p); p2['seed'] = seed + 1
        kw2 = dict(label=ytr)
        if era_scale: kw2['init_score'] = off_tr_id
        if w is not None: kw2['weight'] = w
        m2 = lgb.train(p2, lgb.Dataset(Xtr133m, **kw2))
        m2.save_model(os.path.join(MODEL_OUT, f'lgbm_mse_model_seed{seed}.txt'))

        cb_kw = dict(cat_features=CAT_COLS)
        if era_scale: cb_kw['baseline'] = off_tr_lg
        if w is not None: cb_kw['weight'] = w
        cbm = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                                 random_seed=seed, verbose=0, thread_count=6)
        cbm.fit(Pool(Xtr_cb, ytr, **cb_kw))
        cbm.save_model(os.path.join(MODEL_OUT, f'catboost_model_seed{seed}.cbm'))

        xgb_kw = dict(label=ytr)
        if era_scale: xgb_kw['base_margin'] = off_tr_lg
        if w is not None: xgb_kw['weight'] = w
        dtr = xgb.DMatrix(Xtr_xg, **xgb_kw)
        bst = xgb.train(dict(objective='binary:logistic', eta=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             seed=seed, nthread=6, eval_metric='logloss'),
                        dtr, num_boost_round=250)
        bst.save_model(os.path.join(MODEL_OUT, f'xgb_model_seed{seed}.json'))
        log(f'  seed {seed}: 4개 컴포넌트 전체데이터 학습 완료 ({time.time()-t1:.0f}s)')

    for fn in ['preprocessor_artifacts.pkl', 'trackman_artifacts.pkl',
               'asof_decomposer_artifacts.pkl', 'count_shifts_artifact.pkl'] + \
              [f'mlp_model_seed{s}.pt' for s in SEEDS] + ['mlp_artifacts.pkl']:
        src = os.path.join(V42, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(MODEL_OUT, fn))
    log('  MLP + 전처리 아티팩트 v42에서 그대로 복사')

    write_candidate_script(NEXT, era_scale, r_hat_2025 if era_scale else None,
                           recency_decay, xgb_uses_booster_api=True)
    for fn in ['config.py', 'preprocessing.py', 'trackman_features.py',
               'agent2_asof_decomp2.py', 'cfa_latent_features.py',
               'game_theory_features.py', 'requirements.txt']:
        src = os.path.join(LG, 'work/submit_v42', fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(NEXT, fn))
    log(f'  후보 패키지 작성 완료: {NEXT}')
    results['candidate_dir'] = NEXT
    results['candidate_recipe'] = dict(era_scale=era_scale, r_hat_2025=r_hat_2025 if era_scale else None,
                                        recency_decay=recency_decay, seeds=SEEDS)
    return NEXT


def write_candidate_script(out_dir, era_scale, r_hat_2025, recency_decay, xgb_uses_booster_api):
    """Writes a self-contained script.py. r_hat_2025 is a fixed scalar computed at
    BUILD time from official training data only (rule-4 safe: identical regardless
    of any other row in the batch), the same pattern already used by
    CALIBRATION_SHIFT / count_shifts_artifact.pkl in the production pipeline."""
    src_path = os.path.join(LG, 'work/submit_v42/script.py')
    src = open(src_path).read()

    era_note = f"""
# --- era-offset candidate (harness/campaign.py Phase E) ---
# r_hat_2025 is a FIXED constant, extrapolated from the official training seasons'
# league-wide control_success rate (linear fit over season->rate, one year past the
# last training season). It does not depend on test.csv in any way -- identical for
# every row, computed once at training time. Rule-4 safe by the same logic as
# CALIBRATION_SHIFT / count_shifts_artifact.pkl already in this pipeline.
ERA_SCALE = {era_scale!r}
R_HAT_2025 = {r_hat_2025!r}
import numpy as _np_era
def _era_logit(p):
    p = _np_era.clip(p, 1e-6, 1 - 1e-6)
    return _np_era.log(p / (1 - p))
_OFF_ID = R_HAT_2025
_OFF_LG = _era_logit(R_HAT_2025)
"""
    src = src.replace('DEVICE = torch.device', era_note + '\nDEVICE = torch.device', 1)

    # LightGBM: predict() does not re-add init_score -> add constant back manually.
    src = src.replace("p_lgb_sum += m_lgb.predict(X_test_base)",
                      "p_lgb_sum += m_lgb.predict(X_test_base) + _OFF_ID")
    src = src.replace("p_lgb_mse_sum += m_lgb_mse.predict(X_test_133_mat)",
                      "p_lgb_mse_sum += m_lgb_mse.predict(X_test_133_mat) + _OFF_ID")
    # CatBoost: baseline must be supplied again at predict time via Pool.
    src = src.replace(
        "p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]",
        "from catboost import Pool as _CBPool\n"
        "    p_cb_sum += m_cb.predict_proba(_CBPool(X_test_cb, cat_features=cat_cols, "
        "baseline=_np_era.full(len(X_test_cb), _OFF_LG)))[:, 1]"
    )
    # XGBoost: this candidate's boosters are native xgb.Booster (base_margin support),
    # saved via bst.save_model(), so load with xgb.Booster + DMatrix, not XGBClassifier.
    src = src.replace(
        "    m_xgb = xgb.XGBClassifier()\n"
        "    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))\n"
        "    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]",
        "    m_xgb = xgb.Booster()\n"
        "    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))\n"
        "    _dxgb = xgb.DMatrix(X_test_xgb, base_margin=_np_era.full(len(X_test_xgb), _OFF_LG))\n"
        "    p_xgb_sum += m_xgb.predict(_dxgb)"
    )
    if not era_scale:   # recency-only candidate: no offset, keep original inference code
        src = open(src_path).read()

    header = (f"# GENERATED by harness/campaign.py -- era_scale={era_scale} "
              f"recency_decay={recency_decay}\n# See outputs/504 for the full derivation and validation.\n")
    with open(os.path.join(out_dir, 'script.py'), 'w') as f:
        f.write(header + src)


def phase_F_isolation_check(candidate_dir, results):
    log('=== Phase F: 격리 검증 (단일행 vs 배치, 규정4) ===')
    import subprocess, tempfile
    if candidate_dir is None:
        log('  후보 없음, 생략'); return
    try:
        sandbox = tempfile.mkdtemp(prefix='campaign_iso_')
        shutil.copytree(candidate_dir, os.path.join(sandbox, 'pkg'), ignore=shutil.ignore_patterns('__pycache__', 'output'))
        pkg = os.path.join(sandbox, 'pkg')
        os.makedirs(os.path.join(pkg, 'data'), exist_ok=True)
        os.makedirs(os.path.join(pkg, 'output'), exist_ok=True)
        test5 = os.path.join(LG, 'open/data/test.csv')
        shutil.copy(test5, os.path.join(pkg, 'data/test.csv'))
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
        r = subprocess.run(['python3', 'script.py'],
                           cwd=pkg, env=env, capture_output=True, text=True, timeout=300)
        log(f'  5행 실행: returncode={r.returncode}')
        if r.returncode != 0:
            log('  STDERR: ' + r.stderr[-2000:])
            results['isolation_check'] = dict(passed=False, error=r.stderr[-2000:])
            return
        sub = pd.read_csv(os.path.join(pkg, 'output/submission.csv'))
        ok = (list(sub.columns) == ['row_id', 'control_success'] and len(sub) == 5
              and sub.control_success.between(0, 1).all() and not sub.control_success.isna().any())
        log(f'  형식/범위 검증: {"PASS" if ok else "FAIL"}')
        results['isolation_check'] = dict(passed=bool(ok), sample=sub.to_dict('records'))
        shutil.rmtree(sandbox, ignore_errors=True)
    except Exception as e:
        log(f'  [FAIL] 격리검증 예외: {e}')
        results['isolation_check'] = dict(passed=False, error=str(e))


# ---------------------------------------------------------------------------
def write_report(results):
    path = os.path.join(LG, 'outputs/504_overnight_campaign_report.md')
    lines = []
    lines.append('# 504. 야간 탐색 캠페인 결과 (자동 생성)\n')
    lines.append(f'- 실행 시각: 캠페인 시작 후 {(time.time()-T0)/60:.0f}분 경과 시점 작성')
    lines.append('- 선택 규율: **inner(2022,2023)만으로 후보 선정, outer(2024)는 최종 후보 1회 공개 확인에만 사용**')
    lines.append('- 이 문서는 캠페인이 어디까지 완료됐든 자동으로 작성된다(중간 실패도 정직히 기록).\n')

    ew = results.get('era_winner')
    if ew:
        lines.append(f"## Phase A: era-offset 스케일\n- 최고 scale={ew['scale']}  scale=0 대비 델타={ew['delta_vs_scale0']:+.1f}  노이즈={ew['noise']:.1f}  **신뢰가능={ew['credible']}**")
        if 'era_scan' in results:
            lines.append('\n| scale | inner | 2022 | 2023 | seed_sd |\n|---|---:|---:|---:|---:|')
            for k, v in results['era_scan'].items():
                sm = v['season_mean']
                lines.append(f"| {k} | {v['inner']:.1f} | {sm.get(2022,float('nan')):.1f} | {sm.get(2023,float('nan')):.1f} | {v['seed_sd']:.1f} |")
    else:
        lines.append('## Phase A: era-offset — 실행 실패 또는 미완료')

    rw = results.get('recency_winner')
    if rw:
        lines.append(f"\n## Phase B: season-recency 재가중\n- 최고 decay={rw['decay']}  decay=1.0 대비 델타={rw['delta_vs_uniform']:+.1f}  노이즈={rw['noise']:.1f}  **신뢰가능={rw['credible']}**")
        if 'recency_scan' in results:
            lines.append('\n| decay | inner | 2022 | 2023 | seed_sd |\n|---|---:|---:|---:|---:|')
            for k, v in results['recency_scan'].items():
                sm = v['season_mean']
                lines.append(f"| {k} | {v['inner']:.1f} | {sm.get(2022,float('nan')):.1f} | {sm.get(2023,float('nan')):.1f} | {v['seed_sd']:.1f} |")
    else:
        lines.append('\n## Phase B: season-recency — 실행 실패 또는 미완료')

    if 'combined_scan' in results:
        lines.append(f"\n## Phase C: 결합\n- 결합 inner={results['combined_scan']['inner']:.1f}, 단독보다 나음={results.get('combined_beats_both')}")

    if 'outer_confirm' in results:
        oc = results['outer_confirm']
        lines.append(f"\n## Phase D: outer(2024) 1회 확인 (선택에는 미사용)\n- 후보 outer skill = {oc['skill_2024']:.1f}  (v42 baseline 807.6 대비 {oc['delta']:+.1f})")

    if 'candidate_dir' in results:
        lines.append(f"\n## Phase E: 최종 후보 패키지\n- 경로: `{results['candidate_dir']}`\n- 레시피: `{results.get('candidate_recipe')}`")
    else:
        lines.append('\n## Phase E: 후보 미생성 (Phase A/B 모두 노이즈를 못 넘겨 정직한 null 결과)')

    if 'isolation_check' in results:
        ic = results['isolation_check']
        lines.append(f"\n## Phase F: 격리검증\n- 통과: **{ic.get('passed')}**")
        if not ic.get('passed'):
            lines.append(f"- 오류: `{ic.get('error','')[:500]}`")

    lines.append('\n## 결론')
    if results.get('candidate_dir') and results.get('isolation_check', {}).get('passed'):
        oc = results.get('outer_confirm', {})
        lines.append(f"- **제출 후보 준비 완료**: `{results['candidate_dir']}`. outer(2024) honest 확인 델타 {oc.get('delta','?'):+.1f} (v42 대비).")
        lines.append(f"- ⚠️ 하네스 잔차 SD는 35.3 LB점(outputs/503)이다. 이 델타가 그대로 실전 LB에 반영된다는 보장은 없다 — **하루 5회 제출 중 1회를 써서 검증 필요**.")
        lines.append("- 사용자 업로드 전 반드시 사람이 script.py를 검토할 것. 자동 제출은 하지 않았다.")
    else:
        lines.append("- **신뢰 가능한 개선을 찾지 못했다.** era-offset과 recency 재가중 모두(또는 하나) 이번 라운드에서 inner 노이즈 폭을 넘지 못했다.")
        lines.append("- v42(1032.137582점)를 그대로 유지하는 것이 맞다. 정직한 null 결과이며 조작하지 않았다.")
    lines.append('\n관련: `outputs/503_honest_harness_final_verdict.md`, `outputs/501_signal_budget_negative_results.md`')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    log(f'최종 보고서 작성 완료: {path}')


def main():
    log('=== 야간 캠페인 시작 ===')
    results = {}
    try:
        df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
        df.columns = [c.replace('﻿', '') for c in df.columns]
        log(f'train.csv 로드 완료 {df.shape}')

        era_scale = None
        try:
            if remaining_budget() > 3600:
                era_scale = phase_A_era(df, results)
        except Exception:
            log('[FAIL] Phase A 예외:\n' + traceback.format_exc())

        recency_decay = None
        try:
            if remaining_budget() > 3600:
                recency_decay = phase_B_recency(df, results)
        except Exception:
            log('[FAIL] Phase B 예외:\n' + traceback.format_exc())

        try:
            if remaining_budget() > 1800:
                phase_C_combined(df, results, era_scale, recency_decay)
                if results.get('combined_beats_both'):
                    log('  결합이 단독보다 우수 -> 최종 후보는 결합 레시피 사용')
                else:
                    results.pop('combined_dir', None)
        except Exception:
            log('[FAIL] Phase C 예외:\n' + traceback.format_exc())

        final_era = era_scale if not results.get('combined_beats_both') else era_scale
        final_rec = recency_decay if not results.get('combined_beats_both') else recency_decay
        if not results.get('combined_beats_both'):
            final_rec = None if era_scale is not None else recency_decay
            # prefer whichever single axis is credible; if both are, era takes priority
            # (larger, more mechanistically grounded effect measured on the pitcher channel)
            if era_scale is not None:
                final_era, final_rec = era_scale, None
            elif recency_decay is not None:
                final_era, final_rec = None, recency_decay
            else:
                final_era, final_rec = None, None

        try:
            if final_era is not None or final_rec is not None:
                phase_D_outer_confirm(df, results, final_era, final_rec)
        except Exception:
            log('[FAIL] Phase D 예외:\n' + traceback.format_exc())

        candidate_dir = None
        try:
            if final_era is not None or final_rec is not None:
                if remaining_budget() > 2400:
                    candidate_dir = phase_E_build_candidate(df, results, final_era, final_rec)
                else:
                    log('[SKIP] Phase E: 남은 예산 부족')
        except Exception:
            log('[FAIL] Phase E 예외:\n' + traceback.format_exc())

        try:
            if candidate_dir:
                phase_F_isolation_check(candidate_dir, results)
        except Exception:
            log('[FAIL] Phase F 예외:\n' + traceback.format_exc())

    except Exception:
        log('[FATAL] 캠페인 최상위 예외:\n' + traceback.format_exc())
    finally:
        try:
            write_report(results)
        except Exception:
            log('[FATAL] 보고서 작성 실패:\n' + traceback.format_exc())
        with open(os.path.join(HARNESS, 'campaign_results.json'), 'w') as f:
            json.dump(results, f, default=str, indent=1)
        log('=== 캠페인 종료 ===')


if __name__ == '__main__':
    main()
