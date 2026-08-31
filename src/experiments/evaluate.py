"""
evaluate.py — score any blend/calibration config against the honest cache.

Reads the prediction cache written by build_cache.py and scores an arbitrary
config in milliseconds, with no retraining. That is the whole point: config
search becomes free once the components are cached.

A config is the same knob set the production script.py exposes:
    sub-weights over the three binary GBDTs   (W_LGB_BIN / W_CB_BIN / W_XGB_BIN)
    per-model shifts                          (S_LGB / S_CB / S_XGB)
    top-level blend                           (W_GBDT_BIN / W_MLP_MSE / W_LGB_MSE)
    affine calibration                        (SCALE / SHIFT)

Reported per config:
    per (season, seed) skill, the seed-mean per season, the across-season mean,
    and the seed spread -- the last is the harness's own noise floor and no
    difference smaller than it may be called an improvement.

    venv311/bin/python3 harness/evaluate.py                 # validate + grid
    venv311/bin/python3 harness/evaluate.py --validate-only
"""
import os, glob, argparse, itertools, json
import numpy as np

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')

PROD = dict(w_lgb=0.20, w_cb=0.72, w_xgb=0.08,
            s_lgb=-0.007, s_cb=-0.008, s_xgb=-0.006,
            w_gbdt=0.40, w_mlp=0.40, w_mse=0.20,
            scale=1.10, shift=-0.0045192086)

# blend weights whose real Public-LB score we know -> the harness's report card.
# (all share the same base artifacts, see outputs/500 s.2)
# Each entry is the version's ACTUAL submitted config, calibration included --
# v54 shipped SCALE 1.29 / SHIFT +0.006, not the v42 defaults, and holding the
# calibration fixed while varying only the weights would validate a config that
# was never submitted.
KNOWN_LB = [
    ('v42 .40/.40/.20', dict(w_gbdt=0.40, w_mlp=0.40, w_mse=0.20,
                             scale=1.10, shift=-0.0045192086), 1032.1),
    ('v50 .25/.50/.25', dict(w_gbdt=0.25, w_mlp=0.50, w_mse=0.25,
                             scale=1.10, shift=-0.003500), 1032.0),
    ('v40 .45/.35/.20', dict(w_gbdt=0.45, w_mlp=0.35, w_mse=0.20,
                             scale=1.10, shift=-0.0045192086), 1030.4),
    ('v54 .10/.30/.60', dict(w_gbdt=0.10, w_mlp=0.30, w_mse=0.60,
                             scale=1.29, shift=+0.006), 968.0),
    ('v56 .14/.51/.35', dict(w_gbdt=0.14, w_mlp=0.51, w_mse=0.35,
                             scale=1.10, shift=-0.003500), 915.0),
]
# NOTE: every one of these shipped the same per-count `count_shifts_artifact.pkl`
# (byte-identical across the set-A versions, see outputs/500 s.2). The harness does
# not model that term, so absolute levels differ from the submitted pipeline; it is
# a shared additive constant, so config-to-config comparisons stay meaningful.


def load_cache():
    folds = {}
    for f in sorted(glob.glob(os.path.join(CACHE, 'pred_*.npz'))):
        _, yr, sd = os.path.basename(f)[:-4].split('_')
        yr, sd = int(yr), int(sd)
        y = np.load(os.path.join(CACHE, f'y_{yr}.npy'))
        folds.setdefault(yr, {'y': y, 'seeds': {}})['seeds'][sd] = dict(np.load(f))
    return folds


def predict(c, P):
    e = 1e-6
    lgb_ = np.clip(P['lgb_bin'] + c['s_lgb'], e, 1 - e)
    cb_ = np.clip(P['cb_bin'] + c['s_cb'], e, 1 - e)
    xgb_ = np.clip(P['xgb_bin'] + c['s_xgb'], e, 1 - e)
    gbdt = np.clip(c['w_lgb'] * lgb_ + c['w_cb'] * cb_ + c['w_xgb'] * xgb_, e, 1 - e)
    mse = np.clip(P['lgb_mse'], e, 1 - e)
    raw = c['w_gbdt'] * gbdt + c['w_mlp'] * P['mlp'] + c['w_mse'] * mse
    return np.clip(0.5 + c['scale'] * (raw - 0.5) + c['shift'], e, 1 - e)


def skill(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1 - r)))


def score(cfg, folds):
    c = dict(PROD); c.update(cfg)
    per = {}
    for yr, F in sorted(folds.items()):
        per[yr] = [skill(predict(c, P), F['y']) for _, P in sorted(F['seeds'].items())]
    means = {yr: float(np.mean(v)) for yr, v in per.items()}
    spread = float(np.mean([np.std(v, ddof=1) for v in per.values() if len(v) > 1])) if any(len(v) > 1 for v in per.values()) else 0.0
    return dict(per_season=per, season_mean=means,
                overall=float(np.mean(list(means.values()))), seed_sd=spread)


def rank_agg(rows_scores, years):
    """Average the per-season RANK instead of the per-season skill.

    Raw skill is not comparable across seasons -- 2022 sits near 2130 while 2023
    sits near 685 -- so a plain mean lets 2022 dominate. Rank-averaging removes the
    scale and measured best against the real leaderboard: Pearson +0.744 and
    residual SD 35.3 LB pts, versus +0.400 / 48.3 for the raw mean.
    """
    M = np.array([[s['season_mean'][y] for y in years] for s in rows_scores])
    return np.array([[sorted(M[:, j]).index(x) + 1 for x in M[:, j]]
                     for j in range(M.shape[1])]).mean(0)


def validate(folds):
    print('\n=== 검증: 하네스가 실전 LB 순서를 재현하는가 (n=%d 설정) ===' % len(KNOWN_LB))
    rows = []
    for name, cfg, lb in KNOWN_LB:
        r = score(cfg, folds)
        rows.append((name, lb, r['overall'], r['season_mean'], r['seed_sd']))
    print('%-18s %9s %12s   %s' % ('config', '실전LB', '하네스', '연도별'))
    for n, lb, ov, sm, sd in rows:
        per = ' '.join('%d:%.0f' % (y, v) for y, v in sorted(sm.items()))
        print('%-18s %9.1f %12.1f   %s' % (n, lb, ov, per))
    lb = np.array([r[1] for r in rows]); hv = np.array([r[2] for r in rows])
    if len(rows) > 2:
        from scipy.stats import spearmanr, pearsonr
        sp, pe = spearmanr(lb, hv), pearsonr(lb, hv)
        print('\n  Spearman rho = %+.3f (p=%.3f)   Pearson r = %+.3f' % (sp.statistic, sp.pvalue, pe.statistic))
        print('  하네스 시드 노이즈 (seed_sd 평균) = %.1f점' % np.mean([r[4] for r in rows]))
        verdict = ('사용 가능 — 양의 상관' if sp.statistic > 0.5 else
                   '사용 불가 — 상관 없음/역상관')
        print('  판정: %s' % verdict)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    folds = load_cache()
    if not folds:
        print('cache 없음 — 먼저 build_cache.py 실행'); return
    print('cache: ' + ', '.join('%d(seeds=%s)' % (y, sorted(F['seeds'])) for y, F in sorted(folds.items())))
    base = score({}, folds)
    print('\n프로덕션(v42) 설정 = %.1f  | 연도별 %s | seed_sd %.1f'
          % (base['overall'], base['season_mean'], base['seed_sd']))
    validate(folds)
    if a.validate_only:
        return
    print('\n=== 블렌드 가중치 그리드 (하네스 기준) ===')
    res = []
    for wg in np.arange(0.10, 0.61, 0.05):
        for wm in np.arange(0.10, 0.71, 0.05):
            wms = 1 - wg - wm
            if wms < 0.05 or wms > 0.6:
                continue
            r = score(dict(w_gbdt=wg, w_mlp=wm, w_mse=wms), folds)
            res.append((r['overall'], wg, wm, wms))
    res.sort(reverse=True)
    print('%8s  %5s %5s %5s' % ('skill', 'GBDT', 'MLP', 'MSE'))
    for s, a_, b_, c_ in res[:8]:
        print('%8.1f  %5.2f %5.2f %5.2f' % (s, a_, b_, c_))
    print('  ... 최하위: %8.1f  %5.2f %5.2f %5.2f' % res[-1])
    print('\n  주의: 위 차이가 seed_sd(%.1f)보다 작으면 개선으로 볼 수 없다.' % base['seed_sd'])


if __name__ == '__main__':
    main()
