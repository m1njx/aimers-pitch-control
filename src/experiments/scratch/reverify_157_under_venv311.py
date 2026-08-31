"""
reverify_157_under_venv311.py
157번의 nested-honest 888.43점(GBDT 52% + TabM 48%)이, 실제 제출에 쓰인 venv311
(Python 3.11.15, numpy 1.26.4, xgboost 3.2.0, torch 2.7.1 — 메인 환경과 라이브러리
버전이 다름)에서도 재현되는지 직접 확인. 8차 제출(818.07점, -70.36 오차) 원인 규명의
핵심 실험.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd

import config
from cv_utils import get_cv_folds
from core.eval_utils import run_standard_sota_evaluation, calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills
from dl_common import run_dl_track, DEBIASED_SEEDS_FULL, DEVICE
from track3_model import tabm_factory


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"=== RE-VERIFY 157 UNDER venv311 (device={DEVICE}) ===")
import torch, lightgbm, catboost, xgboost
log(f"Versions: python={sys.version.split()[0]} numpy={np.__version__} torch={torch.__version__} "
    f"lightgbm={lightgbm.__version__} catboost={catboost.__version__} xgboost={xgboost.__version__}")

df_train = pd.read_csv(config.TRAIN_PATH)
sota_mp = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
sota_weights = (0.15, 0.75, 0.10)
sota_shifts = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SEEDS = DEBIASED_SEEDS_FULL

t0 = time.time()
log("--- Step 1: GBDT 5-seed CV under venv311 ---")
r_gbdt = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                       weights=sota_weights, shifts=sota_shifts, random_seeds=SEEDS)
log(f"GBDT (venv311): Skill={r_gbdt['mean_fold_skill']:.2f}점 Brier={r_gbdt['overall_raw_brier']:.6f} "
    f"(main-env reference: 843.69, delta={r_gbdt['mean_fold_skill']-843.69:+.2f}) ({(time.time()-t0)/60:.1f}min)")

val_idx = np.array(r_gbdt['val_indices'])
p_lgb = np.clip(r_gbdt['oof_preds_lgb'][val_idx] + sota_shifts['lgb'], 1e-6, 1 - 1e-6)
p_cb = np.clip(r_gbdt['oof_preds_cb'][val_idx] + sota_shifts['cb'], 1e-6, 1 - 1e-6)
p_xgb = np.clip(r_gbdt['oof_preds_xgb'][val_idx] + sota_shifts['xgb'], 1e-6, 1 - 1e-6)
p_gbdt_ens = np.clip(0.15 * p_lgb + 0.75 * p_cb + 0.10 * p_xgb, 1e-6, 1 - 1e-6)

n = len(df_train)
p_gbdt_full = np.full(n, np.nan)
p_gbdt_full[val_idx] = p_gbdt_ens

t1 = time.time()
log("\n--- Step 2: TabM 5-seed CV under venv311 ---")
result_tabm = run_dl_track(tabm_factory, "T3-TabM-venv311", epochs=10, lr=1e-3, batch_size=4096,
                            seeds=SEEDS, log_fn=log)
log(f"TabM (venv311): Skill={result_tabm['mean_skill']:.2f}점 Brier={result_tabm['overall_brier']:.6f} "
    f"(main-env reference: 790.00, delta={result_tabm['mean_skill']-790.00:+.2f}) ({(time.time()-t1)/60:.1f}min)")
n_collapsed = sum(1 for r in result_tabm['per_seed_fold_skills'] if r['skill_k'] <= 0)
log(f"TabM per-seed-fold collapse count: {n_collapsed}/15 (main-env reference: 3/15)")
for r in result_tabm['per_seed_fold_skills']:
    flag = " COLLAPSE" if r['skill_k'] <= 0 else ""
    log(f"  fold{r['fold']}({r['val_season']}) seed={r['seed']}: skill={r['skill_k']:.2f}{flag}")

p_tabm_full = result_tabm['oof']

# --- Step 3: Nested-honest blend selection (matching 157's exact methodology) ---
log("\n--- Step 3: Nested-honest blend selection (inner 2022/23 -> outer 2024) ---")
folds = get_cv_folds(df_train)
inner_folds = [f for f in folds if f.val_season in (2022, 2023)]
outer_fold = [f for f in folds if f.val_season == 2024][0]
y_full = df_train[config.TARGET_COL].values


def fold_skill_for_blend(w_tabm, fold_list):
    p_blend = np.clip((1 - w_tabm) * p_gbdt_full + w_tabm * p_tabm_full, 1e-6, 1 - 1e-6)
    skills = []
    for fold in fold_list:
        vi = fold.val_idx
        sk, _, _, _ = calc_brier_skill_score(y_full[vi], p_blend[vi])
        skills.append(sk)
    return float(np.mean(skills))


best_w, best_inner = 0.0, -1
for w in np.linspace(0, 0.6, 31):
    sk = fold_skill_for_blend(w, inner_folds)
    if sk > best_inner:
        best_inner, best_w = sk, float(w)

honest_full = fold_skill_for_blend(best_w, folds)
outer_only = fold_skill_for_blend(best_w, [outer_fold])
gbdt_alone_full = fold_skill_for_blend(0.0, folds)

log(f"\nBest w_tabm (inner-selected): {best_w:.2f}")
log(f"Nested-honest full skill (venv311): {honest_full:.2f}점 (157번 main-env reference: 888.43, "
    f"delta={honest_full-888.43:+.2f})")
log(f"Outer(2024)-only skill: {outer_only:.2f}")
log(f"GBDT-alone full skill (venv311, w=0): {gbdt_alone_full:.2f}")

t_elapsed = time.time() - t0
log(f"\n=== TOTAL TIME: {t_elapsed/60:.1f} min ===")

with open('/tmp/reverify_157_venv311_result.json', 'w') as f:
    import json
    json.dump({
        "gbdt_skill": r_gbdt['mean_fold_skill'],
        "gbdt_brier": r_gbdt['overall_raw_brier'],
        "tabm_skill": result_tabm['mean_skill'],
        "tabm_brier": result_tabm['overall_brier'],
        "tabm_n_collapsed": n_collapsed,
        "best_w_tabm": best_w,
        "honest_full_skill": honest_full,
        "outer_only_skill": outer_only,
        "gbdt_alone_full": gbdt_alone_full,
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
                     "xgboost": xgboost.__version__},
    }, f, indent=2)

log("Done. Result saved to /tmp/reverify_157_venv311_result.json")
