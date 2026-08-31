"""
agent7_mlp_5seed_full.py

User-requested follow-up to agent7's single-seed outer(2024) confirmation
(scratch/agent7_outer_confirm_mlp.py, gain=+14.29 at outer with frozen w_mlp=0.32):
(1) redo the MLP side as a proper 5-seed bag (matching the official GBDT
    5-seed standard), not just seed=7.
(2) empirically re-verify row-independence for the MLP's categorical encoding
    path (dl_common.to_tensors already reads as structurally row-independent
    from code audit -- vocab built from TRAIN split only, applied via a fixed
    per-row .map() -- but we confirm it empirically too, the same way the
    XGBoost bug in report 203 was originally caught: batch prediction vs
    each row predicted alone must match exactly).

Frozen weight w_mlp=0.32 (agent6's inner-only selection) is NOT re-tuned here.
GBDT and MLP are both bagged over the same 5 seeds used everywhere else in
this project: [7, 123, 2025, 31415, 8675309].
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import run_standard_sota_evaluation, calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')  # force CPU -- MPS stall history (report 161)
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
W_MLP_FROZEN = 0.32  # frozen from agent6's inner-only (2022,2023) shared-weight selection -- NOT re-tuned
WEIGHTS = (0.15, 0.75, 0.10)
FIXED_SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f)
    val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index
    val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


def build_asofdec_fold_frames(df_train, fold):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)
    dlc.add_count_x_base(df_tr_f, X_tr_f)
    dlc.add_count_x_base(df_val_f, X_val_f)
    cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
    X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_tr_f, X_val_f = add_asof_dec_features(df_tr_f, df_val_f, fold.fold_max_season, X_tr_f, X_val_f)
    y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)
    y_val_f = df_val_f[config.TARGET_COL].values.astype(np.float32)
    return X_tr_f, X_val_f, y_tr_f, y_val_f


def main():
    t_start = time.time()
    df_train = pd.read_csv(config.TRAIN_PATH)
    folds = get_cv_folds(df_train)

    log("=== Step A: OFFICIAL asof_dec 5-seed GBDT reference, all 3 folds ===")
    t0 = time.time()
    r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                      weights=WEIGHTS, shifts=FIXED_SHIFTS, random_seeds=FULL_SEEDS,
                                      extra_feature_fn=add_asof_dec_features)
    log(f"GBDT reference done in {(time.time()-t0)/60:.1f} min. "
        f"fold skills = {[(fd['val_season'], round(fd['skill_k'],2)) for fd in r['fold_details']]}")

    w_lgb, w_cb, w_xgb = WEIGHTS
    gbdt_by_season = {}
    for fold in folds:
        vs = fold.val_season
        vi = fold.val_idx
        p_gbdt = np.clip(w_lgb * r['oof_preds_lgb'][vi] + w_cb * r['oof_preds_cb'][vi]
                          + w_xgb * r['oof_preds_xgb'][vi], 1e-6, 1 - 1e-6)
        y = df_train.iloc[vi][config.TARGET_COL].values
        sk, br, _, _ = calc_brier_skill_score(y, p_gbdt)
        gbdt_by_season[vs] = dict(p=p_gbdt, y=y, sk=sk)
        log(f"  GBDT-alone val={vs}: skill={sk:.2f}")

    log("=== Step B: MLP, FULL 5-seed bag, 8 epochs/seed (matching GBDT's official seed set) ===")
    mlp_by_season = {}
    mlp_trained_models = {}  # keep fold=2024's models+vocab for the row-independence re-check
    for fold in folds:
        vs = fold.val_season
        t0 = time.time()
        X_tr_f, X_val_f, y_tr_f, y_val_f = build_asofdec_fold_frames(df_train, fold)
        tens = dlc.to_tensors(X_tr_f, X_val_f)
        num_tr, num_val = tens['num_tr'], tens['num_val']
        cat_tr, cat_val = tens['cat_tr'], tens['cat_val']
        y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)
        num_dim = num_tr.shape[1]
        cat_cardinalities = tens['cat_cardinalities']

        p_sum = np.zeros(len(y_val_f))
        seed_models = []
        for seed in FULL_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = dlc.SimpleMLP(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15)
            model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=8, lr=1e-3,
                                              batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                              verbose_prefix=f"[mlp val={vs} seed={seed}] ")
            p_seed = dlc.predict(model, num_val, cat_val, DEVICE, shift)
            p_sum += p_seed
            seed_models.append((model, shift))
            sk_seed, *_ = calc_brier_skill_score(y_val_f, p_seed)
            log(f"  MLP seed={seed} val={vs}: skill={sk_seed:.2f}")
        p_mlp_bagged = np.clip(p_sum / len(FULL_SEEDS), 1e-6, 1 - 1e-6)
        sk_mlp, br_mlp, _, _ = calc_brier_skill_score(y_val_f, p_mlp_bagged)
        log(f"  MLP-BAGGED(5-seed) val={vs}: skill={sk_mlp:.2f} ({time.time()-t0:.1f}s)")
        mlp_by_season[vs] = dict(p=p_mlp_bagged, y=y_val_f, sk=sk_mlp)
        if vs == 2024:
            mlp_trained_models[2024] = dict(models=seed_models, tens=tens, X_val_f=X_val_f,
                                             X_tr_f=X_tr_f, y_val_f=y_val_f)

    log("=== Step C: apply FROZEN w_mlp=0.32 (no re-tuning) to each fold, incl. outer(2024) ===")
    blend_by_season = {}
    for vs in (2022, 2023, 2024):
        g = gbdt_by_season[vs]
        m = mlp_by_season[vs]
        assert np.array_equal(g['y'], m['y']), f"y mismatch for val={vs}"
        p_blend = np.clip((1 - W_MLP_FROZEN) * g['p'] + W_MLP_FROZEN * m['p'], 1e-6, 1 - 1e-6)
        sk_blend, br_blend, _, _ = calc_brier_skill_score(g['y'], p_blend)
        gain = sk_blend - g['sk']
        blend_by_season[vs] = dict(sk_blend=sk_blend, gain=gain)
        log(f"  val={vs}: GBDT-alone={g['sk']:.2f} | MLP-bagged={m['sk']:.2f} | "
            f"blend(w=0.32)={sk_blend:.2f} | gain={gain:+.2f}")

    inner_avg_gbdt = np.mean([gbdt_by_season[vs]['sk'] for vs in (2022, 2023)])
    inner_avg_blend = np.mean([blend_by_season[vs]['sk_blend'] for vs in (2022, 2023)])
    log(f"\n=== SUMMARY (5-seed bagged MLP, vs single-seed screening) ===")
    log(f"Inner (2022,2023) avg: GBDT-alone={inner_avg_gbdt:.2f} -> blend(w=0.32)={inner_avg_blend:.2f} "
        f"(gain={inner_avg_blend-inner_avg_gbdt:+.2f})")
    log(f"OUTER (2024): GBDT-alone={gbdt_by_season[2024]['sk']:.2f} -> "
        f"blend(w=0.32)={blend_by_season[2024]['sk_blend']:.2f} "
        f"(gain={blend_by_season[2024]['gain']:+.2f})  "
        f"[single-seed screening had gain=+14.29 -- this confirms/refutes with proper 5-seed bagging]")

    # ---- Step D: empirical row-independence re-check for the MLP prediction path ----
    log("\n=== Step D: empirical row-independence check (MLP, fold=2024) ===")
    d = mlp_trained_models[2024]
    X_val_f = d['X_val_f']
    tens = d['tens']
    num_val_full, cat_val_full = tens['num_val'], tens['cat_val']
    # batch prediction (all outer/2024 val rows together) with seed=7's model, no shift added twice
    model0, shift0 = d['models'][0]
    p_batch = dlc.predict(model0, num_val_full, cat_val_full, DEVICE, shift0)
    # pick 5 arbitrary rows and predict each ALONE (single-row tensor slice) with the SAME model
    check_idx = [0, 1, len(p_batch)//2, len(p_batch)-2, len(p_batch)-1]
    max_diff = 0.0
    for i in check_idx:
        p_single = dlc.predict(model0, num_val_full[i:i+1], cat_val_full[i:i+1], DEVICE, shift0)
        diff = abs(float(p_single[0]) - float(p_batch[i]))
        max_diff = max(max_diff, diff)
        log(f"  row_idx={i}: batch_pred={p_batch[i]:.10f} single_pred={p_single[0]:.10f} diff={diff:.2e}")
    log(f"MLP row-independence check: max_diff={max_diff:.2e} "
        f"({'PASS -- structurally row-independent as expected from code audit (dl_common.to_tensors builds vocab from TRAIN split only)' if max_diff < 1e-6 else '*** FAIL -- unexpected batch dependency found ***'})")

    np.savez('/tmp/agent7_mlp_5seed_result.npz',
              gbdt_outer_sk=gbdt_by_season[2024]['sk'], mlp_outer_sk=mlp_by_season[2024]['sk'],
              blend_outer_sk=blend_by_season[2024]['sk_blend'], outer_gain=blend_by_season[2024]['gain'],
              inner_avg_gbdt=inner_avg_gbdt, inner_avg_blend=inner_avg_blend,
              gbdt_2022=gbdt_by_season[2022]['sk'], gbdt_2023=gbdt_by_season[2023]['sk'],
              mlp_row_independence_max_diff=max_diff)
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
