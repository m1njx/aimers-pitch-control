"""
agent8_outer_confirm_deepmlp.py

Single, honest outer(2024) confirmation of DeepMLP, the winner of agent8's
inner-only (2022,2023) DL architecture diversification screen
(scratch/agent8_dl_diversify_screen.py, outputs/agent8_findings.md):
  DeepMLP: shared best_w=0.26 -> inner avg gain=+14.53 (best of DeepMLP/PLEMLP/LightFTT)
  PLEMLP:  gain=+0.98 (REJECTED, noise floor)
  LightFTT: gain=+3.63 (REJECTED, noise floor, and 25x slower per fold)

Frozen procedure (NO re-tuning after seeing outer):
  - w_deepmlp=0.26 taken as-is from the inner-only shared-weight selection.
    Applied ONCE to fold val=2024. No grid search touches 2024 in this script.
  - GBDT reference: OFFICIAL asof_dec 5-seed construction via the FIXED
    core/eval_utils.py::run_standard_sota_evaluation (row-independence bug,
    report 203, already patched). weights=(0.15,0.75,0.10),
    shifts=(-0.007,-0.008,-0.006), matching report 191's "outer=805.74" label.
  - DeepMLP: EXACT same architecture/recipe as the inner screening
    (256-128-64, BatchNorm, dropout 0.2, single seed=7, 10 epochs, patience=2,
    CPU-forced -- MPS stall history report 161). Categorical encoding via
    dl_common.to_tensors (train-only vocab + fixed per-row .map(), structurally
    row-independent, same pattern already verified for SimpleMLP).
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import run_standard_sota_evaluation, calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')  # force CPU -- MPS stall history (report 161)
SEED = 7
WEIGHTS = (0.15, 0.75, 0.10)
FIXED_SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
W_DEEPMLP_FROZEN = 0.26  # frozen from agent8's inner-only (2022,2023) shared-weight selection
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class DeepMLP(nn.Module):
    """Same architecture as agent8_dl_diversify_screen.py's winner."""
    def __init__(self, num_dim, cat_cardinalities, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.cat_embedder = dlc.CatEmbedder(cat_cardinalities, emb_dim=8, max_emb_dim=24)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


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

    log("=== Step B: DeepMLP, single seed=7, 10 epochs -- SAME recipe as agent8 inner screening, "
        "now also run on fold val=2024 (outer) ===")
    dl_by_season = {}
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

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = DeepMLP(num_dim, cat_cardinalities)
        model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=10, lr=1e-3,
                                          batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                          verbose_prefix=f"[DeepMLP val={vs}] ")
        p_dl = dlc.predict(model, num_val, cat_val, DEVICE, shift)
        sk_dl, br_dl, _, _ = calc_brier_skill_score(y_val_f, p_dl)
        log(f"  DeepMLP-alone val={vs}: skill={sk_dl:.2f} ({time.time()-t0:.1f}s)")
        dl_by_season[vs] = dict(p=p_dl, y=y_val_f, sk=sk_dl)

    log("=== Step C: apply FROZEN w_deepmlp=0.26 (no re-tuning) to each fold, incl. outer(2024) ===")
    blend_by_season = {}
    for vs in (2022, 2023, 2024):
        g = gbdt_by_season[vs]
        m = dl_by_season[vs]
        assert np.array_equal(g['y'], m['y']), f"y mismatch for val={vs}"
        p_blend = np.clip((1 - W_DEEPMLP_FROZEN) * g['p'] + W_DEEPMLP_FROZEN * m['p'], 1e-6, 1 - 1e-6)
        sk_blend, br_blend, _, _ = calc_brier_skill_score(g['y'], p_blend)
        gain = sk_blend - g['sk']
        blend_by_season[vs] = dict(sk_blend=sk_blend, gain=gain)
        log(f"  val={vs}: GBDT-alone={g['sk']:.2f} | DeepMLP-alone={m['sk']:.2f} | "
            f"blend(w=0.26)={sk_blend:.2f} | gain={gain:+.2f}")

    inner_avg_gbdt = np.mean([gbdt_by_season[vs]['sk'] for vs in (2022, 2023)])
    inner_avg_blend = np.mean([blend_by_season[vs]['sk_blend'] for vs in (2022, 2023)])
    log(f"\n=== SUMMARY ===")
    log(f"Inner (2022,2023) avg: GBDT-alone={inner_avg_gbdt:.2f} -> blend(w=0.26)={inner_avg_blend:.2f} "
        f"(gain={inner_avg_blend-inner_avg_gbdt:+.2f}) [consistency check vs screening's +14.53]")
    log(f"OUTER (2024) SINGLE CONFIRMATION: GBDT-alone={gbdt_by_season[2024]['sk']:.2f} -> "
        f"blend(w=0.26)={blend_by_season[2024]['sk_blend']:.2f} "
        f"(gain={blend_by_season[2024]['gain']:+.2f})")
    log(f"[reference label] report 191 pure asof_dec outer(2024) @ fixed shifts = 805.74 "
        f"(this run's GBDT-alone outer = {gbdt_by_season[2024]['sk']:.2f}, should be close)")
    log(f"[comparison] SimpleMLP (agent7) outer confirm gain was +14.29 at w_mlp=0.32 (single-seed)")

    np.savez('/tmp/agent8_outer_confirm_deepmlp_result.npz',
              gbdt_outer_sk=gbdt_by_season[2024]['sk'], dl_outer_sk=dl_by_season[2024]['sk'],
              blend_outer_sk=blend_by_season[2024]['sk_blend'], outer_gain=blend_by_season[2024]['gain'],
              inner_avg_gbdt=inner_avg_gbdt, inner_avg_blend=inner_avg_blend,
              gbdt_2022=gbdt_by_season[2022]['sk'], gbdt_2023=gbdt_by_season[2023]['sk'])
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
