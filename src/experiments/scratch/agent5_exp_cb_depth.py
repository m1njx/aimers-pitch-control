"""
agent4_exp2_cb_depth.py

CatBoost carries 75% of the ensemble weight (vs LightGBM's 15%), so tree-capacity
overfitting there should matter more than agent3's LGBM num_leaves finding.
INNER-ONLY (val=2022,2023) sweep of CatBoost `depth` on top of asof_dec.
Also sweep l2_leaf_reg since agent3's finding (EXP13/14) was about capacity/
regularization interacting with base-rate drift, not leaves specifically.
"""
import sys, time, json
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import config
import agent4_lib as L

df_train = pd.read_csv(config.TRAIN_PATH)


def v_control(df_tr_f, df_val_f, vs, X_tr, X_val, cc, A_tr, A_val):
    return X_tr, X_val, cc, None


VARIANTS = {'control_asofdec': v_control}


def fit_predict_ensemble_cb(X_tr, y_tr, X_val, cat_cols, seed, cb_overrides):
    return L.fit_predict_ensemble(X_tr, y_tr, X_val, cat_cols, seed, cb_overrides=cb_overrides)


# Monkey-patch a variant runner that supports cb_overrides (agent4_lib.run_variants
# only threads lgb_overrides through the 4-tuple return). Build a small local loop
# reusing agent4_lib helper functions directly instead.

def run_cb_sweep(df_train, cb_configs, seeds):
    from cv_utils import get_cv_folds
    from core.eval_utils import calc_brier_skill_score
    folds = get_cv_folds(df_train)
    folds = [folds[0], folds[1]]  # val=2022, 2023 only
    results = {name: [] for name in cb_configs}
    for fold in folds:
        t0 = time.time()
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
        as_of = fold.fold_max_season
        val_season = fold.val_season
        X_tr_b, X_val_b = L.build_base_features(df_tr_f, df_val_f, as_of)
        X_tr_b, X_val_b, A_tr, A_val = L.add_asof_dec(df_tr_f, df_val_f, val_season, X_tr_b, X_val_b)
        cc_b = L.base_cat_cols(X_tr_b)
        y_tr = df_tr_f[config.TARGET_COL].values
        y_val = df_val_f[config.TARGET_COL].values
        L.log(f"fold val={val_season}: base+asof_dec ready in {time.time()-t0:.0f}s")
        for name, cb_ov in cb_configs.items():
            t1 = time.time()
            bag = np.zeros(len(X_val_b))
            for seed in seeds:
                bag += L.fit_predict_ensemble(X_tr_b, y_tr, X_val_b, cc_b, seed, cb_overrides=cb_ov)
            p = np.clip(bag / len(seeds), 1e-6, 1 - 1e-6)
            skill, raw, base, r = calc_brier_skill_score(y_val, p)
            results[name].append(dict(val_season=int(val_season), skill=skill, raw_brier=raw))
            L.log(f"  [{name}] val={val_season} skill={skill:.2f} ({time.time()-t1:.0f}s)")
            with open('~/LG_data/scratch/agent5_exp_cb_depth_screen.json', 'w') as f:
                json.dump(results, f, indent=2)
    summary = {}
    for name, fds in results.items():
        summary[name] = dict(fold_details=fds,
                              inner_mean=float(np.mean([f['skill'] for f in fds])))
    return summary


CB_CONFIGS = {
    'depth6_default': None,
    'depth4': {'depth': 4},
    'depth5': {'depth': 5},
    'depth7': {'depth': 7},
    'depth8': {'depth': 8},
    'l2reg30': {'l2_leaf_reg': 30.0},
    'l2reg3': {'l2_leaf_reg': 3.0},
    'depth4_l2reg30': {'depth': 4, 'l2_leaf_reg': 30.0},
}

L.log("=== agent4 exp2: CatBoost depth/l2reg 2-seed INNER-ONLY screen ===")
t0 = time.time()
summary = run_cb_sweep(df_train, CB_CONFIGS, L.SCREEN_SEEDS)
L.log(f"done in {(time.time()-t0)/60:.1f}min")
print("\n" + "=" * 60)
for name, s in summary.items():
    print(f"{name:<20} inner={s['inner_mean']:.2f}")
print("=" * 60)

with open('~/LG_data/scratch/agent5_exp_cb_depth_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
L.log("=== exp2 DONE ===")
