"""
run_ensemble_weight_only.py — 4-model ensemble weight search only.
Uses best HP from first run. Only trains 4 models once per fold, then searches weights.
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor


def calc_raw_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))

def calc_fold_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = calc_raw_brier(y_true, y_prob)
    baseline_brier = float(r * (1.0 - r))
    if baseline_brier == 0:
        return 0.0, brier, baseline_brier, r
    score = max(0.0, 100000.0 * (1.0 - (brier / baseline_brier)))
    return score, brier, baseline_brier, r

print("Loading data...")
df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train, strategy="time")

# Store OOF predictions per model per fold
lgb_preds, cb_preds, xgb_preds, hgb_preds = [], [], [], []
y_vals = []

for fi, fold in enumerate(folds):
    print(f"\n--- Fold {fi} (val season={fold.val_season}) ---")
    df_tr = df_train.iloc[fold.train_idx].reset_index(drop=True)
    df_va = df_train.iloc[fold.val_idx].reset_index(drop=True)
    y_tr = df_tr[config.TARGET_COL].values
    y_va = df_va[config.TARGET_COL].values
    y_vals.append(y_va)

    prep = PitchPreprocessor()
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)

    # Add count_x_base
    for df_src, X_dst, label in [(df_tr, X_tr, "train"), (df_va, X_va, "val")]:
        base = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
        cc = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_src['strikes_before'].fillna(0).astype(int).astype(str))
        X_dst['count_x_base'] = (cc + '_' + base)

    cat_map = {v: i for i, v in enumerate(X_tr['count_x_base'].unique())}
    X_tr['count_x_base'] = X_tr['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_va['count_x_base'] = X_va['count_x_base'].map(cat_map).fillna(-1).astype(int)

    cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_va.columns.get_loc(c) for c in cat_cols if c in X_va.columns]

    # 1. LightGBM (best: leaves=31)
    m_lgb = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, min_child_samples=20,
        learning_rate=0.05, colsample_bytree=0.8, subsample=0.8,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
    p_lgb = np.clip(m_lgb.predict_proba(X_va)[:, 1] - 0.007, 1e-6, 1-1e-6)
    lgb_preds.append(p_lgb)
    print(f"  LGBM: Brier={calc_raw_brier(y_va, p_lgb):.6f}")

    # 2. CatBoost (best: depth=6, l2=5)
    X_tr_cb, X_va_cb = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_va_cb[c] = X_va_cb[c].astype(int).astype(str)
    for c in [col for col in X_va_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_va_cb[c] = X_va_cb[c].astype(np.float32)

    m_cb = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=5.0,
        random_seed=42, verbose=0, cat_features=cat_cols
    )
    m_cb.fit(X_tr_cb, y_tr)
    p_cb = np.clip(m_cb.predict_proba(X_va_cb)[:, 1] - 0.008, 1e-6, 1-1e-6)
    cb_preds.append(p_cb)
    print(f"  CatBoost: Brier={calc_raw_brier(y_va, p_cb):.6f}")

    # 3. XGBoost (best: max_depth=5, colsample=0.7)
    X_tr_xgb, X_va_xgb = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
        X_va_xgb[c] = X_va_xgb[c].astype('category').cat.codes.astype(np.float32)
    X_tr_xgb = X_tr_xgb.astype(np.float32)
    X_va_xgb = X_va_xgb.astype(np.float32)

    m_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        colsample_bytree=0.7, subsample=0.8,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    m_xgb.fit(X_tr_xgb, y_tr)
    p_xgb = np.clip(m_xgb.predict_proba(X_va_xgb)[:, 1] - 0.006, 1e-6, 1-1e-6)
    xgb_preds.append(p_xgb)
    print(f"  XGBoost: Brier={calc_raw_brier(y_va, p_xgb):.6f}")

    # 4. HistGradientBoosting
    X_tr_hgb, X_va_hgb = X_tr.copy(), X_va.copy()
    for c in cat_cols:
        X_tr_hgb[c] = X_tr_hgb[c].astype(int)
        X_va_hgb[c] = X_va_hgb[c].astype(int)

    m_hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.05,
        categorical_features=cat_idx, random_state=42
    )
    m_hgb.fit(X_tr_hgb, y_tr)
    p_hgb = np.clip(m_hgb.predict_proba(X_va_hgb)[:, 1] - 0.007, 1e-6, 1-1e-6)
    hgb_preds.append(p_hgb)
    print(f"  HistGB: Brier={calc_raw_brier(y_va, p_hgb):.6f}")

# Correlations
print("\n--- Prediction Pearson Correlation Matrix (3-Fold Means) ---")
pairs = [("LGBM", "CatBoost", lgb_preds, cb_preds),
         ("LGBM", "XGBoost", lgb_preds, xgb_preds),
         ("CatBoost", "XGBoost", cb_preds, xgb_preds),
         ("HistGB", "LGBM", hgb_preds, lgb_preds),
         ("HistGB", "CatBoost", hgb_preds, cb_preds),
         ("HistGB", "XGBoost", hgb_preds, xgb_preds)]
corr_dict = {}
for n1, n2, p1, p2 in pairs:
    cs = [pearsonr(p1[i], p2[i])[0] for i in range(3)]
    m = float(np.mean(cs))
    corr_dict[f"{n1}_vs_{n2}"] = m
    print(f"  {n1} vs {n2}: {m:.4f}")

# HistGB solo performance
hgb_briers = [calc_raw_brier(y_vals[i], hgb_preds[i]) for i in range(3)]
hgb_skills = [calc_fold_skill_score(y_vals[i], hgb_preds[i])[0] for i in range(3)]
hgb_aucs = [roc_auc_score(y_vals[i], hgb_preds[i]) for i in range(3)]
print(f"\nHistGB Solo: Brier={np.mean(hgb_briers):.6f}, Skill={np.mean(hgb_skills):.2f}점, AUC={np.mean(hgb_aucs):.6f}")

# Weight search — 3-model and 4-model
print("\n--- Ensemble Weight Search ---")
results = []

# 3-model combos (no HGB)
for i_lgb in range(5, 55, 5):
    for i_cb in range(30, 90, 5):
        i_xgb = 100 - i_lgb - i_cb
        if 5 <= i_xgb <= 35:
            w = (i_lgb/100, i_cb/100, i_xgb/100, 0.0)
            bs, ss, aus = [], [], []
            for fi in range(3):
                p = np.clip(w[0]*lgb_preds[fi] + w[1]*cb_preds[fi] + w[2]*xgb_preds[fi], 1e-6, 1-1e-6)
                bs.append(calc_raw_brier(y_vals[fi], p))
                ss.append(calc_fold_skill_score(y_vals[fi], p)[0])
                aus.append(roc_auc_score(y_vals[fi], p))
            results.append({"w_lgb": w[0], "w_cb": w[1], "w_xgb": w[2], "w_hgb": 0.0,
                            "inner_brier": (bs[0]+bs[1])/2, "mean_brier": np.mean(bs),
                            "mean_skill": np.mean(ss), "mean_auc": np.mean(aus)})

# 4-model combos
for i_lgb in range(5, 45, 5):
    for i_cb in range(30, 80, 5):
        for i_xgb in range(5, 30, 5):
            i_hgb = 100 - i_lgb - i_cb - i_xgb
            if 5 <= i_hgb <= 25:
                w = (i_lgb/100, i_cb/100, i_xgb/100, i_hgb/100)
                bs, ss, aus = [], [], []
                for fi in range(3):
                    p = np.clip(w[0]*lgb_preds[fi] + w[1]*cb_preds[fi] + w[2]*xgb_preds[fi] + w[3]*hgb_preds[fi], 1e-6, 1-1e-6)
                    bs.append(calc_raw_brier(y_vals[fi], p))
                    ss.append(calc_fold_skill_score(y_vals[fi], p)[0])
                    aus.append(roc_auc_score(y_vals[fi], p))
                results.append({"w_lgb": w[0], "w_cb": w[1], "w_xgb": w[2], "w_hgb": w[3],
                                "inner_brier": (bs[0]+bs[1])/2, "mean_brier": np.mean(bs),
                                "mean_skill": np.mean(ss), "mean_auc": np.mean(aus)})

df_res = pd.DataFrame(results).sort_values("inner_brier")
print(f"\nTotal weight combos evaluated: {len(results)}")
print("\nTop 10 Ensemble Weight Candidates (sorted by inner_brier):")
print(df_res.head(10).to_string(index=False))

best = df_res.iloc[0].to_dict()
print(f"\n*** BEST ENSEMBLE: LGBM={best['w_lgb']}, CB={best['w_cb']}, XGB={best['w_xgb']}, HGB={best['w_hgb']}")
print(f"    Mean Brier={best['mean_brier']:.6f}, Skill={best['mean_skill']:.2f}점, AUC={best['mean_auc']:.6f}")

# Save
summary = {
    "hp_tuning": {
        "lgbm_best": {"leaves": 31, "min_child": 20, "colsample": 0.8, "brier": 0.247729, "skill": 773.08},
        "catboost_best": {"depth": 6, "l2": 5.0, "brier": 0.247538, "skill": 849.77},
        "xgboost_best": {"max_depth": 5, "colsample": 0.7, "brier": 0.248156, "skill": 609.08}
    },
    "histgb_solo": {"brier": float(np.mean(hgb_briers)), "skill": float(np.mean(hgb_skills)), "auc": float(np.mean(hgb_aucs))},
    "correlations": corr_dict,
    "best_ensemble": best,
    "top10": df_res.head(10).to_dict(orient="records")
}
with open("~/LG_data/outputs/hp_and_4model_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

print("\nENSEMBLE WEIGHT SEARCH COMPLETED SUCCESSFULLY!")
