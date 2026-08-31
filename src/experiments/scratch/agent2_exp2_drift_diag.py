"""
agent2_exp2_drift_diag.py — DIAGNOSTIC (cheap, LightGBM-only 1 seed).

Hypothesis: control_success has a strong monotone downward season trend
(0.5647, 0.5327, 0.5328, 0.5289, 0.5000, 0.4861). A model trained on the past
therefore predicts a mean that is systematically too HIGH for the validation
season. Brier is extremely sensitive to that mean bias:
    penalty = bias^2  -> skill loss = 1e5 * bias^2 / (r(1-r))
A bias of 0.04 costs ~640 skill points. The pipeline currently applies a tiny
fixed shift of -0.007.

This script measures, per fold:
  - mean(y_val), mean(p_val)  -> actual bias
  - skill at shift=0, at the tuned shift, and at the ORACLE shift
  - what a train-only linear extrapolation of the season mean would predict
"""
import sys, time
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score
from agent2_common import build_base_features, base_cat_cols, log

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_PATH)
    season_mean = df.groupby('season')[config.TARGET_COL].mean()
    print("season target means:\n", season_mean)

    folds = get_cv_folds(df)
    rows = []
    for fold in folds:
        df_tr = df.iloc[fold.train_idx].copy()
        df_val = df.iloc[fold.val_idx].copy()
        X_tr, X_val = build_base_features(df_tr, df_val, fold.fold_max_season)
        cc = base_cat_cols(X_tr)
        cat_idx = [X_tr.columns.get_loc(c) for c in cc]
        y_tr = df_tr[config.TARGET_COL].values
        y_val = df_val[config.TARGET_COL].values
        m = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                               min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                               random_state=7, verbosity=-1, n_jobs=-1)
        m.fit(X_tr, y_tr, categorical_feature=cat_idx)
        p = m.predict_proba(X_val)[:, 1]
        p_tr = m.predict_proba(X_tr)[:, 1]

        bias = p.mean() - y_val.mean()
        # train-only linear extrapolation of season mean
        tr_seasons = sorted(df_tr['season'].unique())
        mus = season_mean.loc[tr_seasons].values
        coef = np.polyfit(tr_seasons, mus, 1)
        mu_hat = np.polyval(coef, fold.val_season)

        res = dict(val_season=int(fold.val_season),
                   y_mean=float(y_val.mean()), p_mean=float(p.mean()),
                   p_train_mean=float(p_tr.mean()), y_train_mean=float(y_tr.mean()),
                   bias=float(bias), mu_hat_linear=float(mu_hat))
        for name, s in [('shift0', 0.0), ('shift_tuned', -0.007), ('oracle', -bias),
                        ('extrap', mu_hat - p_tr.mean())]:
            sk, raw, base, r = calc_brier_skill_score(y_val, np.clip(p + s, 1e-6, 1 - 1e-6))
            res[f'skill_{name}'] = sk
            res[f'shiftval_{name}'] = s
        # full shift sweep
        best_s, best_sk = None, -1
        for s in np.arange(-0.08, 0.021, 0.002):
            sk, *_ = calc_brier_skill_score(y_val, np.clip(p + s, 1e-6, 1 - 1e-6))
            if sk > best_sk:
                best_sk, best_s = sk, s
        res['best_shift'] = float(best_s); res['skill_best'] = float(best_sk)
        rows.append(res)
        log(str(res))

    R = pd.DataFrame(rows)
    print("\n" + R.to_string())
    print("\nMEANS: shift0=%.1f tuned=%.1f extrap=%.1f oracle=%.1f best=%.1f" % (
        R.skill_shift0.mean(), R.skill_shift_tuned.mean(), R.skill_extrap.mean(),
        R.skill_oracle.mean(), R.skill_best.mean()))
    R.to_csv('~/LG_data/scratch/agent2_exp2_drift_diag.csv', index=False)
