"""agent3_lib.py — fast cached feature/fold harness for agent3 exploration.

Builds per-fold preprocessed feature matrices ONCE and caches them to disk,
so every subsequent experiment only pays model-fitting cost.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(os.path.expanduser('~/LG_data'))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor

CACHE = BASE / 'scratch' / 'agent3_cache'
CACHE.mkdir(parents=True, exist_ok=True)


def calc_raw_brier(y, p):
    return float(np.mean((p - y) ** 2))


def calc_skill(y, p):
    r = float(np.mean(y))
    b = calc_raw_brier(y, p)
    base = r * (1 - r)
    return max(0.0, 100000.0 * (1.0 - b / base))


def report(name, fold_preds, fold_y, extra=""):
    """fold_preds/fold_y: list of 3 arrays (2022, 2023, 2024)."""
    skills = [calc_skill(y, p) for y, p in zip(fold_y, fold_preds)]
    briers = [calc_raw_brier(y, p) for y, p in zip(fold_y, fold_preds)]
    inner = float(np.mean(skills[:2]))
    outer = skills[2]
    full = float(np.mean(skills))
    print(f"[{name}] nested-full={full:8.2f} | inner(22,23)={inner:8.2f} | OUTER(2024)={outer:8.2f} "
          f"| folds={[round(s,2) for s in skills]} | brier={[round(b,6) for b in briers]} {extra}")
    return dict(name=name, full=full, inner=inner, outer=outer, skills=skills, briers=briers)


def load_train():
    return pd.read_csv(config.TRAIN_PATH)


def build_cache(force=False):
    marker = CACHE / 'done.json'
    if marker.exists() and not force:
        return json.loads(marker.read_text())
    t0 = time.time()
    df = load_train()
    folds = get_cv_folds(df)
    meta = {}
    for k, fold in enumerate(folds):
        df_tr = df.iloc[fold.train_idx].copy()
        df_va = df.iloc[fold.val_idx].copy()
        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)
        # count_x_base (as in eval_utils SSOT)
        for src, dst in [(df_tr, X_tr), (df_va, X_va)]:
            b = ((src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                 (src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                 (src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
            c = (src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
                 src['strikes_before'].fillna(0).astype(int).astype(str))
            dst['count_x_base'] = c + '_' + b
        cmap = {v: i for i, v in enumerate(X_tr['count_x_base'].unique())}
        X_tr['count_x_base'] = X_tr['count_x_base'].map(cmap).fillna(-1).astype(int)
        X_va['count_x_base'] = X_va['count_x_base'].map(cmap).fillna(-1).astype(int)

        X_tr.to_parquet(CACHE / f'X_tr_{k}.parquet')
        X_va.to_parquet(CACHE / f'X_va_{k}.parquet')
        np.save(CACHE / f'y_tr_{k}.npy', df_tr[config.TARGET_COL].values.astype(np.float32))
        np.save(CACHE / f'y_va_{k}.npy', df_va[config.TARGET_COL].values.astype(np.float32))
        np.save(CACHE / f's_tr_{k}.npy', df_tr['season'].values.astype(np.int32))
        # extra raw side-info for feature engineering experiments
        side_cols = ['row_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
                     'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
                     'balls_before', 'strikes_before', 'outs_before', 'game_type',
                     'pitcher_hand', 'batter_hand', 'li', 'asof_pitcher_n', 'asof_batter_n']
        side_cols = [c for c in side_cols if c in df_tr.columns]
        df_tr[side_cols].to_parquet(CACHE / f'side_tr_{k}.parquet')
        df_va[side_cols].to_parquet(CACHE / f'side_va_{k}.parquet')
        meta[k] = dict(val_season=fold.val_season, fold_max_season=fold.fold_max_season,
                       n_tr=len(X_tr), n_va=len(X_va), cols=list(X_tr.columns))
        print(f"cached fold {k} val={fold.val_season} tr={len(X_tr)} va={len(X_va)} ({time.time()-t0:.0f}s)")
    marker.write_text(json.dumps(meta, indent=1))
    return meta


def get_fold(k, side=False):
    X_tr = pd.read_parquet(CACHE / f'X_tr_{k}.parquet')
    X_va = pd.read_parquet(CACHE / f'X_va_{k}.parquet')
    y_tr = np.load(CACHE / f'y_tr_{k}.npy')
    y_va = np.load(CACHE / f'y_va_{k}.npy')
    s_tr = np.load(CACHE / f's_tr_{k}.npy')
    if side:
        sd_tr = pd.read_parquet(CACHE / f'side_tr_{k}.parquet')
        sd_va = pd.read_parquet(CACHE / f'side_va_{k}.parquet')
        return X_tr, X_va, y_tr, y_va, s_tr, sd_tr, sd_va
    return X_tr, X_va, y_tr, y_va, s_tr


CAT_COLS = None


def cat_cols_of(X):
    return [c for c in X.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']


if __name__ == '__main__':
    m = build_cache(force='--force' in sys.argv)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cols'} for k, v in m.items()}, indent=1))
    print('n_features:', len(m[list(m)[0]]['cols']))
