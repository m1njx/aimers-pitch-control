"""Screen officially permitted player-level historical Trackman profiles.

The competition Q&A reproduced in COMPETITION_RULES.md §5-1 explicitly permits
resolving pitcher/batter IDs and using per-player Trackman summaries from past
seasons.  This script tests only the pitcher profile part as a new feature
family, using the production v14/v16 116-feature GBDT recipe and immutable
per-row lookup tables.  It never reads another validation row while scoring.
"""
import gc
import os
import sys
import time

ROOT = os.path.expanduser("~/LG_data")
ZDIR = os.path.join(ROOT, "track_claude_z")
for path in (ZDIR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pandas as pd
import torch  # Import before GBDT libraries on this host.
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from eval_utils import calc_brier_skill_score
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

SEEDS = [7, 123]
W = (0.15, 0.75, 0.10)
SHIFT = (-0.007, -0.008, -0.006)
CACHE = os.path.join(ROOT, "scratch", "agent3_cache")
OUT = os.path.join(ROOT, "scratch", "id_trackman_screen_results.json")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def add_count_base(raw, X, mapping=None):
    bases = ((raw.runner_on_1b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (raw.runner_on_2b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (raw.runner_on_3b.fillna(0) > 0).astype(int).astype(str))
    count = (raw.balls_before.fillna(0).astype(int).astype(str) + "_" +
             raw.strikes_before.fillna(0).astype(int).astype(str))
    values = count + "_" + bases
    if mapping is None:
        mapping = {v: i for i, v in enumerate(values.unique())}
    X["count_x_base"] = values.map(mapping).fillna(-1).astype(np.int32)
    return mapping


def build_fold(df, fold):
    tr = df.iloc[fold.train_idx].copy()
    va = df.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor().fit(tr, as_of_season=fold.fold_max_season, is_final=False)
    Xtr, Xva = prep.transform(tr), prep.transform(va)
    cmap = add_count_base(tr, Xtr)
    add_count_base(va, Xva, cmap)
    dec = AsofDecomposer2().fit(tr, val_season=fold.val_season)
    for raw, X in ((tr, Xtr), (va, Xva)):
        D = dec.transform(raw)
        D.index = X.index
        for col in D:
            X[col] = D[col]
    for X in (Xtr, Xva):
        for col in X.columns:
            if X[col].dtype == np.float64:
                X[col] = X[col].astype(np.float32)
    return tr, va, Xtr, Xva


def attach_profiles(X, raw, as_of):
    # These tables were built from Trackman seasons <= as_of only.  The lookup
    # key is the current row's pitcher_id; no evaluation-batch aggregation.
    # The old experiment saved this table as Parquet, but the submission-like
    # runtime deliberately has no Parquet dependency. Rebuild it directly from
    # official CSVs, which is also the path a packaged submission would use.
    mapping = pd.read_csv(os.path.join(CACHE, "pitcher_map_raw.csv"))
    mapping = mapping[mapping.margin >= 0.3]
    id_map = dict(zip(mapping.tm_id, mapping.pitcher_id))
    cols = ["season", "pitcher_trackman_id", "pitch_type_group", "auto_pitch_type",
            "balls_before", "strikes_before", "batter_hand", "rel_speed", "spin_rate",
            "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
    tm = pd.read_csv(config.TRACKMAN_PATH, usecols=cols)
    tm = tm[tm.season <= as_of].copy()
    tm["pid"] = tm.pitcher_trackman_id.map(id_map)
    tm = tm.dropna(subset=["pid"])
    tm["pid"] = tm.pid.astype(np.int32)
    phys = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
    def block(data, prefix):
        a = data.groupby("pid")[phys].agg(["mean", "std"])
        a.columns = [f"{prefix}{name}_{stat}" for name, stat in a.columns]
        a[f"{prefix}n"] = data.groupby("pid").size()
        return a
    profile = block(tm, "idtk_all_")
    profile = profile.join(block(tm[tm.pitch_type_group == "fastball"], "idtk_fb_"))
    profile = profile.join(block(tm[tm.pitch_type_group == "breaking"], "idtk_br_"))
    profile = profile.join(block(tm[tm.season == as_of], "idtk_recent_"))
    mix = pd.crosstab(tm.pid, tm.pitch_type_group, normalize="index")
    mix.columns = [f"idtk_mix_{c}" for c in mix.columns]
    profile = profile.join(mix)
    profile["idtk_arsenal_size"] = tm.groupby("pid").auto_pitch_type.nunique()
    profile["idtk_velo_drop"] = profile["idtk_all_rel_speed_mean"] - profile["idtk_all_zone_speed_mean"]
    profile["idtk_move_mag"] = np.hypot(profile["idtk_all_induced_vert_break_mean"], profile["idtk_all_horz_break_mean"])
    profile["idtk_release_scatter"] = np.hypot(profile["idtk_all_rel_height_std"], profile["idtk_all_rel_side_std"])
    values = profile.reindex(raw.pitcher_id.to_numpy())
    values.index = X.index
    X = pd.concat([X, values.astype(np.float32)], axis=1)
    X["id_tkm_profile_mapped"] = values.notna().any(axis=1).astype(np.int8)
    # Pandas mutates the object passed by reference only for direct assignment;
    # return the compact concatenated frame explicitly.
    return X, list(values.columns) + ["id_tkm_profile_mapped"]


def cat_cols(X):
    base = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL, "count_x_base"]
    return [c for c in X.columns if c in base]


def predict(Xtr, Xva, y, cats, seed):
    lgbm = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                               min_child_samples=20, colsample_bytree=0.7,
                               subsample=0.7, verbosity=-1, n_jobs=-1,
                               random_state=seed)
    lgbm.fit(Xtr, y, categorical_feature=[Xtr.columns.get_loc(c) for c in cats])
    pl = lgbm.predict_proba(Xva)[:, 1] + SHIFT[0]

    trc, vac = Xtr.copy(), Xva.copy()
    for c in cats:
        trc[c] = trc[c].fillna(-1).astype(int).astype(str)
        vac[c] = vac[c].fillna(-1).astype(int).astype(str)
    cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05,
                            l2_leaf_reg=10.0, verbose=0, thread_count=-1,
                            random_seed=seed, cat_features=cats)
    cb.fit(trc, y)
    pc = cb.predict_proba(vac)[:, 1] + SHIFT[1]
    del trc, vac, cb

    trx, vax = Xtr.copy(), Xva.copy()
    for c in cats:
        if c != "count_x_base":
            trx[c] = trx[c].astype(np.float32) - 1
            vax[c] = vax[c].astype(np.float32) - 1
    xgbm = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                              colsample_bytree=0.8, subsample=0.8, n_jobs=-1,
                              eval_metric="logloss", random_state=seed)
    xgbm.fit(trx.astype(np.float32), y)
    px = xgbm.predict_proba(vax.astype(np.float32))[:, 1] + SHIFT[2]
    del lgbm, xgbm, trx, vax
    return np.clip(W[0] * pl + W[1] * pc + W[2] * px, 1e-6, 1 - 1e-6)


def main():
    df = pd.read_csv(config.TRAIN_PATH)
    results = {}
    for fold in get_cv_folds(df)[:2]:
        log(f"Building fold {fold.val_season}")
        tr, va, Xtr, Xva = build_fold(df, fold)
        assert Xtr.shape[1] == 116, Xtr.shape
        Xtr, pcols = attach_profiles(Xtr, tr, fold.fold_max_season)
        Xva, _ = attach_profiles(Xva, va, fold.fold_max_season)
        assert Xtr.shape[1] == 116 + len(pcols), Xtr.shape
        cats = cat_cols(Xtr)
        preds = []
        for seed in SEEDS:
            log(f"fold {fold.val_season}, seed {seed}, {Xtr.shape[1]} features")
            preds.append(predict(Xtr, Xva, tr.control_success.to_numpy(), cats, seed))
        p = np.mean(preds, axis=0)
        score, brier, base, rate = calc_brier_skill_score(va.control_success.to_numpy(), p)
        coverage = float(Xva["id_tkm_profile_mapped"].mean())
        results[str(fold.val_season)] = {"skill": score, "brier": brier, "base": base,
                                         "rate": rate, "coverage": coverage,
                                         "n_features": int(Xtr.shape[1])}
        log(f"RESULT fold={fold.val_season}: skill={score:.2f}, brier={brier:.6f}, profile coverage={coverage:.3%}")
        del tr, va, Xtr, Xva, preds
        gc.collect()
    results["inner_mean_skill"] = float(np.mean([v["skill"] for v in results.values() if isinstance(v, dict)]))
    import json
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved {OUT}: {results}")


if __name__ == "__main__":
    main()
