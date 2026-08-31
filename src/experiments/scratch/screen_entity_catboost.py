"""Temporal screen for CatBoost-native pitcher/batter identity features.

IDs are available in each prediction row and CatBoost receives them as fixed
categories. Its target statistics are learned from the training fold only;
there is no aggregation of validation/test rows. This is an allowed train-side
statistic lookup, but must earn its place by forward-season validation.
"""
import gc
import json
import os
import sys
import time

ROOT = os.path.expanduser("~/LG_data")
ZDIR = os.path.join(ROOT, "track_claude_z")
for p in (ZDIR, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2
from eval_utils import calc_brier_skill_score

SEEDS = [7, 123]
OUT = os.path.join(ROOT, "scratch", "entity_catboost_screen_results.json")
BASE_CATS = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL, "count_x_base"]


def log(x):
    print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)


def frame(raw_tr, raw_va, as_of, val_season):
    prep = PitchPreprocessor().fit(raw_tr, as_of_season=as_of, is_final=False)
    a, b = prep.transform(raw_tr), prep.transform(raw_va)
    def count_base(raw):
        bases = ((raw.runner_on_1b.fillna(0) > 0).astype(int).astype(str) + "_" +
                 (raw.runner_on_2b.fillna(0) > 0).astype(int).astype(str) + "_" +
                 (raw.runner_on_3b.fillna(0) > 0).astype(int).astype(str))
        return raw.balls_before.fillna(0).astype(int).astype(str) + "_" + raw.strikes_before.fillna(0).astype(int).astype(str) + "_" + bases
    mp = {x: i for i, x in enumerate(count_base(raw_tr).unique())}
    a["count_x_base"] = count_base(raw_tr).map(mp).fillna(-1).astype(np.int32)
    b["count_x_base"] = count_base(raw_va).map(mp).fillna(-1).astype(np.int32)
    dec = AsofDecomposer2().fit(raw_tr, val_season=val_season)
    da, db = dec.transform(raw_tr), dec.transform(raw_va)
    da.index, db.index = a.index, b.index
    a, b = pd.concat([a, da], axis=1), pd.concat([b, db], axis=1)
    return a, b


def catboost_pred(Xtr, Xva, y, add_entities, seed):
    tr, va = Xtr.copy(), Xva.copy()
    cats = [c for c in BASE_CATS if c in tr.columns]
    if add_entities:
        # Raw IDs are only introduced as categorical tokens, never ordinal values.
        tr["entity_pitcher_id"] = add_entities[0].astype(str).to_numpy()
        va["entity_pitcher_id"] = add_entities[1].astype(str).to_numpy()
        tr["entity_batter_id"] = add_entities[2].astype(str).to_numpy()
        va["entity_batter_id"] = add_entities[3].astype(str).to_numpy()
        cats += ["entity_pitcher_id", "entity_batter_id"]
    for c in cats:
        tr[c] = tr[c].fillna(-1).astype(str)
        va[c] = va[c].fillna(-1).astype(str)
    for c in tr.columns:
        if c not in cats:
            tr[c] = tr[c].astype(np.float32)
            va[c] = va[c].astype(np.float32)
    m = CatBoostClassifier(iterations=300, depth=6, learning_rate=.06,
                            l2_leaf_reg=10.0, verbose=0, random_seed=seed,
                            cat_features=cats, thread_count=-1)
    m.fit(tr, y)
    return np.clip(m.predict_proba(va)[:, 1] - .008, 1e-6, 1 - 1e-6)


def main():
    df = pd.read_csv(config.TRAIN_PATH)
    results = {}
    for fold in get_cv_folds(df)[:2]:
        tr = df.iloc[fold.train_idx].copy(); va = df.iloc[fold.val_idx].copy()
        log(f"building {fold.val_season}")
        Xtr, Xva = frame(tr, va, fold.fold_max_season, fold.val_season)
        assert Xtr.shape[1] == 116, Xtr.shape
        for name, entities in {
            "base": None,
            "pitcher_batter_ids": (tr.pitcher_id, va.pitcher_id, tr.batter_id, va.batter_id),
        }.items():
            ps = []
            for seed in SEEDS:
                log(f"fold={fold.val_season} candidate={name} seed={seed}")
                ps.append(catboost_pred(Xtr, Xva, tr.control_success.to_numpy(), entities, seed))
            p = np.mean(ps, axis=0)
            skill, brier, _, _ = calc_brier_skill_score(va.control_success.to_numpy(), p)
            results.setdefault(name, {})[str(fold.val_season)] = {"skill": skill, "brier": brier}
            log(f"RESULT fold={fold.val_season} {name}: {skill:.2f} / {brier:.6f}")
        del tr, va, Xtr, Xva
        gc.collect()
    for name, vals in results.items():
        vals["inner_mean"] = float(np.mean([vals["2022"]["skill"], vals["2023"]["skill"]]))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(results)


if __name__ == "__main__":
    main()
