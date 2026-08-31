"""Build v64: v50 with situational Trackman CatBoost and failure decomposition."""
import os, sys, shutil, zipfile

ROOT = os.path.expanduser("~/LG_data")
SRC = os.path.join(ROOT, "work", "submit_v63_failure_decomp")
DST = os.path.join(ROOT, "work", "submit_v64_situational_failure")
sys.path[:0] = [os.path.join(ROOT, "scratch"), ROOT]

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import config
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_common import base_cat_cols
from agent3_tkm_sit import build_situational, attach

SEEDS = [7, 123, 2025, 31415, 8675309]


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("data", "output", "work"))

    df = pd.read_csv(config.TRAIN_PATH)
    prep = PitchPreprocessor().fit(df, as_of_season=2024, is_final=True)
    X = prep.transform(df)
    X.index = df.index
    bases = ((df.runner_on_1b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (df.runner_on_2b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (df.runner_on_3b.fillna(0) > 0).astype(int).astype(str))
    raw_count = (df.balls_before.fillna(0).astype(int).astype(str) + "_" +
                 df.strikes_before.fillna(0).astype(int).astype(str) + "_" + bases)
    cmap = {v: i for i, v in enumerate(raw_count.unique())}
    X["count_x_base"] = raw_count.map(cmap).fillna(-1).astype(np.int32)
    dec = AsofDecomposer2().fit(df, val_season=2025)
    A = dec.transform(df); A.index = X.index
    X = pd.concat([X, A], axis=1)

    sit = build_situational(2024)
    X = attach(sit, df, X)
    cats = base_cat_cols(X)
    for c in cats:
        X[c] = X[c].fillna(-1).astype(str)
    for c in X.columns:
        if c not in cats:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)

    y = df.control_success.astype(np.int8).to_numpy()
    for seed in SEEDS:
        print(f"training situational CatBoost seed={seed}", flush=True)
        model = CatBoostClassifier(iterations=250, depth=6, learning_rate=.05,
            l2_leaf_reg=10, loss_function="Logloss", verbose=50, random_seed=seed,
            cat_features=cats, thread_count=-1, allow_writing_files=False)
        model.fit(X, y)
        model.save_model(os.path.join(DST, "model", f"catboost_situational_seed{seed}.cbm"))
    joblib.dump(sit, os.path.join(DST, "model", "situational_trackman_2024.pkl"))

    script_path = os.path.join(DST, "script.py")
    text = open(script_path).read()
    text = text.replace("v63 Failure-Decomposition Super-Ensemble", "v64 Situational+Failure Super-Ensemble")
    marker = "# Pre-cast feature matrices for instant C++ execution\n"
    inject = '''# Situational Trackman features: pitcher x count x batter-hand expectations\n+sit_table = joblib.load(os.path.join(model_dir, 'situational_trackman_2024.pkl'))\n+sit_key = pd.MultiIndex.from_arrays([df_test['pitcher_id'].values,\n+    df_test['balls_before'].clip(0, 3).values,\n+    df_test['strikes_before'].clip(0, 2).values, df_test['batter_hand'].values])\n+sit_add = sit_table.reindex(sit_key)\n+X_test_sit = X_test_base.copy()\n+for c in sit_table.columns:\n+    X_test_sit[c] = sit_add[c].values\n+\n+'''
    text = text.replace(marker, inject + marker)
    old = """    m_cb = CatBoostClassifier()\n+    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))\n+    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]\n+"""
    new = """    m_cb = CatBoostClassifier()\n+    m_cb.load_model(os.path.join(model_dir, f'catboost_situational_seed{seed}.cbm'))\n+    X_sit_cb = X_test_sit.copy()\n+    for c in cat_cols:\n+        X_sit_cb[c] = pd.to_numeric(X_sit_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)\n+    for c in [z for z in X_sit_cb.columns if z not in cat_cols]:\n+        X_sit_cb[c] = pd.to_numeric(X_sit_cb[c], errors='coerce').astype(np.float32)\n+    p_cb_sum += m_cb.predict_proba(X_sit_cb)[:, 1]\n+"""
    if old not in text:
        raise RuntimeError("CatBoost inference marker missing")
    text = text.replace(old, new)
    with open(script_path, "w") as f:
        f.write(text)

    out = DST + ".zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(DST):
            for fn in fs:
                p = os.path.join(dp, fn)
                z.write(p, os.path.relpath(p, DST))
    print(out, os.path.getsize(out))


if __name__ == "__main__":
    main()
