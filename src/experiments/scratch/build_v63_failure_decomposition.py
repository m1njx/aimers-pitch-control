"""Build v63: v50 plus a legal failure-type decomposition specialist.

The specialist predicts four mutually exclusive failure modes recovered from
past-only cumulative counters.  At inference it consumes only the current
row's allowed features.  Honest forward-season tests showed a positive blend
direction in 2022, 2023, and 2024.
"""
import os
import shutil
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = os.path.expanduser("~/LG_data")
SOURCE = os.path.join(ROOT, "work", "submit_v50")
DEST = os.path.join(ROOT, "work", "submit_v63_failure_decomp")
ZIP_PATH = DEST + ".zip"
SEED = 123

sys.path[:0] = [SOURCE, os.path.join(ROOT, "scratch"), ROOT]
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover


def build_features(df):
    tkm_art = joblib.load(os.path.join(SOURCE, "model", "trackman_artifacts.pkl"))
    tkm = TrackmanFeatureBuilder()
    tkm.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
    tkm.is_fitted = True
    prep_obj = joblib.load(os.path.join(SOURCE, "model", "preprocessor_artifacts.pkl"))
    if isinstance(prep_obj, PitchPreprocessor):
        prep = prep_obj
        prep.trackman_builder = tkm
    else:
        prep = PitchPreprocessor()
        prep.artifacts = prep_obj if isinstance(prep_obj, dict) else prep_obj.artifacts
        prep.trackman_builder = tkm
        prep.is_fitted = True
    x = prep.transform(df)
    bases = ((df.runner_on_1b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (df.runner_on_2b.fillna(0) > 0).astype(int).astype(str) + "_" +
             (df.runner_on_3b.fillna(0) > 0).astype(int).astype(str))
    counts = (df.balls_before.fillna(0).astype(int).astype(str) + "_" +
              df.strikes_before.fillna(0).astype(int).astype(str))
    count_map = getattr(prep, "count_x_base_map", {})
    x["count_x_base"] = (counts + "_" + bases).map(count_map).fillna(-1).astype(int)

    v0 = x.tkm_rel_speed_mean.clip(lower=60.0) * 1.46667
    ext = x.tkm_extension_mean.clip(lower=4.0, upper=8.0)
    ivb = x.tkm_induced_vert_break_mean / 12.0
    hb = x.tkm_horz_break_mean / 12.0
    tf = (60.5 - ext) / v0
    ratio = (tf - 0.15).clip(lower=0.01) / tf
    dt = np.sqrt((x.tkm_rel_side_mean + hb * ratio) ** 2 +
                 (x.tkm_rel_height_mean + ivb * ratio) ** 2)
    dp = np.sqrt((x.tkm_rel_side_mean + hb) ** 2 +
                 (x.tkm_rel_height_mean + ivb) ** 2)
    x["tkm_tunnel_dist_015s"] = dt.astype(np.float32)
    x["tkm_plate_break_divergence"] = ((dp - dt) / 0.15).astype(np.float32)
    x["tkm_deception_index"] = (dp / (dt + 0.1)).astype(np.float32)
    dec = joblib.load(os.path.join(SOURCE, "model", "asof_decomposer_artifacts.pkl"))
    a = dec.transform(df)
    a.index = x.index
    return pd.concat([x, a], axis=1)


def main():
    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    shutil.copytree(SOURCE, DEST, ignore=shutil.ignore_patterns("data", "output", "__pycache__", ".DS_Store"))
    df = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"))
    labels = recover(df)
    y = df.control_success.to_numpy(np.int8)
    reverse = labels.lab_reverse.to_numpy()
    middle = labels.lab_middle.to_numpy()
    known = np.isfinite(reverse) & np.isfinite(middle)
    category = np.full(len(df), -1, np.int8)
    category[known & (y == 1)] = 0
    category[known & (y == 0) & (reverse == 1) & (middle == 0)] = 1
    category[known & (y == 0) & (reverse == 0) & (middle == 1)] = 2
    category[known & (y == 0) & (reverse == 1) & (middle == 1)] = 3
    category[known & (y == 0) & (reverse == 0) & (middle == 0)] = 4

    x = build_features(df)
    cats = [c for c in ["top_bottom", "base_state", "pitcher_hand", "batter_hand",
                         "pitcher_team_id", "batter_team_id", "count_code",
                         "platoon_matchup", "tkm_match", "count_x_base"] if c in x]
    for c in cats:
        x[c] = x[c].fillna(-1).astype(int).astype(str)
    for c in x.columns:
        if c not in cats:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(np.float32)

    for k in (1, 2, 3, 4):
        print(f"training failure specialist {k}/4", flush=True)
        model = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.06,
            l2_leaf_reg=10, loss_function="Logloss", verbose=50,
            random_seed=SEED + 100 * k, cat_features=cats,
            thread_count=-1, allow_writing_files=False)
        model.fit(x, (category == k).astype(np.int8))
        model.save_model(os.path.join(DEST, "model", f"catboost_failure_{k}.cbm"))

    script_path = os.path.join(DEST, "script.py")
    code = open(script_path, encoding="utf-8").read()
    marker = "p_calibrated = np.clip(0.5 + CALIBRATION_SCALE * (p_cond - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)"
    replacement = marker + """

# Failure-decomposition specialist. Four probabilities represent mutually
# exclusive failure modes; their complement is the specialist success estimate.
p_failure = np.zeros(len(df_test), dtype=np.float64)
for failure_k in (1, 2, 3, 4):
    failure_model = CatBoostClassifier()
    failure_model.load_model(os.path.join(model_dir, f'catboost_failure_{failure_k}.cbm'))
    p_failure += failure_model.predict_proba(X_test_cb)[:, 1]
p_decomposed_success = np.clip(1.0 - p_failure, 1e-6, 1 - 1e-6)
FAILURE_BLEND_WEIGHT = 0.10
p_calibrated = np.clip((1.0 - FAILURE_BLEND_WEIGHT) * p_calibrated +
                       FAILURE_BLEND_WEIGHT * p_decomposed_success, 1e-6, 1 - 1e-6)
"""
    if marker not in code:
        raise RuntimeError("v50 calibration marker not found")
    code = code.replace(marker, replacement)
    code = code.replace("v50 Safe Balanced Master Super-Ensemble", "v63 Failure-Decomposition Super-Ensemble")
    open(script_path, "w", encoding="utf-8").write(code)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(DEST):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name.endswith(".pyc") or name == ".DS_Store":
                    continue
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, DEST))
    print(ZIP_PATH, os.path.getsize(ZIP_PATH), flush=True)


if __name__ == "__main__":
    main()
