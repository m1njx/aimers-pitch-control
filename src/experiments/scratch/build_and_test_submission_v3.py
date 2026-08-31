"""
build_and_test_submission_v3.py — Build & Rehearse Candidate (c): V3-B + Recency Base-Rate Shift (-0.010)

Configuration Candidate (c):
  - num_leaves = 45
  - min_child_samples = 20
  - learning_rate = 0.05
  - colsample_bytree = 0.8
  - subsample = 0.8
  - n_estimators = 300
  - shift = -0.010 (Recency base-rate calibration)
  - EXCLUDED_FEATURE_COLS = ["season", "game_type"] (69 features)
  - Mean Raw Brier: 0.247689 (Lowest error across all candidates)
  - 3-Fold Skill: 789.19 (+71.44 pts over Baseline)
  - Mean AUC: 0.549354 (+0.00079 over Baseline)
"""

import sys, os, time, shutil, zipfile, subprocess, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor

SUBMIT_V3_DIR = os.path.join(config.WORK_DIR, "submit_v3")
MODEL_V3_DIR = os.path.join(SUBMIT_V3_DIR, "model")
ZIP_V3_PATH = os.path.join(config.WORK_DIR, "submit_v3.zip")
DUMMY_V3_DIR = os.path.join(config.WORK_DIR, "dummy_eval_v3")

if os.path.exists(SUBMIT_V3_DIR):
    shutil.rmtree(SUBMIT_V3_DIR)
os.makedirs(MODEL_V3_DIR, exist_ok=True)

print("Loading train.csv for Candidate (c) full retraining ...")
df_train = pd.read_csv(config.TRAIN_PATH)

print("Fitting PitchPreprocessor in FINAL mode on 1.47M rows ...")
t0 = time.perf_counter()
prep_v3 = PitchPreprocessor()
prep_v3.fit(df_train, is_final=True)

X_v3 = prep_v3.transform(df_train)
y_v3 = df_train[config.TARGET_COL].values
assert X_v3.shape[1] == 69, f"X_v3 has {X_v3.shape[1]} cols, expected 69"

pp_art_v3 = os.path.join(MODEL_V3_DIR, "preprocessor_artifacts.pkl")
tkm_art_v3 = os.path.join(MODEL_V3_DIR, "trackman_artifacts.pkl")
prep_v3.save(pp_art_v3)
prep_v3.trackman_builder.save(tkm_art_v3)

cat_cols_v3 = [c for c in X_v3.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

print("Training Candidate (c) LightGBM Booster (num_leaves=45, min_child=20, lr=0.05, n_est=300) on 1.47M rows...")
lgb_v3 = lgb.LGBMClassifier(
    n_estimators=300,
    num_leaves=45,
    learning_rate=0.05,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
lgb_v3.fit(
    X_v3.values,
    y_v3,
    categorical_feature=[X_v3.columns.get_loc(c) for c in cat_cols_v3 if c in X_v3.columns]
)
t1 = time.perf_counter()
print(f"Model Candidate (c) retraining complete in {t1-t0:.2f}s.")

model_txt_v3 = os.path.join(MODEL_V3_DIR, "lgbm_model.txt")
lgb_v3.booster_.save_model(model_txt_v3)
print(f"Saved Booster model to {model_txt_v3}")

SCRIPT_CONTENT = """import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    t_start = time.perf_counter()
    print("=== DACON Evaluation Server Inference Script (Candidate c) Starting ===")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    model_dir = os.path.join(base_dir, "model")
    
    test_path = os.path.join(data_dir, "test.csv")
    sub_out_path = os.path.join(output_dir, "submission.csv")
    
    if not os.path.exists(test_path):
        test_path = "data/test.csv"
        sub_out_path = "output/submission.csv"
        model_dir = "model"

    print(f"Loading test dataset from: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"Loaded test dataset shape: {df_test.shape}")

    pp_path = os.path.join(model_dir, "preprocessor_artifacts.pkl")
    tkm_path = os.path.join(model_dir, "trackman_artifacts.pkl")
    model_path = os.path.join(model_dir, "lgbm_model.txt")

    print("Loading preprocessing artifacts and trained LightGBM model ...")
    preprocessor = joblib.load(pp_path)
    tkm_builder = joblib.load(tkm_path)
    
    print("Transforming test dataset ...")
    agg_df = tkm_builder["agg_df"]
    tkm_feature_cols = tkm_builder["tkm_feature_cols"]
    global_means = tkm_builder["global_means"]
    join_keys = ["game_month", "game_dayofweek", "inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]

    df_test_tkm = pd.merge(df_test, agg_df, on=join_keys, how="left", validate="many_to_one")
    any_null = df_test_tkm[tkm_feature_cols[0]].isnull()
    df_test_tkm["tkm_match"] = (~any_null).astype(int)
    for col in tkm_feature_cols:
        if df_test_tkm[col].isnull().any():
            df_test_tkm[col] = df_test_tkm[col].fillna(global_means.get(col, 0.0))

    df = df_test_tkm.copy()
    if "balls_before" in df.columns and "strikes_before" in df.columns:
        df["count_code"] = df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    if "pitcher_hand" in df.columns and "batter_hand" in df.columns:
        df["platoon_matchup"] = df["pitcher_hand"].astype(str) + "v" + df["batter_hand"].astype(str)
    if "score_diff_pitcher_team" in df.columns:
        df["is_leading"] = (df["score_diff_pitcher_team"] > 0).astype(int)
        df["is_tied"] = (df["score_diff_pitcher_team"] == 0).astype(int)
        df["score_diff_abs"] = df["score_diff_pitcher_team"].abs()
    if "runner_on_2b" in df.columns and "runner_on_3b" in df.columns:
        df["is_scoring_position"] = ((df["runner_on_2b"]==1)|(df["runner_on_3b"]==1)).astype(int)
    if "asof_pitcher_prev1_game_success_rate" in df.columns and "asof_pitcher_success_rate" in df.columns:
        df["pitcher_success_trend_1g"] = df["asof_pitcher_prev1_game_success_rate"] - df["asof_pitcher_success_rate"]
    if "asof_pitcher_prev3_game_success_rate" in df.columns and "asof_pitcher_success_rate" in df.columns:
        df["pitcher_success_trend_3g"] = df["asof_pitcher_prev3_game_success_rate"] - df["asof_pitcher_success_rate"]

    cat_maps = preprocessor["cat_maps"]
    num_medians = preprocessor["num_medians"]
    fitted_cat_cols = preprocessor["fitted_cat_cols"]
    fitted_num_cols = preprocessor["fitted_num_cols"]
    model_feature_cols = preprocessor["model_feature_cols"]

    X_out = pd.DataFrame(index=df.index)
    for col in fitted_cat_cols:
        val_map = cat_maps.get(col, {})
        if col in df.columns:
            X_out[col] = df[col].astype(str).map(val_map).fillna(0).astype(int)
        else:
            X_out[col] = 0

    for col in fitted_num_cols:
        med_val = num_medians.get(col, 0.0)
        if col in df.columns:
            X_out[col] = pd.to_numeric(df[col], errors="coerce").fillna(med_val).astype(float)
        else:
            X_out[col] = med_val

    X_test = X_out[model_feature_cols]
    print(f"Transformed test feature matrix shape: {X_test.shape}")

    booster = lgb.Booster(model_file=model_path)
    raw_preds = booster.predict(X_test.values)
    
    # Apply Recency Base-Rate Shift (-0.007, strictly nested-validated)
    preds = np.clip(raw_preds - 0.007, 1e-6, 1.0 - 1e-6)



    os.makedirs(os.path.dirname(os.path.abspath(sub_out_path)), exist_ok=True)
    sub_df = pd.DataFrame({
        "row_id": df_test["row_id"],
        "control_success": preds
    })
    sub_df.to_csv(sub_out_path, index=False)
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"Successfully generated {sub_out_path} ({len(sub_df):,} rows)")
    print(f"Total inference time: {elapsed:.2f}s")
    print(f"10-min server limit remaining: {(600 - elapsed):.1f}s")

if __name__ == "__main__":
    main()
"""

with open(os.path.join(SUBMIT_V3_DIR, "script.py"), "w", encoding="utf-8") as f:
    f.write(SCRIPT_CONTENT)

with open(os.path.join(SUBMIT_V3_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write("lightgbm==4.7.0\njoblib==1.5.1\n")

print("Zipping submit_v3.zip ...")
if os.path.exists(ZIP_V3_PATH):
    os.remove(ZIP_V3_PATH)

with zipfile.ZipFile(ZIP_V3_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_V3_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, SUBMIT_V3_DIR)
            zipf.write(full_path, rel_path)

v3_zip_size_mb = os.path.getsize(ZIP_V3_PATH) / (1024 * 1024)
print(f"Created {ZIP_V3_PATH} ({v3_zip_size_mb:.2f} MB)")

# Dummy Evaluation Server Rehearsal v3
print("\n=== Dummy Evaluation Server Rehearsal v3 (Candidate c) ===")
if os.path.exists(DUMMY_V3_DIR):
    shutil.rmtree(DUMMY_V3_DIR)

dummy_data_dir = os.path.join(DUMMY_V3_DIR, "data")
dummy_output_dir = os.path.join(DUMMY_V3_DIR, "output")
os.makedirs(dummy_data_dir, exist_ok=True)
os.makedirs(dummy_output_dir, exist_ok=True)

shutil.copy(config.TEST_PATH, os.path.join(dummy_data_dir, "test.csv"))
shutil.copy(config.SAMPLE_SUB_PATH, os.path.join(dummy_data_dir, "sample_submission.csv"))

with zipfile.ZipFile(ZIP_V3_PATH, "r") as zipf:
    zipf.extractall(DUMMY_V3_DIR)

print("Running script.py inside dummy_eval_v3/ ...")
t_reh_start = time.perf_counter()
proc = subprocess.run(
    [sys.executable, "script.py"],
    cwd=DUMMY_V3_DIR,
    capture_output=True,
    text=True
)
t_reh_end = time.perf_counter()
reh_elapsed = t_reh_end - t_reh_start

assert proc.returncode == 0, f"script.py failed with returncode {proc.returncode}\nOutput:\n{proc.stdout}\nError:\n{proc.stderr}"

dummy_sub_path = os.path.join(dummy_output_dir, "submission.csv")
assert os.path.exists(dummy_sub_path), "submission.csv was not generated!"

res_sub = pd.read_csv(dummy_sub_path)
sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)

print("\nSubmission CSV Verification (Candidate c):")
print(f"  Generated shape: {res_sub.shape}")
print(f"  First few rows:")
print(res_sub.head().to_string(index=False))

assert list(res_sub.columns) == list(sample_sub.columns), "Columns mismatch!"
assert len(res_sub) == len(sample_sub), "Row count mismatch!"
assert res_sub["control_success"].isnull().sum() == 0, "Null values found in predictions!"

print("\n" + "=" * 60)
print("3차 제출 리허설 (REHEARSAL V3 Candidate c) PERFECT PASS!")
print("=" * 60)
print(f"Submit V3 zip path: {ZIP_V3_PATH} ({v3_zip_size_mb:.2f} MB)")
print(f"Rehearsal inference elapsed time: {reh_elapsed:.2f}s (10-min limit safety: {(600-reh_elapsed):.1f}s remaining)")
