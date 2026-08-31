"""
build_and_test_submission.py — Automates DACON submit.zip creation and dummy rehearsal.

Updated with Task 4 Optimal Strong Regularization LightGBM Hyperparameters:
  - num_leaves = 15
  - learning_rate = 0.02
  - n_estimators = 400
  - min_child_samples = 500
  - colsample_bytree = 0.6
  - subsample = 0.6
  - Brier Skill Score: 823.95 (up from 717.75)
  - AUC: 0.550855 (up from 0.548568)
"""
import sys, os, time, shutil, zipfile, subprocess, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from trackman_features import TrackmanFeatureBuilder
from preprocessing import PitchPreprocessor

# ── Step 0: Pre-check Whitelist and Feature Count ─────────────────────────────
print("=== Step 0: Pre-checking Feature Whitelist ===")
assert len(config.MODEL_FEATURE_COLS) == 69, f"Expected 69 features, got {len(config.MODEL_FEATURE_COLS)}"
assert "season" not in config.MODEL_FEATURE_COLS, "season should be excluded"
assert "game_type" not in config.MODEL_FEATURE_COLS, "game_type should be excluded"
assert "pitcher_team_id" in config.MODEL_FEATURE_COLS, "pitcher_team_id should be included"
assert "batter_team_id" in config.MODEL_FEATURE_COLS, "batter_team_id should be included"
print("✅ Whitelist assertions passed: 69 features verified.")

# Paths
SUBMIT_DIR = os.path.join(config.WORK_DIR, "submit")
MODEL_SAVE_DIR = os.path.join(SUBMIT_DIR, "model")
ZIP_PATH = os.path.join(config.WORK_DIR, "submit.zip")
DUMMY_DIR = os.path.join(config.WORK_DIR, "dummy_eval")

# Re-create submit directory structure
if os.path.exists(SUBMIT_DIR):
    shutil.rmtree(SUBMIT_DIR)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── Step 1: Train Final Model on Full train.csv ────────────────────────────────
print("\n=== Step 1: Fitting PitchPreprocessor & Training Strongly Regularized LightGBM Model ===")
t0 = time.perf_counter()
df_train = pd.read_csv(config.TRAIN_PATH)
print(f"Loaded train.csv: {df_train.shape}")

preprocessor = PitchPreprocessor()
preprocessor.fit(df_train, is_final=True)

X_train = preprocessor.transform(df_train)
y_train = df_train[config.TARGET_COL].values
assert X_train.shape[1] == 69, f"Transformed X_train has {X_train.shape[1]} cols, expected 69"
print(f"Transformed X_train shape: {X_train.shape}, NaN count: {X_train.isnull().sum().sum()}")

# Save artifacts into submit/model/
pp_artifact_path = os.path.join(MODEL_SAVE_DIR, "preprocessor_artifacts.pkl")
tkm_artifact_path = os.path.join(MODEL_SAVE_DIR, "trackman_artifacts.pkl")
preprocessor.save(pp_artifact_path)
preprocessor.trackman_builder.save(tkm_artifact_path)

# Train Strongly Regularized LightGBM model (Optimized for Brier Skill Score)
cat_cols = [c for c in X_train.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    num_leaves=15,
    learning_rate=0.02,
    min_child_samples=500,
    colsample_bytree=0.6,
    subsample=0.6,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

print("Training Strong-Reg LightGBM model (num_leaves=15, min_child=500) on 1.47M rows ...")
lgb_model.fit(
    X_train.values,
    y_train,
    categorical_feature=[X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
)
t1 = time.perf_counter()
print(f"Model training complete in {t1-t0:.2f}s.")

# Save LightGBM model
model_txt_path = os.path.join(MODEL_SAVE_DIR, "lgbm_model.txt")
lgb_model.booster_.save_model(model_txt_path)
print(f"Saved LightGBM model booster to {model_txt_path}")

# ── Step 2: Write script.py for Evaluation Server ──────────────────────────────
print("\n=== Step 2: Writing work/submit/script.py ===")
SCRIPT_CONTENT = """import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    t_start = time.perf_counter()
    print("=== DACON Evaluation Server Inference Script Starting ===")

    # 1. Define paths relative to evaluation server working directory
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
    preds = booster.predict(X_test.values)

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

script_file_path = os.path.join(SUBMIT_DIR, "script.py")
with open(script_file_path, "w", encoding="utf-8") as f:
    f.write(SCRIPT_CONTENT)

# ── Step 3: Write requirements.txt ─────────────────────────────────────────────
req_content = "lightgbm==4.7.0\njoblib==1.5.1\n"
req_file_path = os.path.join(SUBMIT_DIR, "requirements.txt")
with open(req_file_path, "w", encoding="utf-8") as f:
    f.write(req_content)

# ── Step 4: Zip submit.zip ────────────────────────────────────────────────────
print("\n=== Step 4: Creating work/submit.zip ===")
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, SUBMIT_DIR)
            zipf.write(full_path, rel_path)

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"Created {ZIP_PATH} ({zip_size_mb:.2f} MB)")

# ── Step 5: Dummy Evaluation Server Rehearsal ──────────────────────────────────
print("\n=== Step 5: Dummy Evaluation Server Rehearsal ===")
if os.path.exists(DUMMY_DIR):
    shutil.rmtree(DUMMY_DIR)

dummy_data_dir = os.path.join(DUMMY_DIR, "data")
dummy_output_dir = os.path.join(DUMMY_DIR, "output")
os.makedirs(dummy_data_dir, exist_ok=True)
os.makedirs(dummy_output_dir, exist_ok=True)

shutil.copy(config.TEST_PATH, os.path.join(dummy_data_dir, "test.csv"))
shutil.copy(config.SAMPLE_SUB_PATH, os.path.join(dummy_data_dir, "sample_submission.csv"))

with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
    zipf.extractall(DUMMY_DIR)

proc = subprocess.run(
    [sys.executable, "script.py"],
    cwd=DUMMY_DIR,
    capture_output=True,
    text=True
)

assert proc.returncode == 0, f"script.py failed with returncode {proc.returncode}"

dummy_sub_path = os.path.join(dummy_output_dir, "submission.csv")
assert os.path.exists(dummy_sub_path), "submission.csv was not generated!"

res_sub = pd.read_csv(dummy_sub_path)
sample_sub = pd.read_csv(config.SAMPLE_SUB_PATH)

print("\nSubmission CSV Verification:")
print(f"  Generated shape: {res_sub.shape}")
print(f"  First few rows:")
print(res_sub.head().to_string(index=False))

assert list(res_sub.columns) == list(sample_sub.columns), "Columns mismatch!"
assert len(res_sub) == len(sample_sub), "Row count mismatch!"
assert res_sub["control_success"].isnull().sum() == 0, "Null values found in predictions!"

print("\n" + "=" * 60)
print("REHEARSAL PASSED PERFECTLY!")
print("=" * 60)
print(f"Submit zip path: {ZIP_PATH} ({zip_size_mb:.2f} MB)")
