"""
audit_skill_calc_and_build_v2.py — Automated Audit & Submit V2 Packaging Script

Task 1: Standardized Skill Score Audit across calculation conventions.
Task 2: Retrain final Strong Reg 15 model on 1.47M rows into work/submit_v2/.
Task 3: Run dummy evaluation rehearsal v2 in work/dummy_eval_v2/.
"""

import sys, os, time, shutil, zipfile, subprocess, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder


def calc_fold_skill(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    unclipped_skill = 100000.0 * (1.0 - (brier / base_brier)) if base_brier > 0 else 0.0
    clipped_skill = max(0.0, unclipped_skill)
    return {
        "r": r,
        "brier": brier,
        "base_brier": base_brier,
        "unclipped_skill": unclipped_skill,
        "clipped_skill": clipped_skill
    }


print("======================================================================")
print("TASK 1: Standardized Skill Score Audit & Re-verification")
print("======================================================================")

df_all = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_all, strategy="time")

MODELS_AUDIT = {
    "Baseline (leaves=63, lr=0.05, min_child=20, n_est=300)": {
        "leaves": 63, "lr": 0.05, "min_child": 20, "colsample": 0.8, "subsample": 0.8, "n_est": 300
    },
    "Strong Reg 15 (leaves=15, lr=0.02, min_child=500, n_est=400)": {
        "leaves": 15, "lr": 0.02, "min_child": 500, "colsample": 0.6, "subsample": 0.6, "n_est": 400
    }
}

audit_summary_rows = []

for model_name, hp in MODELS_AUDIT.items():
    fold_res = []
    all_y_true = []
    all_y_prob = []

    for fi, fold in enumerate(folds):
        df_tr = df_all.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df_all.iloc[fold.val_idx].reset_index(drop=True)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        cat_cols = [c for c in X_va.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

        model = lgb.LGBMClassifier(
            n_estimators=hp["n_est"],
            num_leaves=hp["leaves"],
            learning_rate=hp["lr"],
            min_child_samples=hp["min_child"],
            colsample_bytree=hp["colsample"],
            subsample=hp["subsample"],
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
        preds = model.predict_proba(X_va)[:, 1]

        sk = calc_fold_skill(y_va, preds)
        sk["auc"] = roc_auc_score(y_va, preds)
        sk["fold"] = fi
        sk["val_season"] = fold.val_season
        fold_res.append(sk)

        all_y_true.append(y_va)
        all_y_prob.append(preds)

    # Pooled calculations across 746,504 rows
    y_true_pooled = np.concatenate(all_y_true)
    y_prob_pooled = np.concatenate(all_y_prob)
    sk_pooled = calc_fold_skill(y_true_pooled, y_prob_pooled)
    auc_pooled = roc_auc_score(y_true_pooled, y_prob_pooled)

    f0, f1, f2 = fold_res[0], fold_res[1], fold_res[2]

    # Metrics under all conventions
    mean_3fold_clipped = np.mean([f0["clipped_skill"], f1["clipped_skill"], f2["clipped_skill"]])
    mean_3fold_unclipped = np.mean([f0["unclipped_skill"], f1["unclipped_skill"], f2["unclipped_skill"]])
    mean_2fold_inner_clipped = np.mean([f0["clipped_skill"], f1["clipped_skill"]])
    mean_2fold_inner_unclipped = np.mean([f0["unclipped_skill"], f1["unclipped_skill"]])

    audit_summary_rows.append({
        "model_name": model_name,
        "f0_clipped": f0["clipped_skill"], "f0_unclipped": f0["unclipped_skill"],
        "f1_clipped": f1["clipped_skill"], "f1_unclipped": f1["unclipped_skill"],
        "f2_clipped": f2["clipped_skill"], "f2_unclipped": f2["unclipped_skill"],
        "mean_3fold_clipped": mean_3fold_clipped,
        "mean_3fold_unclipped": mean_3fold_unclipped,
        "mean_2fold_inner_clipped": mean_2fold_inner_clipped,
        "mean_2fold_inner_unclipped": mean_2fold_inner_unclipped,
        "pooled_skill_score": sk_pooled["clipped_skill"],
        "pooled_auc": auc_pooled
    })

audit_df = pd.DataFrame(audit_summary_rows)
audit_df.to_csv("~/LG_data/outputs/36_skill_audit_raw.csv", index=False)

print("\n--- STANDARDIZED SKILL SCORE AUDIT RESULT ---")
print(audit_df.to_string(index=False))


# ==============================================================================
# TASK 2: Final Retraining & Packaging work/submit_v2/
# ==============================================================================
print("\n======================================================================")
print("TASK 2: Retraining Final Strong Reg 15 Model into work/submit_v2/")
print("======================================================================")

SUBMIT_V2_DIR = os.path.join(config.WORK_DIR, "submit_v2")
MODEL_V2_DIR = os.path.join(SUBMIT_V2_DIR, "model")
ZIP_V2_PATH = os.path.join(config.WORK_DIR, "submit_v2.zip")

if os.path.exists(SUBMIT_V2_DIR):
    shutil.rmtree(SUBMIT_V2_DIR)
os.makedirs(MODEL_V2_DIR, exist_ok=True)

print("Fitting PitchPreprocessor in FINAL mode on full train.csv (1.47M rows)...")
t0 = time.perf_counter()
preprocessor = PitchPreprocessor()
preprocessor.fit(df_all, is_final=True)

X_full = preprocessor.transform(df_all)
y_full = df_all[config.TARGET_COL].values
assert X_full.shape[1] == 69, f"X_full has {X_full.shape[1]} cols, expected 69"

pp_art_v2 = os.path.join(MODEL_V2_DIR, "preprocessor_artifacts.pkl")
tkm_art_v2 = os.path.join(MODEL_V2_DIR, "trackman_artifacts.pkl")
preprocessor.save(pp_art_v2)
preprocessor.trackman_builder.save(tkm_art_v2)

cat_cols_full = [c for c in X_full.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

print("Training Strong Reg 15 LightGBM Booster on 1.47M rows...")
final_lgb = lgb.LGBMClassifier(
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
final_lgb.fit(
    X_full.values,
    y_full,
    categorical_feature=[X_full.columns.get_loc(c) for c in cat_cols_full if c in X_full.columns]
)
t1 = time.perf_counter()
print(f"Model retraining complete in {t1-t0:.2f}s.")

model_txt_v2 = os.path.join(MODEL_V2_DIR, "lgbm_model.txt")
final_lgb.booster_.save_model(model_txt_v2)
print(f"Saved Booster model to {model_txt_v2}")

# Write script.py for submit_v2
SCRIPT_CONTENT = """import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    t_start = time.perf_counter()
    print("=== DACON Evaluation Server Inference Script (v2) Starting ===")

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

with open(os.path.join(SUBMIT_V2_DIR, "script.py"), "w", encoding="utf-8") as f:
    f.write(SCRIPT_CONTENT)

with open(os.path.join(SUBMIT_V2_DIR, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write("lightgbm==4.7.0\njoblib==1.5.1\n")

print("Zipping submit_v2.zip ...")
if os.path.exists(ZIP_V2_PATH):
    os.remove(ZIP_V2_PATH)

with zipfile.ZipFile(ZIP_V2_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_V2_DIR):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, SUBMIT_V2_DIR)
            zipf.write(full_path, rel_path)

v2_zip_size_mb = os.path.getsize(ZIP_V2_PATH) / (1024 * 1024)
print(f"Created {ZIP_V2_PATH} ({v2_zip_size_mb:.2f} MB)")


# ==============================================================================
# TASK 3: Dummy Evaluation Server Rehearsal (2nd Rehearsal)
# ==============================================================================
print("\n======================================================================")
print("TASK 3: Dummy Evaluation Server Rehearsal v2")
print("======================================================================")

DUMMY_V2_DIR = os.path.join(config.WORK_DIR, "dummy_eval_v2")
if os.path.exists(DUMMY_V2_DIR):
    shutil.rmtree(DUMMY_V2_DIR)

dummy_data_dir = os.path.join(DUMMY_V2_DIR, "data")
dummy_output_dir = os.path.join(DUMMY_V2_DIR, "output")
os.makedirs(dummy_data_dir, exist_ok=True)
os.makedirs(dummy_output_dir, exist_ok=True)

shutil.copy(config.TEST_PATH, os.path.join(dummy_data_dir, "test.csv"))
shutil.copy(config.SAMPLE_SUB_PATH, os.path.join(dummy_data_dir, "sample_submission.csv"))

with zipfile.ZipFile(ZIP_V2_PATH, "r") as zipf:
    zipf.extractall(DUMMY_V2_DIR)

print("Running script.py inside dummy_eval_v2/ ...")
t_reh_start = time.perf_counter()
proc = subprocess.run(
    [sys.executable, "script.py"],
    cwd=DUMMY_V2_DIR,
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

print("\nSubmission CSV Verification (v2):")
print(f"  Generated shape: {res_sub.shape}")
print(f"  First few rows:")
print(res_sub.head().to_string(index=False))

assert list(res_sub.columns) == list(sample_sub.columns), "Columns mismatch!"
assert len(res_sub) == len(sample_sub), "Row count mismatch!"
assert res_sub["control_success"].isnull().sum() == 0, "Null values found in predictions!"

# Zip structure verification
with zipfile.ZipFile(ZIP_V2_PATH, "r") as zipf:
    namelist = zipf.namelist()
    assert "script.py" in namelist, "script.py missing from zip root!"
    assert "requirements.txt" in namelist, "requirements.txt missing from zip root!"
    assert "model/lgbm_model.txt" in namelist, "model/lgbm_model.txt missing!"

print("\n" + "=" * 60)
print("2차 제출 리허설 (REHEARSAL V2) PERFECT PASS!")
print("=" * 60)
print(f"Submit V2 zip path: {ZIP_V2_PATH} ({v2_zip_size_mb:.2f} MB)")
print(f"Rehearsal inference elapsed time: {reh_elapsed:.2f}s (10-min limit safety: {(600-reh_elapsed):.1f}s remaining)")
