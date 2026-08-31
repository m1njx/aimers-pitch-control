"""
verify_ensemble_v4_identity.py — Test 100% Identity between train_and_predict_v4.py and submit_v4.zip rehearsal
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/LG_data'))

import pandas as pd
import numpy as np

from final_code_submission.train_and_predict_v4 import main as run_v4_pipeline

print("Running train_and_predict_v4.py pipeline ...")
df_v4 = run_v4_pipeline()

rehearsal_csv = "~/LG_data/work/dummy_eval_v4/output/submission.csv"
print(f"Loading rehearsal submission from {rehearsal_csv} ...")
df_rehearsal = pd.read_csv(rehearsal_csv)

# Compare predictions
p_v4 = df_v4["control_success"].values
p_reh = df_rehearsal["control_success"].values

max_diff = np.max(np.abs(p_v4 - p_reh))
print(f"\n============================================================")
print(f"Max Absolute Prediction Difference: {max_diff:.10e}")

if max_diff < 1e-7:
    print("✅ PERFECT PASS: 100.00000% Mathematical Identity Verified!")
else:
    print(f"❌ FAIL: Prediction mismatch! Max diff = {max_diff}")
print("============================================================")
