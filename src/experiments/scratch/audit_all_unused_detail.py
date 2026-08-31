"""
audit_all_unused_detail.py — Audit exact status of DACON baseline_submit files
"""

import os

root_dir = os.path.expanduser("~/LG_data")

baseline_files = [
    "open/baseline_submit.zip",
    "open/baseline_submit/model/rf.pkl",
    "open/baseline_submit/requirements.txt",
    "open/baseline_submit/script.py"
]

print("--- DACON Official Baseline Starter Kit File Audit ---")
for f in baseline_files:
    full_p = os.path.join(root_dir, f)
    exists = os.path.exists(full_p)
    sz = os.path.getsize(full_p) if exists else 0
    print(f"File: {f} | Exists: {exists} | Size: {sz / (1024*1024):.2f} MB")

