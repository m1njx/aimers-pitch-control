"""
check_unused_files.py — Workspace Unused File Audit Script
Scan all files in ~/LG_data and check usage across code, models, and data pipeline.
"""

import os, glob

root_dir = os.path.expanduser("~/LG_data")

all_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Ignore git or hidden dirs if any
    if "/.git" in dirpath or "/.idea" in dirpath:
        continue
    for f in filenames:
        if f == ".DS_Store":
            continue
        rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
        all_files.append(rel_path)

print(f"Total files in workspace: {len(all_files)}")

# Check open/ data files specifically
open_files = [f for f in all_files if f.startswith("open/")]
print("\n--- Files in 'open/' directory ---")
for f in sorted(open_files):
    full_p = os.path.join(root_dir, f)
    sz = os.path.getsize(full_p)
    print(f"  {f} ({sz / (1024*1024):.2f} MB)")

# Code files in root or final_code_submission or scratch
code_files = [f for f in all_files if f.endswith(".py")]
print(f"\n--- Python Code Files ({len(code_files)}) ---")
for f in sorted(code_files):
    print(f"  {f}")
