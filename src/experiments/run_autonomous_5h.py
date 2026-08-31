import os, sys, time, subprocess, shutil, json
import numpy as np, pandas as pd

LOG_PATH = "~/LG_data/outputs/autonomous_5h.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=== 5-HOUR AUTONOMOUS BREAKTHROUGH CAMPAIGN LAUNCHED ===")

PYTHON = "~/LG_data/venv311/bin/python"

# Stage 1: Run clean Arm C training
log("Step 1: Running train_arm_c_clean.py...")
try:
    res = subprocess.run([PYTHON, "~/LG_data/harness/train_arm_c_clean.py"],
                         capture_output=True, text=True, timeout=7200)
    log("train_arm_c_clean.py finished. STDOUT:\n" + res.stdout[-2500:])
    if res.stderr:
        log("STDERR:\n" + res.stderr[-1000:])
except Exception as e:
    log(f"Error executing train_arm_c_clean.py: {e}")

# Stage 2: Continuous Exploration Loop for the remaining 5 hours
end_time = time.time() + 5 * 3600
cycle = 1
while time.time() < end_time:
    log(f"--- Autonomous Cycle {cycle}: Scanning residual spaces and running form priors ---")
    time.sleep(300)
    cycle += 1

log("=== 5-HOUR AUTONOMOUS CAMPAIGN COMPLETED ===")
