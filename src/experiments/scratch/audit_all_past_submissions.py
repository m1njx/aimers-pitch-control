import os
import glob
import re

BASE_DIR = os.path.expanduser('~/LG_data')

# Find all script.py in work directories or other submission dirs
scripts = glob.glob(f"{BASE_DIR}/work/**/script*.py", recursive=True)
scripts += glob.glob(f"{BASE_DIR}/track_gemini*/**/script*.py", recursive=True)
scripts += glob.glob(f"{BASE_DIR}/v*_script.py")

print(f"Found {len(scripts)} scripts. Analyzing...\n")

results = []

for s in sorted(set(scripts)):
    rel = os.path.relpath(s, BASE_DIR)
    with open(s, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract weights if present
    w_gbdt_bin = re.findall(r'W_GBDT_BIN\s*=\s*([0-9.]+)', content)
    w_mlp = re.findall(r'W_MLP(?:_MSE)?\s*=\s*([0-9.]+)', content)
    w_lgb_mse = re.findall(r'W_LGB(?:M)?_MSE\s*=\s*([0-9.]+)', content)
    scale = re.findall(r'(?:CALIBRATION_SCALE|SCALE)\s*=\s*([0-9.]+)', content)
    shift = re.findall(r'(?:CALIBRATION_SHIFT|SHIFT)\s*=\s*([0-9.\-]+)', content)
    
    # Also check if other blend weights exist
    blend_weights = re.findall(r'p_raw\s*=\s*([^;\n]+)', content)
    p_final = re.findall(r'p_calibrated\s*=\s*([^;\n]+)', content)
    
    results.append({
        'path': rel,
        'w_gbdt_bin': w_gbdt_bin[0] if w_gbdt_bin else '-',
        'w_mlp': w_mlp[0] if w_mlp else '-',
        'w_lgb_mse': w_lgb_mse[0] if w_lgb_mse else '-',
        'scale': scale[0] if scale else '-',
        'shift': shift[0] if shift else '-',
        'blend': blend_weights[0].strip() if blend_weights else '-',
    })

print(f"{'Path':<40} | {'GBDT':<6} | {'MLP':<6} | {'MSE':<6} | {'Scale':<6} | {'Shift':<12}")
print("-" * 90)
for r in results:
    print(f"{r['path']:<40} | {r['w_gbdt_bin']:<6} | {r['w_mlp']:<6} | {r['w_lgb_mse']:<6} | {r['scale']:<6} | {r['shift']:<12}")
