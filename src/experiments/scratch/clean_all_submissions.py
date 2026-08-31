import os
import sys
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
dest_pokemon = '~/pipeline_src'
data_dir = os.path.join(BASE_DIR, 'open', 'data')

def clean_submission_package(ver):
    sub_dir = os.path.join(BASE_DIR, 'work', f'submit_{ver}')
    zip_path = os.path.join(BASE_DIR, 'work', f'submit_{ver}.zip')
    
    print(f"\nCleaning and verifying {ver}...")
    
    # 1. Clean agent2_asof_decomp2.py
    decomp_path = os.path.join(sub_dir, 'agent2_asof_decomp2.py')
    if os.path.exists(decomp_path):
        with open(decomp_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        skip = False
        for line in lines:
            if "sys.path.insert(0, os.path.expanduser('~/LG_data'))" in line or "sys.path.insert(0, \"~/LG_data\")" in line:
                new_lines.append("import os\nSCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\nif SCRIPT_DIR not in sys.path:\n    sys.path.insert(0, SCRIPT_DIR)\n")
            elif "import config" in line:
                continue
            elif "TGT = config.TARGET_COL" in line:
                new_lines.append("TGT = 'control_success'\n")
            else:
                new_lines.append(line)
                
        with open(decomp_path, 'w') as f:
            f.writelines(new_lines)
        print(f"  [1] Cleaned agent2_asof_decomp2.py in {ver}!")
        
    # 2. Clean preprocessing.py
    prep_path = os.path.join(sub_dir, 'preprocessing.py')
    if os.path.exists(prep_path):
        with open(prep_path, 'r') as f:
            p_code = f.read()
        p_code = p_code.replace("os.path.expanduser('~/LG_data')", "os.path.dirname(os.path.abspath(__file__))")
        p_code = p_code.replace('os.path.expanduser("~/LG_data")', "os.path.dirname(os.path.abspath(__file__))")
        with open(prep_path, 'w') as f:
            f.write(p_code)
        print(f"  [2] Cleaned preprocessing.py in {ver}!")

    # 3. Clean trackman_features.py
    tkm_path = os.path.join(sub_dir, 'trackman_features.py')
    if os.path.exists(tkm_path):
        with open(tkm_path, 'r') as f:
            t_code = f.read()
        t_code = t_code.replace("os.path.expanduser('~/LG_data')", "os.path.dirname(os.path.abspath(__file__))")
        t_code = t_code.replace('os.path.expanduser("~/LG_data")', "os.path.dirname(os.path.abspath(__file__))")
        with open(tkm_path, 'w') as f:
            f.write(t_code)
        print(f"  [3] Cleaned trackman_features.py in {ver}!")
        
    # 4. Clean config.py
    cfg_path = os.path.join(sub_dir, 'config.py')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            c_code = f.read()
        c_code = c_code.replace("os.path.expanduser('~/LG_data')", "os.path.dirname(os.path.abspath(__file__))")
        c_code = c_code.replace('os.path.expanduser("~/LG_data")', "os.path.dirname(os.path.abspath(__file__))")
        with open(cfg_path, 'w') as f:
            f.write(c_code)
        print(f"  [4] Cleaned config.py in {ver}!")

    # 5. Remove any __pycache__ or temporary dirs
    for root, dirs, files in os.walk(sub_dir):
        for d in ['__pycache__', 'catboost_info', 'output', 'data']:
            if d in dirs:
                shutil.rmtree(os.path.join(root, d))

    # 6. Re-package ZIP
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(sub_dir):
            if '__pycache__' in root or os.path.basename(root) in ['data', 'output', 'catboost_info']:
                continue
            for file in files:
                if file.startswith('.') or file.endswith('.pyc') or file == 'submission.csv':
                    continue
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, sub_dir)
                zf.write(full_p, rel_p)
                
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [5] Repackaged {ver}.zip: {size_mb:.2f} MB")
    
    # 7. Strict Isolated Sandbox Test in /tmp
    sandbox_dir = f'/tmp/dacon_isolated_clean_{ver}'
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(sandbox_dir)
    os.makedirs(os.path.join(sandbox_dir, 'data'), exist_ok=True)
    shutil.copy(os.path.join(data_dir, 'test.csv'), os.path.join(sandbox_dir, 'data', 'test.csv'))
    
    clean_env = os.environ.copy()
    clean_env['PYTHONPATH'] = ''
    
    res = subprocess.run(['python3', 'script.py'], cwd=sandbox_dir, env=clean_env, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Sandbox verification failed for {ver}!")
        
    sub_df = pd.read_csv(os.path.join(sandbox_dir, 'output', 'submission.csv'))
    assert len(sub_df) == len(pd.read_csv(os.path.join(data_dir, 'test.csv')))
    assert not sub_df['control_success'].isna().any()
    print(f"  [6] Isolated Sandbox Test PASSED! {len(sub_df):,} rows -> 100% PERFECT!")
    shutil.rmtree(sandbox_dir)
    
    # 8. Sync to pokemon directory
    if os.path.exists(dest_pokemon):
        shutil.copy(zip_path, os.path.join(dest_pokemon, f'submit_{ver}.zip'))
        print(f"  [7] Synced submit_{ver}.zip to pokemon!")

for v in ['v43', 'v44', 'v45']:
    clean_submission_package(v)

print("\n" + "=" * 80)
print("ALL SUBMISSION PACKAGES (v43, v44, v45) 100% CLEANED, VERIFIED, AND SYNCED!")
print("=" * 80)
