import os
import shutil
import zipfile
import subprocess
import pandas as pd

BASE_DIR = os.path.expanduser('~/LG_data')
work_dir = os.path.join(BASE_DIR, 'work', 'submit_v48')

# 1. Clean agent2_asof_decomp2.py
decomp_path = os.path.join(work_dir, 'agent2_asof_decomp2.py')
with open(decomp_path, 'r') as f:
    decomp_code = f.read()

decomp_code = decomp_code.replace("sys.path.insert(0, os.path.expanduser('~/LG_data'))", "# Clean relative imports")
decomp_code = decomp_code.replace("import config\n\nTGT = config.TARGET_COL", "try:\n    import config\n    TGT = getattr(config, 'TARGET_COL', 'control_success')\nexcept ImportError:\n    TGT = 'control_success'")

with open(decomp_path, 'w') as f:
    f.write(decomp_code)
print("1. Polished agent2_asof_decomp2.py (removed hardcoded path, robust config fallback)")

# 2. Update script.py text
script_path = os.path.join(work_dir, 'script.py')
with open(script_path, 'r') as f:
    script_code = f.read()

script_code = script_code.replace("(v42 Neural Super-Ensemble)", "(v48 15-Seed SWA SimpleMLP Super-Ensemble)")
with open(script_path, 'w') as f:
    f.write(script_code)
print("2. Updated script.py banner to v48 15-Seed SWA SimpleMLP Super-Ensemble")

# 3. Purge all __pycache__, .DS_Store, and leftover temp files
for root, dirs, files in os.walk(work_dir, topdown=False):
    for f in files:
        if f.endswith('.pyc') or f == '.DS_Store':
            os.remove(os.path.join(root, f))
    for d in dirs:
        if d == '__pycache__':
            shutil.rmtree(os.path.join(root, d))
print("3. Purged all __pycache__, .pyc, and .DS_Store files")

# 4. Create clean submit_v48.zip
zip_path = os.path.join(BASE_DIR, 'work', 'submit_v48.zip')
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(work_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, work_dir)
            zf.write(full_path, rel_path)

print(f"4. Built pristine submit_v48.zip: {os.path.getsize(zip_path)/(1024*1024):.2f} MB")

# 5. Verify in isolated sandbox
sandbox_dir = '/tmp/v48_pristine_sandbox'
if os.path.exists(sandbox_dir):
    shutil.rmtree(sandbox_dir)
os.makedirs(sandbox_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(sandbox_dir)

test_data_dir = os.path.join(sandbox_dir, 'data')
os.makedirs(test_data_dir, exist_ok=True)
small_test_csv = os.path.join(test_data_dir, 'test.csv')
pd.DataFrame({
    'row_id': ['TEST_000001', 'TEST_000002'],
    'pitcher_id': [101, 102],
    'batter_id': [201, 202],
    'pitcher_hand': ['R', 'L'],
    'batter_hand': ['R', 'L'],
    'inning': [1, 2],
    'top_bottom': ['TOP', 'BOT'],
    'balls_before': [0, 1],
    'strikes_before': [0, 2],
    'pitcher_team_id': [1, 2],
    'batter_team_id': [2, 1],
    'score_diff_pitcher_team': [0, 1],
    'runner_on_1b': [0, 1],
    'runner_on_2b': [0, 0],
    'runner_on_3b': [0, 0],
    'li': [1.0, 1.2],
    'pitch_seq': [1, 2],
    'season': [2025, 2025],
    'asof_pitcher_success_rate': [0.5, 0.48],
    'asof_pitcher_n': [100, 200],
    'asof_pitcher_fastball_rate': [0.5, 0.6],
    'asof_pitcher_breaking_rate': [0.3, 0.2],
    'asof_pitcher_offspeed_rate': [0.2, 0.2],
    'asof_pitcher_pitchmix_n': [100, 200],
    'asof_pitcher_reverse_rate': [0.5, 0.52],
    'asof_pitcher_middle_rate': [0.1, 0.12],
    'asof_pitcher_ball_rate': [0.35, 0.38],
    'asof_pitcher_strike_rate': [0.65, 0.62],
    'asof_batter_success_rate': [0.5, 0.48],
    'asof_batter_n': [100, 200],
    'asof_batter_middle_rate': [0.1, 0.12],
    'tkm_rel_speed_mean': [145.0, 140.0],
    'tkm_extension_mean': [6.0, 5.8],
    'tkm_rel_side_mean': [-1.5, 1.2],
    'tkm_rel_height_mean': [5.8, 6.0],
    'tkm_induced_vert_break_mean': [15.0, 12.0],
    'tkm_horz_break_mean': [8.0, -7.0],
    'tkm_spin_rate_mean': [2200.0, 2100.0]
}).to_csv(small_test_csv, index=False)

res = subprocess.run([
    'python3',
    'script.py'
], cwd=sandbox_dir, capture_output=True, text=True)

if res.returncode != 0:
    print(f"FAILED: {res.stderr}")
else:
    print("5. Isolated Sandbox Output:")
    print(res.stdout)

# 6. Copy to pokemon
pokemon_zip = '~/pipeline_src/submit_v48.zip'
shutil.copy2(zip_path, pokemon_zip)
print(f"6. Clean pristine submit_v48.zip deployed to {pokemon_zip}")
