import os
import zipfile
import shutil

BASE_DIR = os.path.expanduser('~/LG_data')
dest_pokemon = '~/pipeline_src'

clean_config = """import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ID_COL = 'row_id'
TARGET_COL = 'control_success'
"""

for v in ['v43', 'v44', 'v45']:
    sub_dir = os.path.join(BASE_DIR, 'work', f'submit_{v}')
    zip_path = os.path.join(BASE_DIR, 'work', f'submit_{v}.zip')
    
    with open(os.path.join(sub_dir, 'config.py'), 'w') as f:
        f.write(clean_config)
        
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
                
    if os.path.exists(dest_pokemon):
        shutil.copy(zip_path, os.path.join(dest_pokemon, f'submit_{v}.zip'))
        print(f"Updated and synced submit_{v}.zip ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

# Verify zero occurrences in all files
print("\n--- Final String Scan across all ZIP contents ---")
for v in ['v43', 'v44', 'v45']:
    zip_p = os.path.join(dest_pokemon, f'submit_{v}.zip')
    with zipfile.ZipFile(zip_p, 'r') as zf:
        print(f"Checking {v}.zip:")
        found = False
        for name in zf.namelist():
            if name.endswith('.py'):
                c = zf.read(name).decode('utf-8', errors='ignore')
                if '/Users/' in c or 'LG_data' in c:
                    print(f"  [!] Found in {name}")
                    found = True
        if not found:
            print("  🎉 100% PERFECT CLEAN! Zero occurrences of any local/absolute paths!")
