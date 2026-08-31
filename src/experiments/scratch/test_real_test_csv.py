import subprocess
import shutil

sandbox_dir = '/tmp/v48_pristine_sandbox'
shutil.copy2('~/LG_data/open/data/test.csv', f'{sandbox_dir}/data/test.csv')

res = subprocess.run([
    'python3',
    'script.py'
], cwd=sandbox_dir, capture_output=True, text=True)

if res.returncode != 0:
    print(f"FAILED: {res.stderr}")
else:
    print("SUCCESS on official test.csv:")
    print(res.stdout)
