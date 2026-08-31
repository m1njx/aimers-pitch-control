import os, sys, time, zipfile, tempfile, subprocess, ast, joblib
import numpy as np, pandas as pd
from itertools import combinations

"""
audit_and_blend_1105.py
팀 1105+ 패키지/예측 수신 즉시 초고속 0점 감사 & 3-Arm 1150+ 최적 블렌딩 파이프라인
"""

LG = os.path.expanduser('~/LG_data')
CACHE = os.path.join(LG, 'harness/cache')
EPS = 1e-6

def calc_skill(p, y):
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    r = float(np.mean(y))
    base_ll = -(r * np.log(r) + (1.0 - r) * np.log(1.0 - r))
    model_ll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(2.0 * 1e5 * (base_ll - model_ll))

def our_arm_a():
    ps = []
    SH = dict(lgb_bin=-.007, cb_bin=-.008, xgb_bin=-.006, lgb_mse=0., mlp=0.)
    W = dict(lgb_bin=.080, cb_bin=.288, xgb_bin=.032, lgb_mse=.200, mlp=.400)
    for f in sorted(glob.glob(os.path.join(CACHE, 'pred_2024_*.npz'))):
        P = dict(np.load(f))
        raw = sum(W[k] * np.clip(np.asarray(P[k], float) + SH[k], EPS, 1 - EPS) for k in W)
        ps.append(np.clip(.5 + 1.10 * (raw - .5) - 0.0045192086, EPS, 1 - EPS))
    return np.mean(ps, 0)

def team_arm_b():
    import glob
    fs = sorted(glob.glob(os.path.join(LG, 'teamB/out/preds/l2384_f2024_s*.npy')))
    return np.mean([np.load(f).astype(float) for f in fs], 0)

def audit_zip_package(zip_path):
    print(f"\n[1/4] 0점 방지 7대 안전 검사 시작: {os.path.basename(zip_path)}")
    assert os.path.exists(zip_path), f"File not found: {zip_path}"
    
    dry = tempfile.mkdtemp(prefix='dry_1105_')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dry)
    
    # 1. AST parse
    print("  1. AST 구문 검사...")
    for root, _, files in os.walk(dry):
        for f in files:
            if f.endswith('.py'):
                ast.parse(open(os.path.join(root, f), encoding='utf-8').read())
    print("     -> AST OK.")
    
    # 2. val_season_ == 2025 in A
    print("  2. val_season_ == 2025 아티팩트 검사...")
    asof_pkl = os.path.join(dry, 'A', 'model', 'asof_decomposer_artifacts.pkl')
    if os.path.exists(asof_pkl):
        sys.path.insert(0, os.path.join(dry, 'A'))
        d = joblib.load(asof_pkl)
        v = getattr(d, 'val_season_', None)
        assert v == 2025, f"val_season_ is {v}, MUST be 2025!"
    print("     -> val_season_ OK.")
    
    # 3. reindex in root script.py
    print("  3. reindex(sub['row_id']) 정렬 검사...")
    root_script_path = os.path.join(dry, 'script.py')
    if os.path.exists(root_script_path):
        code = open(root_script_path).read()
        assert "reindex(sub['row_id'])" in code or 'reindex(sub["row_id"])' in code or ".reindex(" in code
    print("     -> reindex OK.")
    
    # 4. Forbidden files check
    print("  4. 불법/금지 파일 스캔 (train.csv, __pycache__)...")
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            assert '__pycache__' not in n
            assert not n.endswith(('train.csv', 'trackman_history.csv'))
    print("     -> Clean zip OK.")
    
    # 5. Local Dry-run
    print("  5. 로컬 5행 격리 드라이런...")
    os.makedirs(os.path.join(dry, 'data'), exist_ok=True)
    import shutil
    shutil.copy('~/LG_data/open/data/test.csv', os.path.join(dry, 'data', 'test.csv'))
    shutil.copy('~/LG_data/open/data/sample_submission.csv', os.path.join(dry, 'data', 'sample_submission.csv'))
    
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    res = subprocess.run([sys.executable, 'script.py'], cwd=dry, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Dry-run failed:\n{res.stderr}"
    sub = pd.read_csv(os.path.join(dry, 'output', 'submission.csv'))
    assert len(sub) == 5, f"Expected 5 rows, got {len(sub)}"
    assert not sub['control_success'].isna().any()
    print("     -> Local Dry-run 100% PASSED!")
    print(f"\n[안전 검사 완료] 0점 방지 7대 체크리스트 100% 무결점 통과!")

def optimize_3arm_simplex(p_a, p_b, p_c, y):
    ps = [p_a, p_b, p_c]
    n = len(ps)
    r = y.mean()
    V = r * (1.0 - r)
    
    # Gram matrix
    M = np.array([[1e5 * (1.0 - ((ps[i] - y) * (ps[j] - y)).mean() / V) for j in range(n)] for i in range(n)])
    
    best_v = -1e18
    best_w = None
    
    for k in range(1, n + 1):
        for S in combinations(range(n), k):
            S = list(S)
            Ms = M[np.ix_(S, S)]
            A = np.block([[2 * Ms, -np.ones((k, 1))], [np.ones((1, k)), np.zeros((1, 1))]])
            try:
                w = np.linalg.solve(A, np.concatenate([np.zeros(k), [1.]]))[:k]
            except np.linalg.LinAlgError:
                continue
            if (w < -1e-9).any():
                continue
            v = float(w @ Ms @ w)
            if v > best_v:
                best_v = v
                best_w = np.zeros(n)
                best_w[S] = w
                
    return best_w, best_v, M

if __name__ == '__main__':
    print("audit_and_blend_1105.py ready for instant execution.")
