import os
import sys
import time
import numpy as np

print("=" * 70)
print("TRACK 4: Global Ensemble Weight & Calibration Optimization across 2022-2024 Folds")
print("=" * 70)
t0 = time.time()

BASE_DIR = os.path.expanduser('~/LG_data')
cache_dir = os.path.join(BASE_DIR, 'scratch', 'cache_final')

val_seasons = [2022, 2023, 2024]
caches = {}
for s in val_seasons:
    d = np.load(os.path.join(cache_dir, f'final_val{s}.npz'))
    caches[s] = {
        'y': d['y'].astype(np.float64),
        'p_lgb': d['p_lgb'].astype(np.float64),
        'p_cb': d['p_cb'].astype(np.float64),
        'p_xgb': d['p_xgb'].astype(np.float64)
    }

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

def eval_config(w_lgb, w_cb, w_xgb, s_lgb, s_cb, s_xgb, scale, shift):
    scores = []
    for s in val_seasons:
        c = caches[s]
        p1 = np.clip(c['p_lgb'] + s_lgb, 1e-6, 1 - 1e-6)
        p2 = np.clip(c['p_cb'] + s_cb, 1e-6, 1 - 1e-6)
        p3 = np.clip(c['p_xgb'] + s_xgb, 1e-6, 1 - 1e-6)
        p_raw = w_lgb * p1 + w_cb * p2 + w_xgb * p3
        p_final = np.clip(0.5 + scale * (p_raw - 0.5) + shift, 1e-6, 1 - 1e-6)
        sc, _ = calc_brier_skill_score(c['y'], p_final)
        scores.append(sc)
    return scores

# 1. v33 SSOT Baseline
v33_scores = eval_config(0.15, 0.75, 0.10, -0.007, -0.008, -0.006, 1.10, -0.0045192086)
print(f"\nv33 SSOT Baseline Scores:")
for s, sc in zip(val_seasons, v33_scores):
    print(f"  {s} Val: {sc:.2f}")
print(f"  3-Fold Mean: {np.mean(v33_scores):.2f}")

# 2. Grid Search over GBDT weights, Scale, and Shift
print("\n--- Running Systematic Grid Search ---")
best_cfg = None
best_mean = -1.0
best_2024 = -1.0

# Grid
lgb_weights = [0.15, 0.20, 0.25, 0.30]
scales = [1.08, 1.09, 1.10, 1.11, 1.12]
shifts = [-0.005, -0.0045, -0.0035, -0.0030, -0.0020, -0.0010]

for w_lgb in lgb_weights:
    w_xgb = 0.08
    w_cb = round(1.0 - w_lgb - w_xgb, 4)
    for sc in scales:
        for sh in shifts:
            scores = eval_config(w_lgb, w_cb, w_xgb, -0.007, -0.008, -0.006, sc, sh)
            mean_sc = np.mean(scores)
            sc_2024 = scores[2]
            # Constraint: 2024 must not degrade vs v33 (826.86)
            if sc_2024 >= v33_scores[2] and mean_sc > best_mean:
                best_mean = mean_sc
                best_cfg = (w_lgb, w_cb, w_xgb, sc, sh, scores)

print(f"\n[OPTIMAL CANDIDATE FOUND]")
w1, w2, w3, sc, sh, opt_scores = best_cfg
print(f"  Weights: LGB={w1}, CB={w2}, XGB={w3}")
print(f"  Calibration: SCALE={sc}, SHIFT={sh}")
print(f"  Per-Fold Scores:")
for s, orig, opt in zip(val_seasons, v33_scores, opt_scores):
    print(f"    {s} Val: {opt:.2f} (Gain vs v33: {opt - orig:+.2f} pts)")
print(f"  Optimal 3-Fold Mean: {np.mean(opt_scores):.2f} (Mean Gain: {np.mean(opt_scores) - np.mean(v33_scores):+.2f} pts)")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
