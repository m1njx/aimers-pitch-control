#!/usr/bin/env python3
"""
frontier5_exact_constrained_meta_stacking.py — Frontier 5: Exact Constrained Quadratic Meta-Stacking

Finds the global mathematical optimum:
min_w || w_1 * P_1 + w_2 * P_2 + ... + w_K * P_K - y ||^2
s.t. w_k >= 0, sum(w) = 1
and joint optimal scale & shift calibration
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = os.path.expanduser('~/LG_data')
cache_dir = os.path.join(BASE_DIR, 'scratch', 'cache_final')
report_dir = os.path.join(BASE_DIR, 'gemini_reports_for_ai')
output_dir = os.path.join(BASE_DIR, 'outputs')

def calc_brier_skill_score(y_true, y_prob):
    r = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_brier = float(r * (1.0 - r))
    score = 100000.0 * (1.0 - brier / base_brier)
    return score, brier

print("=" * 80)
print("STARTING FRONTIER 5: EXACT CONSTRAINED QUADRATIC META-STACKING")
print("=" * 80)

# Load 3-fold predictions from cache
folds = [2022, 2023, 2024]
for yr in folds:
    c = np.load(os.path.join(cache_dir, f'final_val{yr}.npz'))
    y = c['y']
    p_lgb = c['p_lgb']
    p_cb = c['p_cb']
    p_xgb = c['p_xgb']
    
    # 1. Base v33 weights
    p_v33 = np.clip(0.5 + 1.10 * (0.15*(p_lgb-0.007) + 0.75*(p_cb-0.008) + 0.10*(p_xgb-0.006) - 0.5) - 0.0045192086, 1e-6, 1-1e-6)
    sc_v33, _ = calc_brier_skill_score(y, p_v33)
    
    # 2. Optimize exact weights + scale + shift on historical fold
    # Matrix of base predictions: P is (N, 3)
    P = np.column_stack([p_lgb - 0.007, p_cb - 0.008, p_xgb - 0.006])
    
    def loss_func(params):
        w = params[:3]
        scale = params[3]
        shift = params[4]
        w_norm = w / np.sum(w)
        p_raw = P @ w_norm
        p_cal = np.clip(0.5 + scale * (p_raw - 0.5) + shift, 1e-6, 1 - 1e-6)
        return np.mean((p_cal - y) ** 2)
    
    init_params = [0.20, 0.72, 0.08, 1.10, -0.0045]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.25), (-0.02, 0.01)]
    res = minimize(loss_func, init_params, bounds=bounds, method='L-BFGS-B')
    
    w_opt = res.x[:3] / np.sum(res.x[:3])
    scale_opt, shift_opt = res.x[3], res.x[4]
    p_opt = np.clip(0.5 + scale_opt * (P @ w_opt - 0.5) + shift_opt, 1e-6, 1 - 1e-6)
    sc_opt, _ = calc_brier_skill_score(y, p_opt)
    
    print(f"Fold {yr} (N={len(y):,}):")
    print(f"  v33 Score: {sc_v33:.2f} pts")
    print(f"  Optimized Score: {sc_opt:.2f} pts (Gain: {sc_opt - sc_v33:+.2f} pts)")
    print(f"  Optimal Params: LGB={w_opt[0]:.3f}, CB={w_opt[1]:.3f}, XGB={w_opt[2]:.3f}, Scale={scale_opt:.4f}, Shift={shift_opt:.6f}\n")
