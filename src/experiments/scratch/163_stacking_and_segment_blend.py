"""
163_stacking_and_segment_blend.py
사용자 지시("1000점 넘는 새로운 방식 탐색")의 아이디어 1(비선형 스태킹)과
아이디어 2(세그먼트별 동적 블렌딩)를 캐시된 5-seed OOF 예측(GBDT/TabM/TabR/MLP,
report 157/158 세션 산출물)을 재사용해 빠르게 검증.

방법론: 78번 보고서(선형 메타러너 스태킹 REJECT, -3.82~-81.48점)와 다르게
- 스태킹: 비선형 메타러너(shallow LightGBM, max_depth=3)
- 항상 inner(2022,2023)에서만 학습/선택 -> outer(2024) 최초 적용 (nested-honest)
- fold-averaged skill (pooled 금지) 원칙 준수
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_brier_skill_score

df = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df)
y_full = df[config.TARGET_COL].values
n = len(df)

d_gbdt = np.load('/tmp/gbdt_reference_5seed_oof.npz')
d_tabm = np.load('/tmp/tabm_5seed_oof.npz')
d_tabr = np.load('/tmp/tabr_5seed_oof.npz')
d_mlp = np.load('/tmp/baseline_mlp_5seed_oof.npz')

val_idx = d_gbdt['val_idx']
p_gbdt_full = np.full(n, np.nan)
p_gbdt_full[val_idx] = d_gbdt['p_ens']
p_tabm_full = d_tabm['oof']
p_tabr_full = d_tabr['oof']
p_mlp_full = d_mlp['oof']

print("=== Base model correlation (pooled val_idx) ===")
mat = np.vstack([p_gbdt_full[val_idx], p_tabm_full[val_idx], p_tabr_full[val_idx], p_mlp_full[val_idx]])
names = ['gbdt', 'tabm', 'tabr', 'mlp']
corr = np.corrcoef(mat)
for i, ni in enumerate(names):
    print(" ", ni, {names[j]: round(float(corr[i, j]), 4) for j in range(len(names))})

def fold_skills(p_full, fold_list):
    out = []
    for f in fold_list:
        vi = f.val_idx
        sk, _, _, _ = calc_brier_skill_score(y_full[vi], np.clip(p_full[vi], 1e-6, 1 - 1e-6))
        out.append(sk)
    return out

inner_folds = [f for f in folds if f.val_season in (2022, 2023)]
outer_fold = [f for f in folds if f.val_season == 2024][0]

print("\n=== Reference: single-model & existing linear GBDT+TabM blend (888.43 methodology) ===")
for label, p in [('gbdt', p_gbdt_full), ('tabm', p_tabm_full), ('tabr', p_tabr_full), ('mlp', p_mlp_full)]:
    sk_all = fold_skills(p, folds)
    print(f"  {label}: fold_skills={[round(s,2) for s in sk_all]} mean={np.mean(sk_all):.2f}")

best_w, best_inner = 0.0, -1
for w in np.linspace(0, 0.6, 31):
    p_blend = np.clip((1 - w) * p_gbdt_full + w * p_tabm_full, 1e-6, 1 - 1e-6)
    sk = np.mean(fold_skills(p_blend, inner_folds))
    if sk > best_inner:
        best_inner, best_w = sk, float(w)
p_blend_ref = np.clip((1 - best_w) * p_gbdt_full + best_w * p_tabm_full, 1e-6, 1 - 1e-6)
ref_full = np.mean(fold_skills(p_blend_ref, folds))
print(f"  linear GBDT+TabM blend: best_w_tabm(inner)={best_w:.2f} nested_full_skill={ref_full:.2f}")

# ============================================================
# IDEA 1: Nonlinear stacking (shallow LightGBM meta-learner)
# ============================================================
print("\n=== IDEA 1: Nonlinear stacking (shallow LGBM meta-learner, 4 base preds as features) ===")
meta_X_full = np.vstack([p_gbdt_full, p_tabm_full, p_tabr_full, p_mlp_full]).T  # (n,4), NaN outside val_idx

inner_idx = np.concatenate([f.val_idx for f in inner_folds])
outer_idx = outer_fold.val_idx

results_stack = {}
for base_set_name, cols in [('gbdt+tabm', [0, 1]), ('gbdt+tabm+tabr', [0, 1, 2]), ('all4', [0, 1, 2, 3])]:
    Xin = meta_X_full[inner_idx][:, cols]
    yin = y_full[inner_idx]
    Xout = meta_X_full[outer_idx][:, cols]

    meta = lgb.LGBMClassifier(max_depth=3, n_estimators=80, learning_rate=0.05,
                               num_leaves=7, min_child_samples=200, verbose=-1)
    meta.fit(Xin, yin)
    p_meta_outer = np.clip(meta.predict_proba(Xout)[:, 1], 1e-6, 1 - 1e-6)
    sk_outer, _, _, _ = calc_brier_skill_score(y_full[outer_idx], p_meta_outer)

    # nested-full: refit on (inner) only once (as above, standard nested = select/fit on inner, apply to ALL folds
    # for the "full" number, but inner-fold rows reuse the same meta -> only outer is truly honest;
    # report as outer-only (single honest holdout) which is the correct number, plus reference inner-fit score.
    p_meta_inner_selfcheck = np.clip(meta.predict_proba(Xin)[:, 1], 1e-6, 1 - 1e-6)  # circular, for reference only
    sk_inner_ref = np.mean([calc_brier_skill_score(y_full[f.val_idx],
                             np.clip(meta.predict_proba(meta_X_full[f.val_idx][:, cols])[:, 1], 1e-6, 1-1e-6))[0]
                             for f in inner_folds])
    print(f"  [{base_set_name}] outer(2024)-honest skill={sk_outer:.2f}  "
          f"(inner-refit-self-check, circular, NOT honest: {sk_inner_ref:.2f})")
    results_stack[base_set_name] = {'outer_honest': sk_outer, 'inner_circular_ref': sk_inner_ref}

# ============================================================
# IDEA 2: Segment-based dynamic blending (per count_x_base bucket weight)
# ============================================================
print("\n=== IDEA 2: Segment-based dynamic blending (per count-bucket w_tabm, GBDT+TabM only) ===")

def make_count_bucket(dframe):
    b = dframe['balls_before'].fillna(0).astype(int).clip(0, 3)
    s = dframe['strikes_before'].fillna(0).astype(int).clip(0, 2)
    return (b.astype(str) + '_' + s.astype(str)).values

bucket_full = make_count_bucket(df)
buckets = sorted(set(bucket_full[inner_idx].tolist()))
print(f"  buckets found: {buckets}")

seg_w = {}
for bk in buckets:
    idx_bk_inner = inner_idx[bucket_full[inner_idx] == bk]
    if len(idx_bk_inner) < 500:
        seg_w[bk] = best_w  # fallback to global weight if too few samples
        continue
    best_local_w, best_local_sk = best_w, -1e9
    for w in np.linspace(0, 0.8, 17):
        p_local = np.clip((1 - w) * p_gbdt_full[idx_bk_inner] + w * p_tabm_full[idx_bk_inner], 1e-6, 1 - 1e-6)
        sk, _, _, _ = calc_brier_skill_score(y_full[idx_bk_inner], p_local)
        if sk > best_local_sk:
            best_local_sk, best_local_w = sk, float(w)
    seg_w[bk] = best_local_w

print(f"  per-bucket best w_tabm (inner-selected): {seg_w}")

p_dynamic_full = np.full(n, np.nan)
for bk, w in seg_w.items():
    mask = (bucket_full == bk)
    idx_here = np.where(mask)[0]
    idx_here = idx_here[np.isin(idx_here, val_idx)]
    p_dynamic_full[idx_here] = np.clip((1 - w) * p_gbdt_full[idx_here] + w * p_tabm_full[idx_here], 1e-6, 1 - 1e-6)
# any val rows with bucket not seen in inner -> fallback to global best_w
missing = np.isin(val_idx, np.where(np.isnan(p_dynamic_full))[0])
if missing.any():
    idx_missing = val_idx[missing]
    p_dynamic_full[idx_missing] = np.clip((1 - best_w) * p_gbdt_full[idx_missing] + best_w * p_tabm_full[idx_missing], 1e-6, 1 - 1e-6)

sk_dynamic_full = np.mean(fold_skills(p_dynamic_full, folds))
sk_dynamic_outer = np.mean(fold_skills(p_dynamic_full, [outer_fold]))
print(f"  dynamic-segment blend: nested_full_skill={sk_dynamic_full:.2f} outer(2024)_only={sk_dynamic_outer:.2f} "
      f"(vs static linear blend full={ref_full:.2f})")

# ============================================================
# Summary
# ============================================================
summary = {
    'gbdt_alone': float(np.mean(fold_skills(p_gbdt_full, folds))),
    'linear_blend_full': float(ref_full),
    'linear_blend_best_w': best_w,
    'stacking': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results_stack.items()},
    'dynamic_segment_blend_full': float(sk_dynamic_full),
    'dynamic_segment_blend_outer_only': float(sk_dynamic_outer),
    'corr_matrix': corr.tolist(),
}
with open('/tmp/163_result.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n=== DONE. Saved /tmp/163_result.json ===")
print(json.dumps(summary, indent=2))
