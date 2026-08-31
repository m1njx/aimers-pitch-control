"""
agent6_outer_2024_test.py

Inner folds (2022, 2023)에서 발견한 w_mlp=0.32를
outer (2024) fold에 정확히 1회만 적용하는 정직한 검증.

예상: +41.59 점 개선 → v50 1032 + 41 = 1073 목표
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')
SEED = 7
W_MLP = 0.32  # Inner에서 발견한 최적 공유 가중치
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}

LGB_PARAMS = dict(n_estimators=250, num_leaves=45, learning_rate=0.05, min_child_samples=20,
                   colsample_bytree=0.7, subsample=0.7, random_state=SEED, verbosity=-1, n_jobs=-1)
CB_PARAMS = dict(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                  random_seed=SEED, verbose=0, thread_count=-1)
XGB_PARAMS = dict(n_estimators=250, max_depth=5, learning_rate=0.05, colsample_bytree=0.8,
                   subsample=0.8, random_state=SEED, n_jobs=-1, eval_metric='logloss')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

print("="*100)
print("                Agent6 MLP Blending - Outer(2024) Validation")
print("="*100)
print(f"Inner results recap:")
print(f"  fold 2022: GBDT=2067.92, best blend w=0.28 → 2099.45 (+31.53)")
print(f"  fold 2023: GBDT=684.71, best blend w=0.36 → 737.16 (+52.45)")
print(f"  Shared w_mlp=0.32: inner avg=1417.91 vs GBDT=1376.32 (+41.59)")
print(f"\nNow testing on outer fold (2024)...")
print("="*100)

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train)
fold_2024 = [f for f in folds if f.val_season == 2024][0]

log(f"Building fold 2024 (outer) feature frames...")
df_tr = df_train.iloc[fold_2024.train_idx].copy()
df_val = df_train.iloc[fold_2024.val_idx].copy()

prep = PitchPreprocessor()
prep.fit(df_tr, as_of_season=2023, is_final=False)
X_tr = prep.transform(df_tr)
X_val = prep.transform(df_val)

dlc.add_count_x_base(df_tr, X_tr)
dlc.add_count_x_base(df_val, X_val)
cat_map = {v: i for i, v in enumerate(X_tr['count_x_base'].unique())}
X_tr['count_x_base'] = X_tr['count_x_base'].map(cat_map).fillna(-1).astype(int)
X_val['count_x_base'] = X_val['count_x_base'].map(cat_map).fillna(-1).astype(int)

dec = AsofDecomposer2().fit(df_tr, val_season=2024)
tr_feats = dec.transform(df_tr)
val_feats = dec.transform(df_val)
tr_feats.index = X_tr.index
val_feats.index = X_val.index
X_tr = pd.concat([X_tr, tr_feats], axis=1)
X_val = pd.concat([X_val, val_feats], axis=1)

y_tr = df_tr[config.TARGET_COL].values.astype(np.float32)
y_val = df_val[config.TARGET_COL].values.astype(np.float32)

log(f"X_tr={X_tr.shape} X_val={X_val.shape}")

# GBDT
cat_cols = [c for c in X_tr.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
            or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

log("Training GBDT ensemble (LGB+CB+XGB)...")
m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
m_lgb.fit(X_tr, y_tr, categorical_feature=cat_idx)
p_lgb = np.clip(m_lgb.predict_proba(X_val)[:, 1] + SHIFTS['lgb'], 1e-6, 1 - 1e-6)

X_tr_cb, X_val_cb = X_tr.copy(), X_val.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
    X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
m_cb = CatBoostClassifier(cat_features=cat_cols, **CB_PARAMS)
m_cb.fit(X_tr_cb, y_tr)
p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + SHIFTS['cb'], 1e-6, 1 - 1e-6)

X_tr_x, X_val_x = X_tr.copy(), X_val.copy()
for c in cat_cols:
    X_tr_x[c] = X_tr_x[c].astype('category').cat.codes.astype(np.float32)
    X_val_x[c] = X_val_x[c].astype('category').cat.codes.astype(np.float32)
m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
m_xgb.fit(X_tr_x.astype(np.float32), y_tr)
p_xgb = np.clip(m_xgb.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS['xgb'], 1e-6, 1 - 1e-6)

w_lgb, w_cb, w_xgb = WEIGHTS
p_gbdt = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
sk_gbdt, br_gbdt, _, _ = calc_brier_skill_score(y_val, p_gbdt)
log(f"GBDT alone: skill={sk_gbdt:.2f} brier={br_gbdt:.6f}")

# MLP
log("Training MLP (CPU, single seed)...")
tens = dlc.to_tensors(X_tr, X_val)
num_tr, num_val = tens['num_tr'], tens['num_val']
cat_tr, cat_val = tens['cat_tr'], tens['cat_val']
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

torch.manual_seed(SEED)
np.random.seed(SEED)
model = dlc.SimpleMLP(num_tr.shape[1], tens['cat_cardinalities'], hidden=(128, 64), dropout=0.15)
model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=8, lr=1e-3,
                                  batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                  verbose_prefix=f"[mlp 2024] ")
p_mlp = dlc.predict(model, num_val, cat_val, DEVICE, shift)
sk_mlp, br_mlp, _, _ = calc_brier_skill_score(y_val, p_mlp)
log(f"MLP alone: skill={sk_mlp:.2f} brier={br_mlp:.6f}")

corr = float(np.corrcoef(p_mlp, p_gbdt)[0, 1])
log(f"corr(MLP, GBDT) = {corr:.4f}")

# Blend with inner-discovered w_mlp=0.32
p_blend = np.clip((1 - W_MLP) * p_gbdt + W_MLP * p_mlp, 1e-6, 1 - 1e-6)
sk_blend, br_blend, _, _ = calc_brier_skill_score(y_val, p_blend)

print("\n" + "="*100)
print("📊 OUTER(2024) VALIDATION RESULTS")
print("="*100)
print(f"   GBDT alone:  Skill = {sk_gbdt:.2f}")
print(f"   MLP alone:   Skill = {sk_mlp:.2f}")
print(f"   Blended (w_mlp={W_MLP:.2f}): Skill = {sk_blend:.2f}")
print(f"   Gain:        {sk_blend - sk_gbdt:+.2f} pts")
print("="*100)

V50_BASELINE = 1032
estimated_lb = V50_BASELINE + (sk_blend - sk_gbdt)
print(f"\n🎯 Expected Public LB: ~{estimated_lb:.0f} (v50 {V50_BASELINE} + {sk_blend - sk_gbdt:+.2f})")
print(f"   Target: 1150")
print(f"   Gap: {1150 - estimated_lb:+.0f}")
print("="*100)
