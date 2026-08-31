"""
ensemble_optimize.py — v33 앙상블 가중치 최적화 (실용적 접근)

전략:
- Preprocessing은 전체 데이터에 한번만 수행 (저장된 artifacts 사용)
- GBDT/MLP 모델만 3-fold temporal CV로 학습/검증
- 가중치 그리드서치로 최적 조합 탐색

이는 v33 script.py와 동일한 피처 세트를 사용하므로
가중치 최적화의 공정한 비교가 가능합니다.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Fix data path issue: config.py uses relative paths from submit_v33
# but actual data is at ~/LG_data/open/data
LG_DATA_ROOT = os.path.expanduser("~/LG_data")
DATA_DIR = os.path.join(LG_DATA_ROOT, "open", "data")

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn

from agent2_asof_decomp2 import AsofDecomposer2
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder

print("=" * 70)
print("V33 앙상블 가중치 최적화 — 3-Fold Temporal CV")
print("=" * 70)
t0 = time.time()

# ==========================================
# Configuration
# ==========================================
SEEDS = [7, 123, 2025, 31415, 8675309]
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

# Current baseline weights (v33)
BASELINE_W_MLP = 0.35
BASELINE_W_LGB = 0.15
BASELINE_W_CB = 0.75
BASELINE_W_XGB = 0.10

# Search grid for W_MLP
W_MLP_GRID = [0.33, 0.34, 0.35, 0.36, 0.37, 0.38]

# GBDT weight candidates (must sum to 1.0)
GBDT_WEIGHT_CANDIDATES = [
    # (LGB, CB, XGB, label)
    (0.15, 0.75, 0.10, "baseline"),
    (0.20, 0.72, 0.08, "proposed"),
    (0.18, 0.74, 0.08, "mid-1"),
    (0.20, 0.70, 0.10, "mid-2"),
    (0.22, 0.70, 0.08, "aggressive-lgb"),
    (0.17, 0.73, 0.10, "conservative"),
    (0.20, 0.75, 0.05, "low-xgb"),
    (0.15, 0.77, 0.08, "high-cb"),
]

# Calibration params from v33
CALIBRATION_SCALE = 1.10
CALIBRATION_SHIFT = -0.0045192086

# Shift corrections from v33
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

# CV Config (from config.py)
CV_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
CV_MIN_TRAIN_SEASONS = 3
# Folds: [2019-2021→2022], [2019-2022→2023], [2019-2023→2024]


# ==========================================
# Model Classes
# ==========================================
class CatEmbedder(nn.Module):
    def __init__(self, cat_cardinalities, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(max_emb_dim, max(2, int(card ** 0.25 * emb_dim))))
            for card in cat_cardinalities
        ])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        if len(self.embs) == 0:
            return torch.zeros(x_cat.shape[0], 0, device=x_cat.device)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class SimpleMLP(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


# ==========================================
# 1. Load data and prepare features (once)
# ==========================================
print("\n[1/4] Loading and preprocessing full dataset...")
train_path = os.path.join(DATA_DIR, "train.csv")
print(f"  Loading: {train_path}")
df_train = pd.read_csv(train_path)
print(f"  Loaded {len(df_train):,} rows, seasons: {sorted(df_train['season'].unique())}")

# Load existing preprocessor and transform
print("  Loading saved preprocessor artifacts...")
prep = joblib.load(os.path.join(MODEL_DIR, 'preprocessor_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder().load(os.path.join(MODEL_DIR, 'trackman_artifacts.pkl'))
if isinstance(prep, PitchPreprocessor):
    prep.trackman_builder = tkm_builder
else:
    # If prep is just the artifacts dict, create PitchPreprocessor
    prep_obj = PitchPreprocessor()
    prep_obj.artifacts = prep
    prep_obj.trackman_builder = tkm_builder
    prep_obj.is_fitted = True
    prep = prep_obj

X_full = prep.transform(df_train)

# Derive count_x_base
base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_full['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# 3D Pitch Tunneling
v0 = X_full['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_full['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_full['tkm_rel_side_mean']
rel_height = X_full['tkm_rel_height_mean']
ivb = X_full['tkm_induced_vert_break_mean'] / 12.0
hb = X_full['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_full['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_full['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_full['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

# Asof decomposition
dec = joblib.load(os.path.join(MODEL_DIR, 'asof_decomposer_artifacts.pkl'))
A_full = dec.transform(df_train)
A_full.index = X_full.index
X_full = pd.concat([X_full, A_full], axis=1)

y_full = df_train['control_success'].values
seasons = df_train['season'].values

cat_cols = [c for c in X_full.columns if c in [
    'top_bottom', 'base_state', 'pitcher_hand', 'batter_hand',
    'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup',
    'tkm_match', 'count_x_base'
]]
num_cols = [c for c in X_full.columns if c not in cat_cols]

print(f"  Feature matrix: {X_full.shape[1]} columns ({len(num_cols)} num, {len(cat_cols)} cat)")

# Prepare CatBoost/XGBoost variants
X_full_cb = X_full.copy()
for c in cat_cols:
    X_full_cb[c] = pd.to_numeric(X_full_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_full_cb.columns if col not in cat_cols]:
    X_full_cb[c] = pd.to_numeric(X_full_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

X_full_xgb = X_full.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_full_xgb[c] = X_full_xgb[c].astype(np.float32)
    else:
        X_full_xgb[c] = (X_full_xgb[c] - 1).astype(np.float32)
X_full_xgb = X_full_xgb.astype(np.float32)

# ==========================================
# 2. 3-Fold CV Model Training
# ==========================================
print(f"\n[2/4] Running 3-Fold Temporal CV...")

# Generate fold indices
fold_infos = []
for i in range(CV_MIN_TRAIN_SEASONS, len(CV_SEASONS)):
    train_seasons = CV_SEASONS[:i]
    val_season = CV_SEASONS[i]
    train_mask = np.isin(seasons, train_seasons)
    val_mask = (seasons == val_season)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    fold_infos.append((train_idx, val_idx, train_seasons, val_season))
    print(f"  Fold {i-CV_MIN_TRAIN_SEASONS}: train={train_seasons} ({len(train_idx):,} rows) → val={val_season} ({len(val_idx):,} rows)")

n = len(df_train)
oof_lgb = np.full(n, np.nan)
oof_cb = np.full(n, np.nan)
oof_xgb = np.full(n, np.nan)
oof_mlp = np.full(n, np.nan)

fold_aucs = []

for fold_i, (train_idx, val_idx, train_seasons, val_season) in enumerate(fold_infos):
    print(f"\n{'─'*60}")
    print(f"  Fold {fold_i}: train={train_seasons} → val={val_season}")
    print(f"{'─'*60}")

    y_tr = y_full[train_idx]
    y_va = y_full[val_idx]

    # ── LightGBM (5 seeds) ──
    print(f"  Training LightGBM (5 seeds)...")
    p_lgb_sum = np.zeros(len(val_idx))
    for seed in SEEDS:
        params_lgb = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': seed,
            'verbose': -1,
            'n_estimators': 300,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        dtrain = lgb.Dataset(X_full.iloc[train_idx], label=y_tr)
        m = lgb.train(params_lgb, dtrain)
        p_lgb_sum += m.predict(X_full.iloc[val_idx])
    p_lgb_fold = np.clip(p_lgb_sum / len(SEEDS) + S_LGB, 1e-6, 1 - 1e-6)

    # ── CatBoost (5 seeds) ──
    print(f"  Training CatBoost (5 seeds)...")
    p_cb_sum = np.zeros(len(val_idx))
    for seed in SEEDS:
        m_cb = CatBoostClassifier(
            iterations=300,
            learning_rate=0.06,
            depth=6,
            cat_features=cat_cols,
            random_seed=seed,
            verbose=0
        )
        m_cb.fit(X_full_cb.iloc[train_idx], y_tr)
        p_cb_sum += m_cb.predict_proba(X_full_cb.iloc[val_idx])[:, 1]
    p_cb_fold = np.clip(p_cb_sum / len(SEEDS) + S_CB, 1e-6, 1 - 1e-6)

    # ── XGBoost (5 seeds) ──
    print(f"  Training XGBoost (5 seeds)...")
    p_xgb_sum = np.zeros(len(val_idx))
    for seed in SEEDS:
        m_xgb = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            tree_method='hist',
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
        m_xgb.fit(X_full_xgb.iloc[train_idx], y_tr)
        p_xgb_sum += m_xgb.predict_proba(X_full_xgb.iloc[val_idx])[:, 1]
    p_xgb_fold = np.clip(p_xgb_sum / len(SEEDS) + S_XGB, 1e-6, 1 - 1e-6)

    # ── MLP (5 seeds) ──
    print(f"  Training MLP (5 seeds)...")
    # Build MLP artifacts from train fold
    mean = X_full.iloc[train_idx][num_cols].values.mean(axis=0).astype(np.float32)
    std_arr = X_full.iloc[train_idx][num_cols].values.std(axis=0).astype(np.float32)
    std_arr[std_arr < 1e-6] = 1.0

    cat_vocabs = {}
    cat_cardinalities = []
    for c in cat_cols:
        vals = X_full.iloc[train_idx][c].astype(str).unique()
        vocab = {v: i for i, v in enumerate(sorted(vals))}
        cat_vocabs[c] = vocab
        cat_cardinalities.append(len(vocab) + 1)

    # Train tensors
    num_z_tr = np.nan_to_num((X_full.iloc[train_idx][num_cols].values.astype(np.float32) - mean) / std_arr, nan=0.0)
    cat_arr_tr = []
    for c in cat_cols:
        unk_idx = len(cat_vocabs[c])
        cat_arr_tr.append(X_full.iloc[train_idx][c].astype(str).map(cat_vocabs[c]).fillna(unk_idx).values)
    cat_arr_tr = np.stack(cat_arr_tr, axis=1).astype(np.int64)

    X_num_t_tr = torch.tensor(num_z_tr, dtype=torch.float32)
    X_cat_t_tr = torch.tensor(cat_arr_tr, dtype=torch.int64)
    y_t_tr = torch.tensor(y_tr, dtype=torch.float32)

    # Val tensors
    num_z_va = np.nan_to_num((X_full.iloc[val_idx][num_cols].values.astype(np.float32) - mean) / std_arr, nan=0.0)
    cat_arr_va = []
    for c in cat_cols:
        unk_idx = len(cat_vocabs[c])
        cat_arr_va.append(X_full.iloc[val_idx][c].astype(str).map(cat_vocabs[c]).fillna(unk_idx).values)
    cat_arr_va = np.stack(cat_arr_va, axis=1).astype(np.int64)

    X_num_t_va = torch.tensor(num_z_va, dtype=torch.float32)
    X_cat_t_va = torch.tensor(cat_arr_va, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(X_num_t_tr, X_cat_t_tr, y_t_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)

    p_mlp_sum = np.zeros(len(val_idx))
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SimpleMLP(len(num_cols), cat_cardinalities, hidden=(128, 64), dropout=0.15)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(5):
            for bx_num, bx_cat, by in loader:
                opt.zero_grad()
                out = model(bx_num, bx_cat)
                loss = crit(out, by)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_num_t_va, X_cat_t_va)
            p = torch.sigmoid(logits).numpy()
        p_mlp_sum += np.clip(p, 1e-6, 1 - 1e-6)
    p_mlp_fold = np.clip(p_mlp_sum / len(SEEDS), 1e-6, 1 - 1e-6)

    # Store OOF predictions
    oof_lgb[val_idx] = p_lgb_fold
    oof_cb[val_idx] = p_cb_fold
    oof_xgb[val_idx] = p_xgb_fold
    oof_mlp[val_idx] = p_mlp_fold

    # Per-fold individual model AUC
    auc_lgb = roc_auc_score(y_va, p_lgb_fold)
    auc_cb = roc_auc_score(y_va, p_cb_fold)
    auc_xgb = roc_auc_score(y_va, p_xgb_fold)
    auc_mlp = roc_auc_score(y_va, p_mlp_fold)

    fold_aucs.append({
        'fold': fold_i,
        'val_season': val_season,
        'n_val': len(y_va),
        'auc_lgb': auc_lgb,
        'auc_cb': auc_cb,
        'auc_xgb': auc_xgb,
        'auc_mlp': auc_mlp,
    })

    print(f"\n  Fold {fold_i} Individual AUCs:")
    print(f"    LightGBM:  {auc_lgb:.6f}")
    print(f"    CatBoost:  {auc_cb:.6f}")
    print(f"    XGBoost:   {auc_xgb:.6f}")
    print(f"    MLP:       {auc_mlp:.6f}")

# ==========================================
# [3/4] Grid Search: W_MLP × GBDT Weights
# ==========================================
print(f"\n{'='*70}")
print("[3/4] 앙상블 가중치 그리드 서치")
print(f"{'='*70}")

# Valid OOF indices
valid_mask = ~(np.isnan(oof_lgb) | np.isnan(oof_cb) | np.isnan(oof_xgb) | np.isnan(oof_mlp))
y_valid = y_full[valid_mask]
lgb_valid = oof_lgb[valid_mask]
cb_valid = oof_cb[valid_mask]
xgb_valid = oof_xgb[valid_mask]
mlp_valid = oof_mlp[valid_mask]

print(f"  Valid OOF samples: {valid_mask.sum():,} / {n:,}")

results = []

for w_mlp in W_MLP_GRID:
    w_gbdt = 1.0 - w_mlp
    for w_lgb, w_cb, w_xgb, label in GBDT_WEIGHT_CANDIDATES:
        # Compute ensemble prediction
        p_gbdt = w_lgb * lgb_valid + w_cb * cb_valid + w_xgb * xgb_valid
        p_raw = w_gbdt * p_gbdt + w_mlp * mlp_valid

        # Apply calibration
        p_cal = np.clip(0.5 + CALIBRATION_SCALE * (p_raw - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)

        auc = roc_auc_score(y_valid, p_cal)
        ll = log_loss(y_valid, p_cal)

        # Also compute without calibration
        p_raw_clip = np.clip(p_raw, 1e-6, 1 - 1e-6)
        auc_raw = roc_auc_score(y_valid, p_raw_clip)
        ll_raw = log_loss(y_valid, p_raw_clip)

        results.append({
            'w_mlp': round(w_mlp, 2),
            'w_gbdt': round(w_gbdt, 2),
            'w_lgb': w_lgb,
            'w_cb': w_cb,
            'w_xgb': w_xgb,
            'label': label,
            'auc_calibrated': auc,
            'auc_raw': auc_raw,
            'logloss_cal': ll,
            'logloss_raw': ll_raw,
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('auc_calibrated', ascending=False).reset_index(drop=True)

# ==========================================
# [4/4] Results Summary
# ==========================================
print(f"\n{'='*70}")
print("[4/4] 최적화 결과 요약")
print(f"{'='*70}")

print("\n┌─────────────────────────────────────────────────────┐")
print("│ 개별 모델 Fold별 AUC                                │")
print("└─────────────────────────────────────────────────────┘")
fold_df = pd.DataFrame(fold_aucs)
for _, row in fold_df.iterrows():
    print(f"  Fold {int(row['fold'])} (val={int(row['val_season'])}, n={int(row['n_val']):,}):")
    print(f"    LGB={row['auc_lgb']:.6f}  CB={row['auc_cb']:.6f}  XGB={row['auc_xgb']:.6f}  MLP={row['auc_mlp']:.6f}")

print(f"\n  ── 3-Fold 평균 AUC ──")
print(f"    LightGBM:  {fold_df['auc_lgb'].mean():.6f}")
print(f"    CatBoost:  {fold_df['auc_cb'].mean():.6f}")
print(f"    XGBoost:   {fold_df['auc_xgb'].mean():.6f}")
print(f"    MLP:       {fold_df['auc_mlp'].mean():.6f}")

print(f"\n┌─────────────────────────────────────────────────────┐")
print(f"│ TOP 20 앙상블 가중치 조합 (by AUC calibrated)       │")
print(f"└─────────────────────────────────────────────────────┘")
top20 = results_df.head(20)
print(f"  {'W_MLP':>5} {'W_LGB':>5} {'W_CB':>5} {'W_XGB':>5} {'Label':<15} {'AUC_cal':>10} {'AUC_raw':>10} {'LL_cal':>10}")
print(f"  {'─'*75}")
for _, row in top20.iterrows():
    print(f"  {row['w_mlp']:5.2f} {row['w_lgb']:5.2f} {row['w_cb']:5.2f} {row['w_xgb']:5.2f} {row['label']:<15} {row['auc_calibrated']:10.6f} {row['auc_raw']:10.6f} {row['logloss_cal']:10.6f}")

# Best result
best = results_df.iloc[0]
print(f"\n{'─'*60}")
print(f"🏆 최적 가중치 조합:")
print(f"    W_MLP  = {best['w_mlp']:.2f}  (GBDT={best['w_gbdt']:.2f})")
print(f"    W_LGB  = {best['w_lgb']:.2f}  (GBDT 내부)")
print(f"    W_CB   = {best['w_cb']:.2f}  (GBDT 내부)")
print(f"    W_XGB  = {best['w_xgb']:.2f}  (GBDT 내부)")
print(f"    AUC (calibrated) = {best['auc_calibrated']:.6f}")
print(f"    AUC (raw)        = {best['auc_raw']:.6f}")
print(f"    LogLoss (cal)    = {best['logloss_cal']:.6f}")
print(f"{'─'*60}")

# Compare baseline vs best
baseline_row = results_df[
    (results_df['w_mlp'] == BASELINE_W_MLP) &
    (results_df['label'] == 'baseline')
]
if len(baseline_row) > 0:
    bl = baseline_row.iloc[0]
    print(f"\n📊 Baseline (v33) vs Best 비교:")
    print(f"  {'Metric':<20} {'Baseline':<15} {'Best':<15} {'Delta':<12}")
    print(f"  {'─'*62}")
    print(f"  {'AUC (calibrated)':<20} {bl['auc_calibrated']:.6f}       {best['auc_calibrated']:.6f}       {best['auc_calibrated']-bl['auc_calibrated']:+.6f}")
    print(f"  {'AUC (raw)':<20} {bl['auc_raw']:.6f}       {best['auc_raw']:.6f}       {best['auc_raw']-bl['auc_raw']:+.6f}")
    print(f"  {'LogLoss':<20} {bl['logloss_cal']:.6f}       {best['logloss_cal']:.6f}       {best['logloss_cal']-bl['logloss_cal']:+.6f}")
    print(f"\n  Baseline: W_MLP={bl['w_mlp']:.2f}, LGB={bl['w_lgb']:.2f}, CB={bl['w_cb']:.2f}, XGB={bl['w_xgb']:.2f}")
    print(f"  Best:     W_MLP={best['w_mlp']:.2f}, LGB={best['w_lgb']:.2f}, CB={best['w_cb']:.2f}, XGB={best['w_xgb']:.2f}")

# Per-fold AUC with best vs baseline weights
print(f"\n┌─────────────────────────────────────────────────────┐")
print(f"│ Fold별 Ensemble AUC (Best vs Baseline)              │")
print(f"└─────────────────────────────────────────────────────┘")
print(f"  {'Fold':<6} {'Val':<6} {'Baseline AUC':<15} {'Best AUC':<15} {'Δ AUC':<10}")
print(f"  {'─'*55}")

for fold_i, (train_idx, val_idx, train_seasons, val_season) in enumerate(fold_infos):
    y_va = y_full[val_idx]

    # Baseline ensemble
    p_gbdt_bl = BASELINE_W_LGB * oof_lgb[val_idx] + BASELINE_W_CB * oof_cb[val_idx] + BASELINE_W_XGB * oof_xgb[val_idx]
    p_raw_bl = (1 - BASELINE_W_MLP) * p_gbdt_bl + BASELINE_W_MLP * oof_mlp[val_idx]
    p_cal_bl = np.clip(0.5 + CALIBRATION_SCALE * (p_raw_bl - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)
    auc_bl = roc_auc_score(y_va, p_cal_bl)

    # Best ensemble
    p_gbdt_best = best['w_lgb'] * oof_lgb[val_idx] + best['w_cb'] * oof_cb[val_idx] + best['w_xgb'] * oof_xgb[val_idx]
    p_raw_best = best['w_gbdt'] * p_gbdt_best + best['w_mlp'] * oof_mlp[val_idx]
    p_cal_best = np.clip(0.5 + CALIBRATION_SCALE * (p_raw_best - 0.5) + CALIBRATION_SHIFT, 1e-6, 1 - 1e-6)
    auc_best = roc_auc_score(y_va, p_cal_best)

    print(f"  {fold_i:<6} {val_season:<6} {auc_bl:<15.6f} {auc_best:<15.6f} {auc_best-auc_bl:+.6f}")

# Save results
output_path = os.path.join(MODEL_DIR, "ensemble_optimization_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\n  전체 결과 저장: {output_path}")

# Save OOF predictions for downstream analysis
oof_path = os.path.join(MODEL_DIR, "oof_predictions.npz")
np.savez(oof_path, oof_lgb=oof_lgb, oof_cb=oof_cb, oof_xgb=oof_xgb, oof_mlp=oof_mlp, y=y_full, seasons=seasons)
print(f"  OOF 예측 저장: {oof_path}")

elapsed = time.time() - t0
print(f"\n총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
print("=" * 70)
