"""
build_submission_v14.py

GBDT(asof_dec, 5-seed) + SimpleMLP(asof_dec, 5-seed) blend, w_mlp=0.32 (frozen,
inner-only selected by agent6, re-confirmed on outer(2024) with proper 5-seed
bagging in report 205: outer gain=+8.56, row-independence PASS empirically
(max_diff=5.96e-08)).

GBDT side: REUSES v13's already-trained model artifacts verbatim (no retraining --
asof_dec features/hyperparameters are unchanged; v12->v13 only changed the
inference-side XGB categorical encoding, and the already-trained XGB model is
compatible with that fix since train.csv's dense category ranges made the old
per-fold `.cat.codes` on the FULL training set equal to `value-1` anyway --
see report 203 section 3). Uses v11-style PER-MODEL FIXED shifts
(-0.007/-0.008/-0.006 applied before blending, weights 0.15/0.75/0.10) --
this matches EXACTLY what report 205's "GBDT-alone" reference used (NOT v12/13's
post-blend extrapolated shift -- that combination has never been validated
together with the MLP blend, so we don't ship an untested combination).

MLP side: trains 5 seeds fresh on FULL train.csv (is_final style, matching how
v11's GBDT was trained on all of 2019-2024), using the exact same architecture/
recipe as agent6/7/8's SimpleMLP screening (dl_common.SimpleMLP, hidden=(128,64),
dropout=0.15, 8 epochs, patience=2, CPU-forced). Categorical vocab + numeric
normalization stats are fit from the FULL train.csv and saved as artifacts
(train-only, row-independent by construction -- same pattern verified in 205).
"""
import sys, os, shutil, zipfile, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

import config
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')  # force CPU -- MPS stall history (report 161)
BASE_DIR = Path(os.path.expanduser('~/LG_data'))
SRC_V13_MODEL = BASE_DIR / 'work/submit_v13/model'
SUBMIT_DIR = BASE_DIR / 'work/submit_v14'
MODEL_DIR = SUBMIT_DIR / 'model'

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006  # matches report 205's GBDT-alone reference exactly
W_MLP = 0.32  # frozen, inner-only selected (agent6), outer-confirmed 5-seed (report 205, gain=+8.56)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if SUBMIT_DIR.exists():
    shutil.rmtree(SUBMIT_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("[Task 0] Reuse v13 GBDT model artifacts verbatim (no retraining)")
print("=" * 70)
gbdt_files = ['asof_decomposer_artifacts.pkl', 'preprocessor_artifacts.pkl', 'trackman_artifacts.pkl'] + \
    [f'{fam}_model_seed{s}.{ext}' for fam, ext in [('lgbm', 'txt'), ('catboost', 'cbm'), ('xgb', 'json')] for s in SEEDS]
for fname in gbdt_files:
    shutil.copy(SRC_V13_MODEL / fname, MODEL_DIR / fname)
    print(f"  copied: {fname}")

print("\n" + "=" * 70)
print("[Task 1] MLP: full-data preprocessing (matches v13's GBDT pipeline exactly)")
print("=" * 70)

t0_train = time.time()
df_train = pd.read_csv(config.TRAIN_PATH)
y_train = df_train[config.TARGET_COL].values.astype(np.float32)
print(f"Loaded train dataset: {df_train.shape[0]:,} rows")

prep = PitchPreprocessor()
prep.fit(df_train, as_of_season=None, is_final=True)
X_train = prep.transform(df_train)
dlc.add_count_x_base(df_train, X_train)
cat_map_countbase = {v: i for i, v in enumerate(X_train['count_x_base'].unique())}
X_train['count_x_base'] = X_train['count_x_base'].map(cat_map_countbase).fillna(-1).astype(int)

print("\n--- asof_dec 분해 피처 fit (전체 train.csv, deploy val_season=2025) ---")
dec = AsofDecomposer2().fit(df_train, val_season=2025)
A_train = dec.transform(df_train)
A_train.index = X_train.index
X_train = pd.concat([X_train, A_train], axis=1)
print(f"최종 피처 수: {X_train.shape[1]}")

print("\n" + "=" * 70)
print("[Task 2] MLP: build train-only vocab/norm-stats + 5-seed training")
print("=" * 70)

cat_cols = [c for c in dlc.CAT_COLS_BASE if c in X_train.columns]
num_cols = [c for c in X_train.columns if c not in cat_cols]

num_raw = X_train[num_cols].astype(np.float32).values
mean = np.nanmean(num_raw, axis=0)
std = np.nanstd(num_raw, axis=0)
std[std < 1e-8] = 1.0
num_z = np.nan_to_num((num_raw - mean) / std, nan=0.0)

cat_cardinalities = []
cat_vocabs = {}
cat_cols_arr = []
for c in cat_cols:
    train_vals = X_train[c].astype(str)
    vocab = {v: i for i, v in enumerate(pd.unique(train_vals))}
    unk_idx = len(vocab)
    cat_vocabs[c] = vocab
    cat_cols_arr.append(train_vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
    cat_cardinalities.append(unk_idx + 1)

cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_train), 0), dtype=np.int64)
num_tr_t = torch.tensor(num_z, dtype=torch.float32)
cat_tr_t = torch.tensor(cat_arr, dtype=torch.int64)
y_tr_t = torch.tensor(y_train, dtype=torch.float32)
num_dim = num_tr_t.shape[1]

print(f"num_dim={num_dim}  cat_cardinalities={cat_cardinalities}")

mlp_shifts = {}
for seed in SEEDS:
    t1 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = dlc.SimpleMLP(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15)
    model, shift = dlc.train_generic(model, num_tr_t, cat_tr_t, y_tr_t, epochs=8, lr=1e-3,
                                      batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                      verbose_prefix=f"[mlp seed={seed}] ")
    torch.save(model.state_dict(), MODEL_DIR / f'mlp_model_seed{seed}.pt')
    mlp_shifts[seed] = shift
    print(f"  MLP seed={seed} trained in {time.time()-t1:.1f}s, calibration_shift={shift:+.4f}")

joblib.dump(dict(cat_cols=cat_cols, num_cols=num_cols, cat_vocabs=cat_vocabs,
                  cat_cardinalities=cat_cardinalities, mean=mean, std=std,
                  num_dim=num_dim, seeds=SEEDS, mlp_shifts=mlp_shifts,
                  count_x_base_map=cat_map_countbase),
            MODEL_DIR / 'mlp_artifacts.pkl')
print("MLP artifacts saved.")

t_train_duration = time.time() - t0_train
print(f"\nFull training completed in {t_train_duration:.1f}s")

# =========================================================================
print("\n" + "=" * 70)
print("[Task 3] Writing inference script.py (GBDT row-independence-fixed + MLP blend)")
print("=" * 70)

script_content = r"""import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn
from agent2_asof_decomp2 import AsofDecomposer2  # unpickle 시 클래스 정의 필요

t0 = time.time()
print("Starting DACON Submission Inference Pipeline (GBDT + asof_dec + SimpleMLP blend, 5-seed each)...")

DEVICE = torch.device('cpu')
SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006
W_MLP = 0.32  # frozen, inner-only selected + 5-seed outer(2024)-confirmed (report 205, gain=+8.56)


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
    def __init__(self, num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1):
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


data_dir = os.path.join(SCRIPT_DIR, "data")
if not os.path.exists(data_dir):
    data_dir = "data"
output_dir = os.path.join(SCRIPT_DIR, "output")
if not os.path.exists(output_dir):
    output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(SCRIPT_DIR, "model")

test_path = os.path.join(data_dir, "test.csv")
if not os.path.exists(test_path):
    test_path = "data/test.csv"

print(f"Loading test data from: {test_path}")
df_test = pd.read_csv(test_path)
print(f"Test data shape: {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

# --- base feature prep (matches PitchPreprocessor pipeline) ---
prep = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep.trackman_builder = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
X_test = prep.transform(df_test)

base_str = ((df_test['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_test['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_test['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_test['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_test['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

# --- asof_dec 분해 피처 (train.csv로만 fit된 고정 테이블을 이 test.csv 행들에 개별 조회만
#     수행 -- 이 배치의 다른 행이나 배치 분포를 전혀 참조하지 않음, 대회 규정4 준수) ---
dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))
A_test = dec.transform(df_test)
A_test.index = X_test.index
X_test = pd.concat([X_test, A_test], axis=1)

cat_cols = [c for c in X_test.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]

# =====================================================================
# GBDT branch
# =====================================================================
X_test_cb = X_test.copy()
for c in cat_cols:
    X_test_cb[c] = X_test_cb[c].astype(int).astype(str)
for c in [col for col in X_test_cb.columns if col not in cat_cols]:
    X_test_cb[c] = X_test_cb[c].astype(np.float32)

# FIX (report 203): fixed value-1 arithmetic transform instead of batch-dependent
# `.astype('category').cat.codes` -- each row's XGB category code depends only on
# its own already-encoded value, matching exactly what the model was trained on
# (train.csv's dense 1..N category ranges made the old per-fold `.cat.codes` on
# the FULL training set equal to this same value-1 transform).
X_test_xgb = X_test.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_test_xgb[c] = X_test_xgb[c].astype(np.float32)
    else:
        X_test_xgb[c] = (X_test_xgb[c] - 1).astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

print("Predicting with GBDT 3-model ensemble (5-seed bagged, asof_dec included)...")
p_lgb_sum = np.zeros(len(df_test))
p_cb_sum = np.zeros(len(df_test))
p_xgb_sum = np.zeros(len(df_test))
for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb_sum += m_lgb.predict(X_test)
    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb_sum += m_cb.predict_proba(X_test_cb)[:, 1]
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_sum += m_xgb.predict_proba(X_test_xgb)[:, 1]

n_seeds = len(SEEDS)
p_lgb = np.clip(p_lgb_sum / n_seeds + S_LGB, 1e-6, 1 - 1e-6)
p_cb = np.clip(p_cb_sum / n_seeds + S_CB, 1e-6, 1 - 1e-6)
p_xgb = np.clip(p_xgb_sum / n_seeds + S_XGB, 1e-6, 1 - 1e-6)
p_gbdt = np.clip(W_LGB * p_lgb + W_CB * p_cb + W_XGB * p_xgb, 1e-6, 1 - 1e-6)
print(f"GBDT ensemble done ({time.time()-t0:.1f}s elapsed)")

# =====================================================================
# MLP branch -- categorical encoding via a FIXED train-only vocab dict
# (built once at training time, saved in mlp_artifacts.pkl), applied to
# each row independently via .map() -- structurally row-independent,
# empirically re-verified in report 205 (batch vs single-row diff=5.96e-08).
# =====================================================================
art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
num_cols, cat_cols_mlp = art['num_cols'], art['cat_cols']
mean, std = art['mean'], art['std']
cat_vocabs = art['cat_vocabs']
cat_cardinalities = art['cat_cardinalities']
num_dim = art['num_dim']

num_raw = X_test[num_cols].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean) / std, nan=0.0)
num_t = torch.tensor(num_z, dtype=torch.float32)

cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_test[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_test), 0), dtype=np.int64)
cat_t = torch.tensor(cat_arr, dtype=torch.int64)

print("Predicting with MLP 5-seed bagged ensemble...")
p_mlp_sum = np.zeros(len(df_test))
for seed in SEEDS:
    model = SimpleMLP(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.15)
    state = torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt'), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(num_t, cat_t)
        p = torch.sigmoid(logits).numpy()
    shift = art['mlp_shifts'][seed]
    p_mlp_sum += np.clip(p + shift, 1e-6, 1 - 1e-6)
p_mlp = np.clip(p_mlp_sum / n_seeds, 1e-6, 1 - 1e-6)
print(f"MLP ensemble done ({time.time()-t0:.1f}s elapsed)")

# =====================================================================
# Blend (frozen w_mlp=0.32, inner-only selected + 5-seed outer-confirmed, report 205)
# =====================================================================
p_final = np.clip((1 - W_MLP) * p_gbdt + W_MLP * p_mlp, 1e-6, 1 - 1e-6)

sub = pd.DataFrame({
    'row_id': df_test['row_id'],
    'control_success': p_final
})
sub_path = os.path.join(output_dir, 'submission.csv')
sub.to_csv(sub_path, index=False)

elapsed = time.time() - t0
print(f"Inference completed & submission saved to {sub_path} in {elapsed:.2f} seconds!")
"""

with open(SUBMIT_DIR / 'script.py', 'w', encoding='utf-8') as f:
    f.write(script_content)
print("script.py written successfully.")

print("\n" + "=" * 70)
print("[Task 4] Writing requirements.txt (GBDT + torch; torch left unpinned -- server pre-installs it, avoids version conflict per 8차 practice)")
print("=" * 70)
req_content = """lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=1.7.0
"""
with open(SUBMIT_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)
print(req_content)

print("\n" + "=" * 70)
print("[Task 5] Copying local pipeline modules")
print("=" * 70)
modules_to_copy = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
for m_file in modules_to_copy:
    shutil.copy(BASE_DIR / m_file, SUBMIT_DIR / m_file)
    print(f"  Copied: {m_file}")
shutil.copy(BASE_DIR / 'scratch' / 'agent2_asof_decomp2.py', SUBMIT_DIR / 'agent2_asof_decomp2.py')
print("  Copied: agent2_asof_decomp2.py (AsofDecomposer2 클래스 정의, unpickle에 필요)")

print("\n" + "=" * 70)
print("[Task 6] Packaging submit_v14.zip")
print("=" * 70)
zip_path = BASE_DIR / 'work/submit_v14.zip'
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = sorted(zipf.namelist())
print(f"Created zip archive: {zip_path} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
print(f"Zip contents ({len(namelist)} files): {namelist}")

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py', 'agent2_asof_decomp2.py']
missing = [m for m in required_modules if m not in namelist]
print(f"Missing required local modules: {missing} (should be empty)")

with open('/tmp/submit_v14_build_result.json', 'w') as f:
    json.dump({
        "zip_path": str(zip_path), "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
        "train_duration_seconds": t_train_duration, "files": namelist, "missing_required_modules": missing,
        "seeds": SEEDS, "config": "GBDT(15/75/10, asof_dec) + SimpleMLP(asof_dec) blend w_mlp=0.32",
        "cv_note": "outer(2024) blend gain=+8.56 vs GBDT-alone (report 205), row-independence PASS (max_diff=5.96e-08)",
    }, f, indent=2)

print(f"\nBUILD COMPLETE. Total time: {(time.time()-t0_train)/60:.1f} min")
