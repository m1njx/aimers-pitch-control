import sys
import os
import shutil
import zipfile
import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import config
from preprocessing import PitchPreprocessor

import torch
from tabm_inference_model import TabM

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
import pickle

warnings.filterwarnings('ignore')

BASE_DIR = Path(os.path.expanduser('~/LG_data'))
OUTPUTS_DIR = BASE_DIR / 'outputs'
SUBMIT_DIR = BASE_DIR / 'work/submit_v9'
MODEL_DIR = SUBMIT_DIR / 'model'
DUMMY_DIR = BASE_DIR / 'work/dummy_eval_v9'

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006
W_GBDT, W_TABM = 0.52, 0.48  # 157번: GBDT(1-0.48) + TabM(0.48), nested-honest 888.43점

if SUBMIT_DIR.exists():
    shutil.rmtree(SUBMIT_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if DUMMY_DIR.exists():
    shutil.rmtree(DUMMY_DIR)
DUMMY_DIR.mkdir(parents=True, exist_ok=True)

NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("=" * 70)
print("[Task 0] Preprocessing (shared by GBDT + TabM) — rebuilt under Python 3.11.15 / numpy 1.26.4")
print("=" * 70)

t0_train = time.time()
df_train = pd.read_csv(config.TRAIN_PATH)
y_train = df_train[config.TARGET_COL].values
print(f"Loaded train dataset: {df_train.shape[0]:,} rows")
print(f"numpy={np.__version__} pandas={pd.__version__} torch={torch.__version__} python={sys.version.split()[0]}")

prep = PitchPreprocessor()
prep.fit(df_train, as_of_season=None, is_final=True)
X_train = prep.transform(df_train)

base_str = ((df_train['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df_train['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df_train['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df_train['strikes_before'].fillna(0).astype(int).astype(str))
X_train['count_x_base'] = (cc_str + '_' + base_str)
cat_map_countbase = {v: i for i, v in enumerate(X_train['count_x_base'].unique())}
X_train['count_x_base'] = X_train['count_x_base'].map(cat_map_countbase).fillna(-1).astype(int)
prep.count_x_base_map = cat_map_countbase

CAT_COLS = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL, 'count_x_base']
cat_cols = [c for c in CAT_COLS if c in X_train.columns]
num_cols = [c for c in X_train.columns if c not in cat_cols]
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

print("\n" + "=" * 70)
print("[Task 1] Full Re-training: GBDT 3종 (5-seed, full data) — 전부 새 환경에서 재학습")
print("=" * 70)

X_tr_cb = X_train.copy()
for c in cat_cols:
    X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
    X_tr_cb[c] = X_tr_cb[c].astype(np.float32)

X_tr_xgb = X_train.copy()
for c in cat_cols:
    X_tr_xgb[c] = X_tr_xgb[c].astype('category').cat.codes.astype(np.float32)
X_tr_xgb = X_tr_xgb.astype(np.float32)

for seed in SEEDS:
    print(f"\n--- [GBDT seed={seed}] Training LGBM / CatBoost / XGBoost ---")
    t1 = time.time()
    m_lgb = lgb.LGBMClassifier(n_estimators=250, num_leaves=45, learning_rate=0.05,
                                min_child_samples=20, colsample_bytree=0.7, subsample=0.7,
                                random_state=seed, verbosity=-1, n_jobs=-1)
    m_lgb.fit(X_train, y_train, categorical_feature=cat_idx)
    m_lgb.booster_.save_model(str(MODEL_DIR / f'lgbm_model_seed{seed}.txt'))
    print(f"  LightGBM seed={seed} done in {time.time()-t1:.1f}s")

    t2 = time.time()
    m_cb = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=10.0,
                               random_seed=seed, verbose=0, cat_features=cat_cols, thread_count=-1)
    m_cb.fit(X_tr_cb, y_train)
    m_cb.save_model(str(MODEL_DIR / f'catboost_model_seed{seed}.cbm'))
    print(f"  CatBoost seed={seed} done in {time.time()-t2:.1f}s")

    t3 = time.time()
    m_xgb = xgb.XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                               colsample_bytree=0.8, subsample=0.8, random_state=seed,
                               n_jobs=-1, eval_metric='logloss')
    m_xgb.fit(X_tr_xgb, y_train)
    m_xgb.save_model(str(MODEL_DIR / f'xgb_model_seed{seed}.json'))
    print(f"  XGBoost seed={seed} done in {time.time()-t3:.1f}s")

joblib.dump(prep, MODEL_DIR / 'preprocessor_artifacts.pkl')
joblib.dump(prep.trackman_builder, MODEL_DIR / 'trackman_artifacts.pkl')
print("\nGBDT training complete. Preprocessor & Trackman artifacts saved (numpy 1.26.4-native pickle).")

print("\n" + "=" * 70)
print("[Task 2] Full Re-training: TabM (5-seed, full data, count_x_base included)")
print("=" * 70)

# --- Fit DL preprocessing artifacts on FULL data ---
num_arr = X_train[num_cols].astype(np.float32).values
num_mean = np.nanmean(num_arr, axis=0)
num_std = np.nanstd(num_arr, axis=0)
num_std[num_std < 1e-8] = 1.0
num_z = np.nan_to_num((num_arr - num_mean) / num_std, nan=0.0)

cat_vocabs = []
cat_arrs = []
cat_cardinalities = []
for c in cat_cols:
    vals = X_train[c].astype(str)
    vocab = {v: i for i, v in enumerate(pd.unique(vals))}
    unk_idx = len(vocab)
    cat_vocabs.append(vocab)
    cat_arrs.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
    cat_cardinalities.append(unk_idx + 1)
cat_arr = np.stack(cat_arrs, axis=1) if cat_arrs else np.zeros((len(X_train), 0), dtype=np.int64)

print(f"DL feature matrix: num={num_z.shape}, cat={cat_arr.shape}, cardinalities={cat_cardinalities}")

# Save DL preprocessing artifacts for script.py to reuse at inference time
dl_artifacts = dict(
    num_cols=num_cols, cat_cols=cat_cols, num_mean=num_mean, num_std=num_std,
    cat_vocabs=cat_vocabs, cat_cardinalities=cat_cardinalities,
    count_x_base_map=cat_map_countbase,
)
import pickle
with open(MODEL_DIR / 'dl_preprocessing_artifacts.pkl', 'wb') as f:
    pickle.dump(dl_artifacts, f)
print("DL preprocessing artifacts saved.")

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Training device: {DEVICE}")

num_tr_t = torch.tensor(num_z, dtype=torch.float32)
cat_tr_t = torch.tensor(cat_arr, dtype=torch.int64)
y_tr_t = torch.tensor(y_train, dtype=torch.float32)
num_dim = num_tr_t.shape[1]

# small dev split for early stopping + shift search (same convention as dl_common.train_generic)
rng = np.random.RandomState(1)
perm = rng.permutation(len(y_train))
n_dev = int(len(y_train) * 0.05)
dev_idx, tr_idx = perm[:n_dev], perm[n_dev:]

EPOCHS = 10
tabm_shifts = {}
for seed in SEEDS:
    print(f"\n--- [TabM seed={seed}] Training on full data ---")
    t1 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TabM(num_dim, cat_cardinalities, seed=seed).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_tr = len(tr_idx)
    batch_size = 4096
    best_dev_loss = float('inf')
    best_state = None
    patience, bad_epochs = 2, 0
    num_dev_t, cat_dev_t = num_tr_t[dev_idx], cat_tr_t[dev_idx]
    y_dev_np = y_tr_t[dev_idx].numpy()

    for epoch in range(EPOCHS):
        model.train()
        ep_perm = np.random.permutation(n_tr)
        total_loss = 0.0
        for i in range(0, n_tr, batch_size):
            idx = tr_idx[ep_perm[i:i + batch_size]]
            xb_num = num_tr_t[idx].to(DEVICE)
            xb_cat = cat_tr_t[idx].to(DEVICE)
            yb = y_tr_t[idx].to(DEVICE)
            opt.zero_grad()
            logits = model(xb_num, xb_cat)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_tr

        model.eval()
        with torch.no_grad():
            dev_logits = model(num_dev_t.to(DEVICE), cat_dev_t.to(DEVICE)).cpu()
            dev_loss = loss_fn(dev_logits, torch.tensor(y_dev_np)).item()
        print(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.5f} dev_loss={dev_loss:.5f}")

        if dev_loss < best_dev_loss - 1e-5:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p_dev = torch.sigmoid(model(num_dev_t.to(DEVICE), cat_dev_t.to(DEVICE))).cpu().numpy()
    best_s, best_b = 0.0, float(np.mean((np.clip(p_dev, 1e-6, 1 - 1e-6) - y_dev_np) ** 2))
    for s in np.linspace(-0.05, 0.05, 41):
        p_shift = np.clip(p_dev + s, 1e-6, 1 - 1e-6)
        b = float(np.mean((p_shift - y_dev_np) ** 2))
        if b < best_b:
            best_b, best_s = b, s
    tabm_shifts[seed] = float(best_s)
    print(f"  Best post-hoc shift: {best_s:+.4f} (dev brier {best_b:.6f})")

    torch.save(model.state_dict(), MODEL_DIR / f'tabm_model_seed{seed}.pt')
    print(f"  Saved: tabm_model_seed{seed}.pt ({time.time()-t1:.1f}s)")

with open(MODEL_DIR / 'tabm_shifts.json', 'w') as f:
    json.dump(tabm_shifts, f)
print(f"\nTabM shifts: {tabm_shifts}")

t_train_duration = time.time() - t0_train
print(f"\nFull TabM training completed in {t_train_duration:.1f}s")

# =========================================================================
# WORK: Write script.py (GBDT 5-seed + TabM 5-seed blend)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 3] Writing inference script.py")
print("=" * 70)

script_content = r"""import sys
import os
import time
import json
import pickle
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
from tabm_inference_model import TabM

t0 = time.time()
print("Starting DACON 8th Submission Inference Pipeline (GBDT 5-seed + TabM 5-seed blend)...")

SEEDS = [7, 123, 2025, 31415, 8675309]
W_LGB, W_CB, W_XGB = 0.15, 0.75, 0.10
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006
W_GBDT, W_TABM = 0.52, 0.48

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

# --- GBDT feature prep (matches PitchPreprocessor pipeline) ---
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

cat_cols = [c for c in X_test.columns if c in ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']]
cat_idx = [X_test.columns.get_loc(c) for c in cat_cols if c in X_test.columns]

X_test_cb = X_test.copy()
for c in cat_cols:
    X_test_cb[c] = X_test_cb[c].astype(int).astype(str)
for c in [col for col in X_test_cb.columns if col not in cat_cols]:
    X_test_cb[c] = X_test_cb[c].astype(np.float32)

X_test_xgb = X_test.copy()
for c in cat_cols:
    X_test_xgb[c] = X_test_xgb[c].astype('category').cat.codes.astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

print("Predicting with GBDT 3-model ensemble (5-seed bagged)...")
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

# --- TabM feature prep (uses saved DL preprocessing artifacts) ---
print("Predicting with TabM (5-seed bagged)...")
with open(os.path.join(model_dir, 'dl_preprocessing_artifacts.pkl'), 'rb') as f:
    dl_art = pickle.load(f)

num_cols, dl_cat_cols = dl_art['num_cols'], dl_art['cat_cols']
num_mean, num_std = dl_art['num_mean'], dl_art['num_std']
cat_vocabs, cat_cardinalities = dl_art['cat_vocabs'], dl_art['cat_cardinalities']

# X_test already has count_x_base (raw category string form was overwritten above with GBDT's
# integer cat_map; DL uses its own independently-fitted vocab, so rebuild count_x_base string
# form fresh for the DL path to avoid cross-contamination with the GBDT integer encoding)
X_test_dl = prep.transform(df_test)
count_x_base_dl_map = dl_art['count_x_base_map']
X_test_dl['count_x_base'] = count_x_base_raw.map(count_x_base_dl_map).fillna(-1).astype(int)

num_arr = X_test_dl[num_cols].astype(np.float32).values
num_z = np.nan_to_num((num_arr - num_mean) / num_std, nan=0.0)

cat_arrs = []
for c, vocab in zip(dl_cat_cols, cat_vocabs):
    vals = X_test_dl[c].astype(str)
    unk_idx = len(vocab)
    cat_arrs.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_arrs, axis=1) if cat_arrs else np.zeros((len(X_test_dl), 0), dtype=np.int64)

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
num_t = torch.tensor(num_z, dtype=torch.float32)
cat_t = torch.tensor(cat_arr, dtype=torch.int64)

with open(os.path.join(model_dir, 'tabm_shifts.json'), 'r') as f:
    tabm_shifts = json.load(f)

p_tabm_sum = np.zeros(len(df_test))
num_dim = num_t.shape[1]
for seed in SEEDS:
    model = TabM(num_dim, cat_cardinalities, seed=seed)
    state = torch.load(os.path.join(model_dir, f'tabm_model_seed{seed}.pt'), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(num_t.to(DEVICE), cat_t.to(DEVICE))
        p = torch.sigmoid(logits).cpu().numpy()
    shift = tabm_shifts.get(str(seed), tabm_shifts.get(seed, 0.0))
    p_tabm_sum += np.clip(p + shift, 1e-6, 1 - 1e-6)

p_tabm = np.clip(p_tabm_sum / n_seeds, 1e-6, 1 - 1e-6)
print(f"TabM ensemble done ({time.time()-t0:.1f}s elapsed)")

# --- Final blend ---
p_final = np.clip(W_GBDT * p_gbdt + W_TABM * p_tabm, 1e-6, 1 - 1e-6)

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

# =========================================================================
# WORK: requirements.txt (torch deliberately omitted — pre-installed on server)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 4] Writing requirements.txt (torch omitted — server has 2.7.1+cu128 preinstalled)")
print("=" * 70)
req_content = """lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=1.7.0
"""
with open(SUBMIT_DIR / 'requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)
print(req_content)

# =========================================================================
# WORK: Copy local modules (GBDT deps + TabM inference model)
# =========================================================================
print("\n" + "=" * 70)
print("[Task 5] Copying local pipeline modules")
print("=" * 70)
modules_to_copy = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py']
for m_file in modules_to_copy:
    shutil.copy(BASE_DIR / m_file, SUBMIT_DIR / m_file)
    print(f"  Copied: {m_file}")
shutil.copy(BASE_DIR / 'scratch' / 'tabm_inference_model.py', SUBMIT_DIR / 'tabm_inference_model.py')
print("  Copied: tabm_inference_model.py (self-contained TabM class)")

# =========================================================================
# WORK: Package submit_v9.zip
# =========================================================================
print("\n" + "=" * 70)
print("[Task 6] Packaging submit_v9.zip")
print("=" * 70)
zip_path = BASE_DIR / 'work/submit_v9.zip'
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMIT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(SUBMIT_DIR)
            zipf.write(file_path, arcname)

with zipfile.ZipFile(zip_path, 'r') as zipf:
    namelist = sorted(zipf.namelist())
print(f"Created zip archive: {zip_path} (Size: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
print(f"Zip contents ({len(namelist)} files): {namelist}")

required_modules = ['preprocessing.py', 'trackman_features.py', 'config.py', 'cv_utils.py', 'tabm_inference_model.py']
missing = [m for m in required_modules if m not in namelist]
print(f"Missing required local modules: {missing} (should be empty)")

with open('/tmp/submit_v9_build_result.json', 'w') as f:
    json.dump({
        "zip_path": str(zip_path), "zip_size_mb": zip_path.stat().st_size / (1024 * 1024),
        "train_duration_seconds": t_train_duration, "files": namelist, "missing_required_modules": missing,
        "seeds": SEEDS, "weights": {"gbdt": W_GBDT, "tabm": W_TABM},
    }, f, indent=2)

print(f"\nBUILD COMPLETE. Total time: {(time.time()-t0_train)/60:.1f} min")
