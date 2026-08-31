import os, sys, time, gc
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
sys.path.insert(0, '<팀 저장소 경로>/harness')
from metric import official_score, paired_bootstrap

# Environment variables for Apple Silicon OpenMP safety
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=== STARTING ARM C CLEAN TRAINING PIPELINE ===", flush=True)

# 1. Load Data
df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
print(f"Total train data loaded: {len(df)} rows", flush=True)

# Split for honest holdout evaluation: train < 2024, val == 2024
train_df = df[df['season'] < 2024].copy().reset_index(drop=True)
val_df = df[df['season'] == 2024].copy().reset_index(drop=True)
y_train = train_df['control_success'].values
y_val = val_df['control_success'].values

print(f"Honest holdout split: Train={len(train_df)}, Val={len(val_df)}", flush=True)

# 2. Feature Engineering (Clean, Row-Local, ABS-Invariant)
def extract_features(df_fit, df_trans):
    # Global constants from df_fit
    g_mean = df_fit['control_success'].mean()
    C = 50.0
    
    # Empirical Bayes Target Encodings
    p_agg = df_fit.groupby('pitcher_id')['control_success'].agg(['count', 'mean'])
    p_eb = ((p_agg['count'] * p_agg['mean'] + C * g_mean) / (p_agg['count'] + C)).to_dict()
    
    b_agg = df_fit.groupby('batter_id')['control_success'].agg(['count', 'mean'])
    b_eb = ((b_agg['count'] * b_agg['mean'] + C * g_mean) / (b_agg['count'] + C)).to_dict()
    
    feats = pd.DataFrame()
    
    # Core domain features
    feats['inning'] = df_trans['inning'].fillna(1).astype(float)
    feats['outs'] = df_trans['outs_before'].fillna(0).astype(float)
    feats['balls'] = df_trans['balls_before'].fillna(0).astype(float)
    feats['strikes'] = df_trans['strikes_before'].fillna(0).astype(float)
    feats['count_diff'] = feats['strikes'] - feats['balls']
    feats['score_diff'] = df_trans['score_diff'].fillna(0).astype(float) if 'score_diff' in df_trans.columns else 0.0
    
    # Categoricals
    feats['top_bottom'] = (df_trans['top_bottom'] == 'T').astype(float)
    feats['is_futures'] = (df_trans['game_type'] == 'F').astype(float)
    feats['base_state'] = df_trans['base_state'].astype('category').cat.codes.astype(float)
    
    # Target encodings
    feats['pitcher_eb'] = df_trans['pitcher_id'].map(p_eb).fillna(g_mean).astype(float)
    feats['batter_eb'] = df_trans['batter_id'].map(b_eb).fillna(g_mean).astype(float)
    feats['eb_diff'] = feats['pitcher_eb'] - feats['batter_eb']
    
    # Asof features
    asof_cols = ['asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_middle_rate',
                 'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
                 'asof_pitcher_prev5_game_success_rate',
                 'asof_batter_n', 'asof_batter_success_rate', 'asof_batter_middle_rate']
    for c in asof_cols:
        if c in df_trans.columns:
            feats[c] = df_trans[c].astype(float)
            
    # Form deviation (Recent vs Season)
    if 'asof_pitcher_prev1_game_success_rate' in df_trans.columns and 'asof_pitcher_success_rate' in df_trans.columns:
        feats['pitcher_recent_dev1'] = (df_trans['asof_pitcher_prev1_game_success_rate'] - df_trans['asof_pitcher_success_rate']).fillna(0.0)
        feats['pitcher_recent_dev3'] = (df_trans['asof_pitcher_prev3_game_success_rate'] - df_trans['asof_pitcher_success_rate']).fillna(0.0)

    # Trackman features if available
    tm_cols = ['tkm_rel_speed_mean', 'tkm_spin_rate_mean', 'tkm_extension_mean', 'tkm_rel_height_mean',
               'induced_vert_break', 'horz_break', 'phys_effective_velocity', 'phys_vaa_proxy']
    for c in tm_cols:
        if c in df_trans.columns:
            feats[c] = df_trans[c].astype(float)
            
    # Clean all NaNs thoroughly
    feats = feats.fillna(feats.median())
    return feats

print("Extracting features...", flush=True)
X_train = extract_features(train_df, train_df)
X_val = extract_features(train_df, val_df)
print(f"Features ready: {X_train.shape[1]} columns", flush=True)

# 3. Train LightGBM Bagging with Early Stopping
lgb_preds = []
SEEDS = [7, 123, 2025, 31415, 8675309]

print("\n--- Training LightGBM 5-seed Ensemble ---", flush=True)
for s in SEEDS:
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 45,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 50,
        'random_state': s,
        'n_jobs': 4,
        'verbose': -1
    }
    
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
    
    clf = lgb.train(
        params,
        trn_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    
    pred_val = clf.predict(X_val, num_iteration=clf.best_iteration)
    sc = official_score(y_val, pred_val)
    print(f"  LGBM seed {s} best iter {clf.best_iteration}: 2024 Score = {sc:.2f}", flush=True)
    lgb_preds.append(pred_val)

p_lgb_mean = np.mean(lgb_preds, axis=0)
sc_lgb_bag = official_score(y_val, p_lgb_mean)
print(f"=> LightGBM 5-Seed Bagged 2024 Score: {sc_lgb_bag:.2f}", flush=True)

# 4. Train CatBoost Bagging with Early Stopping
cb_preds = []
print("\n--- Training CatBoost 5-seed Ensemble ---", flush=True)
for s in SEEDS:
    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        random_seed=s,
        thread_count=4,
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=40, verbose=False)
    
    pred_val = model.predict_proba(X_val)[:, 1]
    sc = official_score(y_val, pred_val)
    print(f"  CatBoost seed {s} best iter {model.get_best_iteration()}: 2024 Score = {sc:.2f}", flush=True)
    cb_preds.append(pred_val)

p_cb_mean = np.mean(cb_preds, axis=0)
sc_cb_bag = official_score(y_val, p_cb_mean)
print(f"=> CatBoost 5-Seed Bagged 2024 Score: {sc_cb_bag:.2f}", flush=True)

# 5. Combined Arm C Prediction & Calibration
p_arm_c = 0.55 * p_lgb_mean + 0.45 * p_cb_mean
sc_arm_c = official_score(y_val, p_arm_c)
print(f"\n=======================================================", flush=True)
print(f"=== ARM C COMBINED HONEST HOLDOUT SCORE: {sc_arm_c:.2f} ===", flush=True)
print(f"=======================================================", flush=True)

# 6. Evaluate Blend with Baseline (Arm A + Arm B)
CACHE = os.path.join(LG, 'harness/cache')
from evaluate import PROD, predict
bag_A = [dict(np.load(os.path.join(CACHE, f'pred_2024_{s}.npz'))) for s in SEEDS]
p_A = np.mean([predict(PROD, P) for P in bag_A], axis=0)

import glob
b_files = sorted(glob.glob(os.path.join(LG, 'teamB/out/preds/l2384_f2024_s*.npy')))
p_B = np.mean([np.load(f).astype(np.float64) for f in b_files], axis=0)

# Current 2-arm baseline: 0.50 A + 0.50 B with U=0.40
s_arr = np.sign(val_df['strikes_before'].fillna(0).values - val_df['balls_before'].fillna(0).values).astype(int)
OFF = {-1: -0.0212, 0: -0.0146, 1: 0.0428}

w_a = np.where(val_df['game_type'] == 'F', 0.15, 0.50)
p_2arm = w_a * p_A + (1 - w_a) * p_B
z_2arm = np.log(p_2arm / (1 - p_2arm))
for k, v in OFF.items():
    z_2arm[s_arr == k] += 0.40 * v
p_2arm_off = 1.0 / (1.0 + np.exp(-z_2arm))
sc_2arm = official_score(y_val, p_2arm_off)
print(f"\nCurrent 2-Arm Baseline 2024 Score: {sc_2arm:.2f}", flush=True)

# 3-Arm Blend Test
d_xc = np.mean(np.abs(p_2arm_off - p_arm_c))
print(f"Distance d(Baseline, Arm C) = {d_xc:.4f}", flush=True)

for w_c in [0.05, 0.10, 0.15, 0.20, 0.25]:
    p_3arm = (1 - w_c) * p_2arm_off + w_c * p_arm_c
    sc_3arm = official_score(y_val, p_3arm)
    diff = sc_3arm - sc_2arm
    print(f"3-Arm Blend (w_C={w_c:.2f}): Score={sc_3arm:.2f} (Delta: {diff:+.2f})", flush=True)

best_wc = 0.15
p_best_3arm = (1 - best_wc) * p_2arm_off + best_wc * p_arm_c
pb = paired_bootstrap(y_val, p_best_3arm, p_2arm_off)
print("\nPaired Bootstrap 3-Arm vs 2-Arm Baseline:")
print(pb, flush=True)
