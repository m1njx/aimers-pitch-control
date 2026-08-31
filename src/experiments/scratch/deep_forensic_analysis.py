"""
v50 ~ v53 점수 하락 원인 정밀 분석 & 1100점 돌파 가능성 탐색
- 5행 테스트셋에서 Brier Skill Score가 어떻게 결정되는지
- 캘리브레이션 파라미터의 실질적 영향도
- v50 holdout에서 최적 파라미터 그리드 서치
"""
import os, sys, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
model_dir = os.path.join(BASE_DIR, 'work', 'submit_v50', 'model')

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

# Load full training data
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
is_val24 = (df['season'] == 2024).values
is_val23 = (df['season'] == 2023).values
is_train_v50 = (df['season'] < 2024).values  # v50 trained on <2024

y = df['control_success'].values.astype(np.float32)

# Reproduce v50 feature pipeline exactly
sys.path.insert(0, os.path.join(BASE_DIR, 'work', 'submit_v50'))
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

tkm_art = joblib.load(os.path.join(model_dir, 'trackman_artifacts.pkl'))
tkm_builder = TrackmanFeatureBuilder()
tkm_builder.artifacts = tkm_art if isinstance(tkm_art, dict) else tkm_art.artifacts
tkm_builder.is_fitted = True

prep_art = joblib.load(os.path.join(model_dir, 'preprocessor_artifacts.pkl'))
prep = PitchPreprocessor()
prep.artifacts = prep_art if isinstance(prep_art, dict) else prep_art.artifacts
prep.trackman_builder = tkm_builder
prep.is_fitted = True

dec = joblib.load(os.path.join(model_dir, 'asof_decomposer_artifacts.pkl'))

X_base = prep.transform(df)
base_str = ((df['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
            (df['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
cc_str = (df['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          df['strikes_before'].fillna(0).astype(int).astype(str))
count_x_base_raw = (cc_str + '_' + base_str)
cat_map = getattr(prep, 'count_x_base_map', {})
X_base['count_x_base'] = count_x_base_raw.map(cat_map).fillna(-1).astype(int)

v0 = X_base['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
ext = X_base['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
rel_side = X_base['tkm_rel_side_mean']
rel_height = X_base['tkm_rel_height_mean']
ivb = X_base['tkm_induced_vert_break_mean'] / 12.0
hb = X_base['tkm_horz_break_mean'] / 12.0

t_flight = (60.5 - ext) / v0
t_tunnel = (t_flight - 0.15).clip(lower=0.01)
r_ratio = t_tunnel / t_flight
d_tunnel = np.sqrt((rel_side + hb * r_ratio)**2 + (rel_height + ivb * r_ratio)**2)
d_plate = np.sqrt((rel_side + hb)**2 + (rel_height + ivb)**2)

X_base['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
X_base['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
X_base['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

A_all = dec.transform(df)
A_all.index = X_base.index
X_base = pd.concat([X_base, A_all], axis=1)

v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
dist_to_plate = (60.5 - ext).clip(lower=50.0)

b = df['balls_before'].fillna(0).values
s = df['strikes_before'].fillna(0).values
li = df['li'].fillna(1.0).values
r2 = (df['runner_on_2b'].fillna(0) > 0).astype(float).values
r3 = (df['runner_on_3b'].fillna(0) > 0).astype(float).values
score_diff = df['score_diff_pitcher_team'].fillna(0).values
inning = df['inning'].fillna(1).values
fb_rate = df['asof_pitcher_fastball_rate'].fillna(0.5).values
br_rate = df['asof_pitcher_breaking_rate'].fillna(0.3).values
off_rate = df['asof_pitcher_offspeed_rate'].fillna(0.2).values
platoon_code = (df['pitcher_hand'].astype(str) == df['batter_hand'].astype(str)).astype(float).values

X_133 = X_base.copy()
X_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
X_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
X_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)
X_133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
X_133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
X_133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
X_133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
X_133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(score_diff**2 / 10.0, 0, 5.0))).astype(np.float32)
X_133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
X_133['feat_platoon_fastball_inter'] = (platoon_code * fb_rate).astype(np.float32)
X_133['feat_platoon_breaking_inter'] = (platoon_code * br_rate).astype(np.float32)
X_133['feat_platoon_offspeed_inter'] = (platoon_code * off_rate).astype(np.float32)
X_133['feat_late_inning_clutch'] = ((inning >= 7).astype(float) * li).astype(np.float32)

cat_cols = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id', 'count_code', 'platoon_matchup', 'tkm_match', 'count_x_base']

SEEDS = [7, 123, 2025, 31415, 8675309]

# ====== GBDT predictions on full data (including val24) ======
print("Computing v50-exact GBDT predictions on full dataset...")
p_lgb = np.zeros(len(df), dtype=np.float64)
p_cb = np.zeros(len(df), dtype=np.float64)
p_xgb_arr = np.zeros(len(df), dtype=np.float64)
p_lgb_mse = np.zeros(len(df), dtype=np.float64)

X_cb = X_base.copy()
for c in cat_cols:
    X_cb[c] = pd.to_numeric(X_cb[c], errors='coerce').fillna(-1).astype(int).astype(str)
for c in [col for col in X_cb.columns if col not in cat_cols]:
    X_cb[c] = pd.to_numeric(X_cb[c], errors='coerce').fillna(0.0).astype(np.float32)

X_xgb = X_base.copy()
for c in cat_cols:
    if c == 'count_x_base':
        X_xgb[c] = X_xgb[c].astype(np.float32)
    else:
        X_xgb[c] = (X_xgb[c] - 1).astype(np.float32)
X_xgb = X_xgb.astype(np.float32)

X_133_mat = X_133.values.astype(np.float32)

for seed in SEEDS:
    m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
    p_lgb += m_lgb.predict(X_base) / len(SEEDS)
    m_cb = CatBoostClassifier()
    m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
    p_cb += m_cb.predict_proba(X_cb)[:, 1] / len(SEEDS)
    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
    p_xgb_arr += m_xgb.predict_proba(X_xgb)[:, 1] / len(SEEDS)
    m_mse = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_mse_model_seed{seed}.txt'))
    p_lgb_mse += m_mse.predict(X_133_mat) / len(SEEDS)

W_LGB_BIN, W_CB_BIN, W_XGB_BIN = 0.20, 0.72, 0.08
S_LGB, S_CB, S_XGB = -0.007, -0.008, -0.006

p_lgb_bin = np.clip(p_lgb + S_LGB, 1e-6, 1 - 1e-6)
p_cb_bin = np.clip(p_cb + S_CB, 1e-6, 1 - 1e-6)
p_xgb_bin = np.clip(p_xgb_arr + S_XGB, 1e-6, 1 - 1e-6)
p_gbdt_bin = np.clip(W_LGB_BIN * p_lgb_bin + W_CB_BIN * p_cb_bin + W_XGB_BIN * p_xgb_bin, 1e-6, 1 - 1e-6)
p_gbdt_mse = np.clip(p_lgb_mse, 1e-6, 1 - 1e-6)

# ====== SimpleMLP predictions ======
print("Computing SimpleMLP predictions...")

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

class SimpleMLP_MSE(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

mlp_art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
num_cols_mlp, cat_cols_mlp = mlp_art['num_cols'], mlp_art['cat_cols']
mean_mlp, std_mlp = mlp_art['mean'], mlp_art['std']
cat_vocabs = mlp_art['cat_vocabs']
cat_cardinalities = mlp_art['cat_cardinalities']
num_dim = mlp_art['num_dim']

num_raw = X_133[num_cols_mlp].astype(np.float32).values
num_z = np.nan_to_num((num_raw - mean_mlp) / std_mlp, nan=0.0)
num_t = torch.tensor(num_z, dtype=torch.float32)

cat_cols_arr = []
for c in cat_cols_mlp:
    vocab = cat_vocabs[c]
    unk_idx = len(vocab)
    vals = X_133[c].astype(str)
    cat_cols_arr.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
cat_arr = np.stack(cat_cols_arr, axis=1) if cat_cols_arr else np.zeros((len(X_133), 0), dtype=np.int64)
cat_t = torch.tensor(cat_arr, dtype=torch.long)

p_mlp_sum = np.zeros(len(df), dtype=np.float64)
for seed in SEEDS:
    mlp_net = SimpleMLP_MSE(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12)
    mlp_net.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt'), map_location='cpu'))
    mlp_net.eval()
    with torch.no_grad():
        # Process in chunks to avoid memory issues
        bs = 50000
        preds = []
        for i in range(0, len(num_t), bs):
            preds.append(mlp_net(num_t[i:i+bs], cat_t[i:i+bs]).numpy())
        p_mlp_sum += np.concatenate(preds)
p_mlp = p_mlp_sum / len(SEEDS)

# ====== v50 raw blend ======
W_GBDT_BIN_V50 = 0.25
W_MLP_MSE_V50 = 0.50
W_LGB_MSE_V50 = 0.25
p_raw = W_GBDT_BIN_V50 * p_gbdt_bin + W_MLP_MSE_V50 * p_mlp + W_LGB_MSE_V50 * p_gbdt_mse

# Apply count shifts (v50)
count_shifts = joblib.load(os.path.join(model_dir, 'count_shifts_artifact.pkl'))
counts_arr = (df['balls_before'].fillna(0).astype(int).astype(str) + '_' + df['strikes_before'].fillna(0).astype(int).astype(str)).values
p_cond = p_raw.copy()
for cc, s_val in count_shifts.items():
    p_cond[counts_arr == cc] += s_val

print("\n" + "="*80)
print("PART 1: v50 Holdout BSS with current calibration (SCALE=1.10, SHIFT=-0.0035)")
print("="*80)

p_v50_cal = np.clip(0.5 + 1.10 * (p_cond - 0.5) + (-0.0035), 1e-6, 1 - 1e-6)

bss_24_v50 = brier_skill(y[is_val24], p_v50_cal[is_val24])
bss_23_v50 = brier_skill(y[is_val23], p_v50_cal[is_val23])
print(f"v50 BSS on 2024 holdout: {bss_24_v50:.2f}")
print(f"v50 BSS on 2023 holdout: {bss_23_v50:.2f}")
print(f"v50 raw blend mean (before cal): {p_raw[is_val24].mean():.6f}")
print(f"v50 calibrated mean (2024): {p_v50_cal[is_val24].mean():.6f}")
print(f"Actual label mean (2024): {y[is_val24].mean():.6f}")

# ====== PART 2: Massive grid search over calibration parameters ======
print("\n" + "="*80)
print("PART 2: Calibration Grid Search (SCALE x SHIFT)")
print("="*80)

best_bss_24 = -99999
best_params_24 = None
best_bss_23 = -99999
best_params_23 = None
best_bss_avg = -99999
best_params_avg = None

results = []

for scale in np.arange(0.90, 1.30, 0.01):
    for shift in np.arange(-0.020, 0.020, 0.001):
        p_cal = np.clip(0.5 + scale * (p_cond - 0.5) + shift, 1e-6, 1 - 1e-6)
        bss_24 = brier_skill(y[is_val24], p_cal[is_val24])
        bss_23 = brier_skill(y[is_val23], p_cal[is_val23])
        bss_avg = (bss_24 + bss_23) / 2.0
        results.append((scale, shift, bss_24, bss_23, bss_avg))
        if bss_24 > best_bss_24:
            best_bss_24 = bss_24
            best_params_24 = (scale, shift)
        if bss_23 > best_bss_23:
            best_bss_23 = bss_23
            best_params_23 = (scale, shift)
        if bss_avg > best_bss_avg:
            best_bss_avg = bss_avg
            best_params_avg = (scale, shift)

print(f"Best for 2024: SCALE={best_params_24[0]:.2f}, SHIFT={best_params_24[1]:.3f} → BSS={best_bss_24:.2f}")
print(f"Best for 2023: SCALE={best_params_23[0]:.2f}, SHIFT={best_params_23[1]:.3f} → BSS={best_bss_23:.2f}")
print(f"Best AVG(23+24): SCALE={best_params_avg[0]:.2f}, SHIFT={best_params_avg[1]:.3f} → BSS_avg={best_bss_avg:.2f}")

# Show v50 current vs best
print(f"\nv50 current (1.10, -0.0035): BSS_24={bss_24_v50:.2f}")
print(f"Best 2024 improvement: +{best_bss_24 - bss_24_v50:.2f} points")

# ====== PART 3: Grid search WITHOUT count shifts ======
print("\n" + "="*80)
print("PART 3: Calibration Grid Search WITHOUT count shifts")
print("="*80)

best_bss_24_nc = -99999
best_params_24_nc = None
best_bss_avg_nc = -99999
best_params_avg_nc = None

for scale in np.arange(0.90, 1.30, 0.01):
    for shift in np.arange(-0.020, 0.020, 0.001):
        p_cal_nc = np.clip(0.5 + scale * (p_raw - 0.5) + shift, 1e-6, 1 - 1e-6)
        bss_24_nc = brier_skill(y[is_val24], p_cal_nc[is_val24])
        bss_23_nc = brier_skill(y[is_val23], p_cal_nc[is_val23])
        bss_avg_nc = (bss_24_nc + bss_23_nc) / 2.0
        if bss_24_nc > best_bss_24_nc:
            best_bss_24_nc = bss_24_nc
            best_params_24_nc = (scale, shift)
        if bss_avg_nc > best_bss_avg_nc:
            best_bss_avg_nc = bss_avg_nc
            best_params_avg_nc = (scale, shift)

print(f"Best for 2024 (no count shifts): SCALE={best_params_24_nc[0]:.2f}, SHIFT={best_params_24_nc[1]:.3f} → BSS={best_bss_24_nc:.2f}")
print(f"Best AVG (no count shifts): SCALE={best_params_avg_nc[0]:.2f}, SHIFT={best_params_avg_nc[1]:.3f} → BSS_avg={best_bss_avg_nc:.2f}")

# ====== PART 4: Blend weight grid search ======
print("\n" + "="*80)
print("PART 4: Blend Weight Grid Search (GBDT vs MLP vs MSE)")
print("="*80)

best_bss_blend = -99999
best_blend_params = None

for w_gbdt in np.arange(0.10, 0.60, 0.05):
    for w_mlp in np.arange(0.10, 0.70, 0.05):
        w_mse = 1.0 - w_gbdt - w_mlp
        if w_mse < 0.05 or w_mse > 0.60:
            continue
        p_blend = w_gbdt * p_gbdt_bin + w_mlp * p_mlp + w_mse * p_gbdt_mse
        # Apply best calibration from Part 2
        sc, sh = best_params_avg
        p_cal_b = np.clip(0.5 + sc * (p_blend - 0.5) + sh, 1e-6, 1 - 1e-6)
        bss_24_b = brier_skill(y[is_val24], p_cal_b[is_val24])
        bss_23_b = brier_skill(y[is_val23], p_cal_b[is_val23])
        bss_avg_b = (bss_24_b + bss_23_b) / 2.0
        if bss_avg_b > best_bss_blend:
            best_bss_blend = bss_avg_b
            best_blend_params = (w_gbdt, w_mlp, w_mse, sc, sh, bss_24_b, bss_23_b)

w_g, w_m, w_ms, sc_b, sh_b, bss24_b, bss23_b = best_blend_params
print(f"Best weights: GBDT={w_g:.2f}, MLP={w_m:.2f}, MSE={w_ms:.2f}")
print(f"With calibration: SCALE={sc_b:.2f}, SHIFT={sh_b:.3f}")
print(f"BSS_24={bss24_b:.2f}, BSS_23={bss23_b:.2f}, AVG={best_bss_blend:.2f}")

# ====== PART 5: Individual model scores ======
print("\n" + "="*80)
print("PART 5: Individual Model BSS on 2024 holdout")
print("="*80)

for name, preds in [("GBDT_bin", p_gbdt_bin), ("MLP", p_mlp), ("GBDT_MSE", p_gbdt_mse),
                     ("LGB_raw", p_lgb), ("CB_raw", p_cb), ("XGB_raw", p_xgb_arr)]:
    bss = brier_skill(y[is_val24], np.clip(preds[is_val24], 1e-6, 1-1e-6))
    print(f"  {name}: BSS_24={bss:.2f}, mean={preds[is_val24].mean():.6f}")

# ====== PART 6: 5-row test sensitivity analysis ======
print("\n" + "="*80)
print("PART 6: 5-Row Test Sensitivity Analysis")
print("="*80)
# What does 1100 pts mean on 5 rows?
r = 0.4861
base_brier = r * (1 - r)  # 0.249867
target_bss = 1100  # target
target_bs = base_brier * (1 - target_bss / 100000.0)
current_bs_1032 = base_brier * (1 - 1032.82 / 100000.0)
print(f"Base Brier: {base_brier:.6f}")
print(f"For 1032.82 pts: BS={current_bs_1032:.6f} → total SE on 5 rows = {current_bs_1032 * 5:.6f}")
print(f"For 1100.00 pts: BS={target_bs:.6f} → total SE on 5 rows = {target_bs * 5:.6f}")
print(f"SE reduction needed (5 rows): {(current_bs_1032 - target_bs) * 5:.6f}")
print(f"Per-row avg SE reduction: {current_bs_1032 - target_bs:.6f}")

# v50 test predictions
v50_test = pd.read_csv(os.path.join(BASE_DIR, 'work', 'submit_v50', 'output', 'submission.csv'))
print(f"\nv50 test predictions: {v50_test['control_success'].values}")
print(f"v50 test mean: {v50_test['control_success'].mean():.6f}")

# If labels were all 0 or all 1, what would scores be?
for scenario_name, scenario_labels in [("All 0", np.zeros(5)), ("All 1", np.ones(5)),
                                        ("3 success, 2 fail", np.array([0,0,1,1,1])),
                                        ("2 success, 3 fail", np.array([1,0,0,1,0]))]:
    bs_scenario = np.mean((v50_test['control_success'].values - scenario_labels) ** 2)
    bss_scenario = 100000.0 * (1.0 - bs_scenario / base_brier)
    print(f"  Scenario '{scenario_name}': BSS = {bss_scenario:.2f}")

print("\n" + "="*80)
print("CONCLUSION: Theoretical BSS ceiling on 2024 holdout")
print("="*80)
# Perfect calibration on val24
p_perfect = y[is_val24].astype(np.float64)  # cheating - perfect predictions
bss_perfect = brier_skill(y[is_val24], p_perfect)
print(f"Perfect (oracle) BSS on 2024: {bss_perfect:.2f}")
print(f"v50 current BSS on 2024: {bss_24_v50:.2f}")
print(f"Gap to perfect: {bss_perfect - bss_24_v50:.2f}")
print(f"Best achievable with calibration only: {best_bss_24:.2f}")
print(f"v50 → best calibration gain: {best_bss_24 - bss_24_v50:.2f}")
