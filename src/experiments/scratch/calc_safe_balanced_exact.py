import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

BASE_DIR = os.path.expanduser('~/LG_data')
data_dir = os.path.join(BASE_DIR, 'open', 'data')
work_v42_dir = os.path.join(BASE_DIR, 'work', 'submit_v42')
model_dir = os.path.join(work_v42_dir, 'model')

sys.path.insert(0, work_v42_dir)
from preprocessing import PitchPreprocessor
from trackman_features import TrackmanFeatureBuilder
from agent2_asof_decomp2 import AsofDecomposer2

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill(y_true, y_prob, r=0.4861):
    base_brier = r * (1 - r)
    return 100000.0 * (1.0 - brier_score(y_true, y_prob) / base_brier)

df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df_val24 = df[df['season'] == 2024].reset_index(drop=True)
df_val23 = df[df['season'] == 2023].reset_index(drop=True)
y_24 = df_val24['control_success'].values.astype(np.float32)
y_23 = df_val23['control_success'].values.astype(np.float32)

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
count_shifts = joblib.load(os.path.join(model_dir, 'count_shifts_artifact.pkl'))

def get_features_and_preds(df_eval):
    X_base = prep.transform(df_eval)
    base_str = ((df_eval['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_eval['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_eval['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (df_eval['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_eval['strikes_before'].fillna(0).astype(int).astype(str))
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

    A_eval = dec.transform(df_eval)
    A_eval.index = X_base.index
    X_base = pd.concat([X_base, A_eval], axis=1)

    v_rel = X_base['tkm_rel_speed_mean'].clip(lower=60.0)
    spin = X_base['tkm_spin_rate_mean'].clip(lower=500.0)
    dist_to_plate = (60.5 - ext).clip(lower=50.0)

    X_133 = X_base.copy()
    X_133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
    X_133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X_133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X_133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0)**2 + (hb * 12.0)**2) / spin).astype(np.float32)

    b = df_eval['balls_before'].fillna(0).values
    s = df_eval['strikes_before'].fillna(0).values
    li = df_eval['li'].fillna(1.0).values
    r2 = (df_eval['runner_on_2b'].fillna(0) > 0).astype(float).values
    r3 = (df_eval['runner_on_3b'].fillna(0) > 0).astype(float).values
    score_diff = df_eval['score_diff_pitcher_team'].fillna(0).values
    inning = df_eval['inning'].fillna(1).values
    fb_rate = df_eval['asof_pitcher_fastball_rate'].fillna(0.5).values
    br_rate = df_eval['asof_pitcher_breaking_rate'].fillna(0.3).values
    off_rate = df_eval['asof_pitcher_offspeed_rate'].fillna(0.2).values
    platoon_code = (df_eval['pitcher_hand'].astype(str) == df_eval['batter_hand'].astype(str)).astype(float).values

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

    SEEDS = [7, 123, 2025, 31415, 8675309]
    p_lgb_sum = np.zeros(len(df_eval))
    p_cb_sum = np.zeros(len(df_eval))
    p_xgb_sum = np.zeros(len(df_eval))
    p_lgb_mse_sum = np.zeros(len(df_eval))

    X_133_mat = X_133.values.astype(np.float32)

    for seed in SEEDS:
        m_lgb = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_model_seed{seed}.txt'))
        p_lgb_sum += m_lgb.predict(X_base)
        m_cb = CatBoostClassifier()
        m_cb.load_model(os.path.join(model_dir, f'catboost_model_seed{seed}.cbm'))
        p_cb_sum += m_cb.predict_proba(X_cb)[:, 1]
        m_xgb = xgb.XGBClassifier()
        m_xgb.load_model(os.path.join(model_dir, f'xgb_model_seed{seed}.json'))
        p_xgb_sum += m_xgb.predict_proba(X_xgb)[:, 1]
        m_lgb_mse = lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_mse_model_seed{seed}.txt'))
        p_lgb_mse_sum += m_lgb_mse.predict(X_133_mat)

    n_seeds = len(SEEDS)
    p_lgb_bin = np.clip(p_lgb_sum / n_seeds - 0.007, 1e-6, 1 - 1e-6)
    p_cb_bin = np.clip(p_cb_sum / n_seeds - 0.008, 1e-6, 1 - 1e-6)
    p_xgb_bin = np.clip(p_xgb_sum / n_seeds - 0.006, 1e-6, 1 - 1e-6)
    p_gbdt_bin = np.clip(0.20 * p_lgb_bin + 0.72 * p_cb_bin + 0.08 * p_xgb_bin, 1e-6, 1 - 1e-6)
    p_gbdt_mse = np.clip(p_lgb_mse_sum / n_seeds, 1e-6, 1 - 1e-6)

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

    art = joblib.load(os.path.join(model_dir, 'mlp_artifacts.pkl'))
    num_cols_mlp, cat_cols_mlp = art['num_cols'], art['cat_cols']
    mean_mlp, std_mlp = art['mean'], art['std']
    cat_vocabs = art['cat_vocabs']
    cat_cardinalities = art['cat_cardinalities']
    num_dim = art['num_dim']

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

    p_mlp_sum = np.zeros(len(df_eval), dtype=np.float64)
    for seed in SEEDS:
        mlp_net = SimpleMLP_MSE(num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12)
        mlp_net.load_state_dict(torch.load(os.path.join(model_dir, f'mlp_model_seed{seed}.pt')))
        mlp_net.eval()
        with torch.no_grad():
            p_mlp_sum += mlp_net(num_t, cat_t).numpy()
    p_mlp_mse = p_mlp_sum / len(SEEDS)

    counts_eval = (df_eval['balls_before'].fillna(0).astype(int).astype(str) + '_' + df_eval['strikes_before'].fillna(0).astype(int).astype(str)).values
    return p_gbdt_bin, p_mlp_mse, p_gbdt_mse, counts_eval

p24_gbdt_bin, p24_mlp_mse, p24_gbdt_mse, c24 = get_features_and_preds(df_val24)
p23_gbdt_bin, p23_mlp_mse, p23_gbdt_mse, c23 = get_features_and_preds(df_val23)

# 1. v42 Baseline
p24_v42 = np.clip(0.5 + 1.10 * (0.40*p24_gbdt_bin + 0.40*p24_mlp_mse + 0.20*p24_gbdt_mse - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
p23_v42 = np.clip(0.5 + 1.10 * (0.40*p23_gbdt_bin + 0.40*p23_mlp_mse + 0.20*p23_gbdt_mse - 0.5) - 0.0045192086, 1e-6, 1 - 1e-6)
s24_v42 = brier_skill(y_24, p24_v42)
s23_v42 = brier_skill(y_23, p23_v42)
avg_v42 = 0.5 * (s24_v42 + s23_v42)

# 2. Safe Balanced (GBDT 25% / MLP 50% / MSE 25%, Scale 1.10, Shift -0.0035)
p24_bal = 0.25*p24_gbdt_bin + 0.50*p24_mlp_mse + 0.25*p24_gbdt_mse
p23_bal = 0.25*p23_gbdt_bin + 0.50*p23_mlp_mse + 0.25*p23_gbdt_mse
for cc, s_val in count_shifts.items():
    p24_bal[c24 == cc] += s_val
    p23_bal[c23 == cc] += s_val
p24_bal_cal = np.clip(0.5 + 1.10 * (p24_bal - 0.5) - 0.003500, 1e-6, 1 - 1e-6)
p23_bal_cal = np.clip(0.5 + 1.10 * (p23_bal - 0.5) - 0.003500, 1e-6, 1 - 1e-6)
s24_bal = brier_skill(y_24, p24_bal_cal)
s23_bal = brier_skill(y_23, p23_bal_cal)
avg_bal = 0.5 * (s24_bal + s23_bal)

print(f"v42 Baseline 2-Year Score:      {avg_v42:.2f} pts (2023: {s23_v42:.2f}, 2024: {s24_v42:.2f})")
print(f"Safe Balanced 2-Year Score:    {avg_bal:.2f} pts (2023: {s23_bal:.2f}, 2024: {s24_bal:.2f})")
print(f"Gain over v42:                 +{avg_bal - avg_v42:.2f} pts")
