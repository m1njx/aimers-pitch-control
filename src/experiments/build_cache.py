"""
build_cache.py — Honest validation harness for the DACON pitch-control project.

WHY THIS EXISTS
---------------
`outputs/502` proved the old local check (running final-mode submission artifacts
on 2024) is ANTI-correlated with the real leaderboard: Spearman rho = -0.198.
Those artifacts were trained with is_final=True on 2019-2024, so scoring them on
2024 is an in-sample measurement, and the draws that fit 2024 best did WORST on
the 2025 leaderboard.

This harness fixes the three defects named in 502 section 4:
  1. the evaluation season is excluded from training (honest refit per fold),
  2. several independent training draws per fold (LB retrain noise is +-34 pts),
  3. several evaluation seasons, since 2024 is a base-rate outlier (.4861).

It writes a PREDICTION CACHE: per (eval_season, seed, component) validation
predictions. Once cached, scoring any blend/calibration config is instant and
needs no retraining -- that is what makes config search possible at all.

Component recipe is copied verbatim from work/submit_v42/train_submit_v16.py and
train_mlp_only.py so the harness measures the production architecture.

USAGE
-----
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           KMP_DUPLICATE_LIB_OK=TRUE          # REQUIRED on macOS, see outputs/502 s.5
    venv311/bin/python3 harness/build_cache.py --years 2024 2023 --seeds 7 123 2025

Resumable: an existing cache file for a (season, seed) pair is skipped.
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

LG = os.path.expanduser('~/LG_data')
SRC = os.path.join(LG, 'work/submit_v42')      # canonical copy of the pipeline modules
sys.path.insert(0, SRC)
sys.path.insert(0, LG)

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import torch
import torch.nn as nn

from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

CACHE = os.path.join(LG, 'harness/cache')
os.makedirs(CACHE, exist_ok=True)

CAT_COLS = ['top_bottom', 'base_state', 'pitcher_hand', 'batter_hand',
            'pitcher_team_id', 'batter_team_id', 'count_code',
            'platoon_matchup', 'tkm_match', 'count_x_base']


# ----------------------------------------------------------------------------
# feature construction -- mirrors work/submit_v42/script.py lines 104-170
# ----------------------------------------------------------------------------
def build_features(df, prep, dec, cat_map):
    X = prep.transform(df)

    base_str = ((df['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (df['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df['strikes_before'].fillna(0).astype(int).astype(str))
    X['count_x_base'] = (cc_str + '_' + base_str).map(cat_map).fillna(-1).astype(int)

    # 3D pitch tunneling (module 264)
    v0 = X['tkm_rel_speed_mean'].clip(lower=60.0) * 1.46667
    ext = X['tkm_extension_mean'].clip(lower=4.0, upper=8.0)
    rel_side, rel_height = X['tkm_rel_side_mean'], X['tkm_rel_height_mean']
    ivb, hb = X['tkm_induced_vert_break_mean'] / 12.0, X['tkm_horz_break_mean'] / 12.0
    t_flight = (60.5 - ext) / v0
    r_ratio = (t_flight - 0.15).clip(lower=0.01) / t_flight
    d_tunnel = np.sqrt((rel_side + hb * r_ratio) ** 2 + (rel_height + ivb * r_ratio) ** 2)
    d_plate = np.sqrt((rel_side + hb) ** 2 + (rel_height + ivb) ** 2)
    X['tkm_tunnel_dist_015s'] = d_tunnel.astype(np.float32)
    X['tkm_plate_break_divergence'] = ((d_plate - d_tunnel) / 0.15).astype(np.float32)
    X['tkm_deception_index'] = (d_plate / (d_tunnel + 0.1)).astype(np.float32)

    A = dec.transform(df)
    A.index = X.index
    X = pd.concat([X, A], axis=1)                       # -> 119 features

    # ---- the extra 14 that make the 133-feature matrix ----
    X133 = X.copy()
    v_rel = X['tkm_rel_speed_mean'].clip(lower=60.0)
    spin = X['tkm_spin_rate_mean'].clip(lower=500.0)
    dist_to_plate = (60.5 - ext)
    X133['phys_effective_velocity'] = (v_rel * (60.5 / dist_to_plate)).astype(np.float32)
    X133['phys_vaa_proxy'] = (np.arctan((rel_height - 2.5 + ivb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X133['phys_haa_proxy'] = (np.arctan((rel_side + hb) / dist_to_plate) * (180.0 / np.pi)).astype(np.float32)
    X133['phys_spin_efficiency'] = (np.sqrt((ivb * 12.0) ** 2 + (hb * 12.0) ** 2) / spin).astype(np.float32)

    b = df['balls_before'].fillna(0).values
    s = df['strikes_before'].fillna(0).values
    li = df['li'].fillna(1.0).values
    r2 = (df['runner_on_2b'].fillna(0) > 0).astype(float).values
    r3 = (df['runner_on_3b'].fillna(0) > 0).astype(float).values
    sd = df['score_diff_pitcher_team'].fillna(0).values
    inn = df['inning'].fillna(1).values
    fb = df['asof_pitcher_fastball_rate'].fillna(0.5).values
    br = df['asof_pitcher_breaking_rate'].fillna(0.3).values
    off = df['asof_pitcher_offspeed_rate'].fillna(0.2).values
    plat = (df['pitcher_hand'].astype(str) == df['batter_hand'].astype(str)).astype(float).values

    X133['feat_count_advantage'] = (s - 1.5 * b).astype(np.float32)
    X133['feat_full_count'] = ((b == 3) & (s == 2)).astype(np.float32)
    X133['feat_pitcher_ahead'] = ((s > b) & (s >= 2)).astype(np.float32)
    X133['feat_pitcher_behind'] = ((b > s) & (b >= 2)).astype(np.float32)
    X133['feat_clutch_pressure'] = (li * (1.0 + r2 + r3) * np.exp(-np.clip(sd ** 2 / 10.0, 0, 5.0))).astype(np.float32)
    X133['feat_scoring_position'] = (r2 + r3).astype(np.float32)
    X133['feat_platoon_fastball_inter'] = (plat * fb).astype(np.float32)
    X133['feat_platoon_breaking_inter'] = (plat * br).astype(np.float32)
    X133['feat_platoon_offspeed_inter'] = (plat * off).astype(np.float32)
    X133['feat_late_inning_clutch'] = ((inn >= 7).astype(float) * li).astype(np.float32)
    return X, X133


def cast_cb(X):
    Z = X.copy()
    for c in CAT_COLS:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(-1).astype(int).astype(str)
    for c in [c for c in Z.columns if c not in CAT_COLS]:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0.0).astype(np.float32)
    return Z


def cast_xgb(X):
    Z = X.copy()
    for c in CAT_COLS:
        Z[c] = Z[c].astype(np.float32) if c == 'count_x_base' else (Z[c] - 1).astype(np.float32)
    return Z.astype(np.float32)


class CatEmbedder(nn.Module):
    def __init__(self, cards, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(max_emb_dim, max(2, int(c ** 0.25 * emb_dim)))) for c in cards])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x):
        if not len(self.embs):
            return torch.zeros(x.shape[0], 0)
        return torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=1)


class SimpleMLP_MSE(nn.Module):
    """Identical to work/submit_v42/script.py -- sigmoid head, trained under MSE."""
    def __init__(self, num_dim, cards, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cards)
        layers, prev = [], num_dim + self.cat_embedder.out_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        return self.net(torch.cat([xn, self.cat_embedder(xc)], dim=1)).squeeze(-1)


def mlp_arrays(X133, art=None):
    if art is None:
        num_cols = [c for c in X133.columns if c not in CAT_COLS]
        cat_cols = [c for c in X133.columns if c in CAT_COLS]
        mean = X133[num_cols].mean(axis=0).values.astype(np.float32)
        std = X133[num_cols].std(axis=0).values.astype(np.float32)
        std[std == 0] = 1.0
        vocabs = {c: {v: i for i, v in enumerate(X133[c].astype(str).unique())} for c in cat_cols}
        art = dict(num_cols=num_cols, cat_cols=cat_cols, mean=mean, std=std,
                   vocabs=vocabs, cards=[len(vocabs[c]) + 1 for c in cat_cols])
    nz = np.nan_to_num((X133[art['num_cols']].values.astype(np.float32) - art['mean']) / art['std'], nan=0.0)
    ca = np.stack([X133[c].astype(str).map(art['vocabs'][c]).fillna(len(art['vocabs'][c])).astype(np.int64).values
                   for c in art['cat_cols']], axis=1)
    return nz, ca, art


# ----------------------------------------------------------------------------
def run_fold(df, year, seeds):
    tr = df[df.season < year]
    va = df[df.season == year]
    print(f'\n=== eval_season {year}: train {len(tr):,} (<= {year-1})  val {len(va):,} ===', flush=True)
    t0 = time.time()

    prep = PitchPreprocessor()
    prep.fit(tr, as_of_season=year - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    base_str = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              tr['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cc_str + '_' + base_str).unique())}

    dec = AsofDecomposer2()
    dec.fit(tr, val_season=year)

    Xtr, Xtr133 = build_features(tr, prep, dec, cat_map)
    Xva, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)
    yva = va['control_success'].values.astype(np.float64)
    print(f'  features built {Xtr.shape[1]}/{Xtr133.shape[1]}  ({time.time()-t0:.0f}s)', flush=True)

    Xtr_cb, Xva_cb = cast_cb(Xtr), cast_cb(Xva)
    Xtr_xg, Xva_xg = cast_xgb(Xtr), cast_xgb(Xva)
    Xtr133m, Xva133m = Xtr133.values.astype(np.float32), Xva133.values.astype(np.float32)
    nz_tr, ca_tr, art = mlp_arrays(Xtr133)
    nz_va, ca_va, _ = mlp_arrays(Xva133, art)

    np.save(os.path.join(CACHE, f'y_{year}.npy'), yva)

    for seed in seeds:
        f = os.path.join(CACHE, f'pred_{year}_{seed}.npz')
        if os.path.exists(f):
            print(f'  seed {seed}: cached, skip', flush=True)
            continue
        t1 = time.time()
        out = {}

        p = dict(objective='regression', metric='rmse', learning_rate=0.05, num_leaves=31,
                 seed=seed, verbose=-1, n_estimators=300, min_child_samples=50,
                 subsample=0.8, colsample_bytree=0.8)
        out['lgb_bin'] = lgb.train(p, lgb.Dataset(Xtr, label=ytr)).predict(Xva)

        m = CatBoostClassifier(iterations=300, learning_rate=0.06, depth=6,
                               cat_features=CAT_COLS, random_seed=seed, verbose=0,
                               thread_count=6)
        m.fit(Xtr_cb, ytr)
        out['cb_bin'] = m.predict_proba(Xva_cb)[:, 1]

        m = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=5,
                              random_state=seed, tree_method='hist', subsample=0.8,
                              colsample_bytree=0.8, eval_metric='logloss', n_jobs=6)
        m.fit(Xtr_xg, ytr)
        out['xgb_bin'] = m.predict_proba(Xva_xg)[:, 1]

        p133 = dict(p); p133['seed'] = seed + 1
        out['lgb_mse'] = lgb.train(p133, lgb.Dataset(Xtr133m, label=ytr)).predict(Xva133m)

        torch.manual_seed(seed)
        net = SimpleMLP_MSE(len(art['num_cols']), art['cards'])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(torch.tensor(nz_tr), torch.tensor(ca_tr),
                                            torch.tensor(ytr, dtype=torch.float32))
        dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
        net.train()
        for _ in range(5):
            for bn, bc, by in dl:
                opt.zero_grad(); crit(net(bn, bc), by).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out['mlp'] = net(torch.tensor(nz_va), torch.tensor(ca_va)).numpy().astype(np.float64)

        np.savez_compressed(f, **out)
        print(f'  seed {seed}: 5 components trained+cached ({time.time()-t1:.0f}s)', flush=True)

    print(f'=== {year} done in {(time.time()-t0)/60:.1f} min ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2024, 2023, 2022])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123, 2025])
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        print('WARNING: set OMP_NUM_THREADS=1 (see outputs/502 s.5) or this will deadlock on macOS')
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    for y in a.years:
        run_fold(df, y, a.seeds)
