"""
agent8_dl_diversify_screen.py

Agent8's DL architecture diversification, per orchestrator instructions:
try architectures BEYOND the already-confirmed SimpleMLP (agent6/agent7's
w_mlp=0.32 blending, outer(2024) single-seed gain=+14.29, 5-seed re-check
in progress separately). Candidates:
  (a) DeepMLP    -- wider/deeper cat-embedding MLP (256-128-64, BatchNorm)
  (b) PLEMLP     -- Piecewise Linear Encoding (dl_common.PLEEncoder, already
                     implemented) for numeric features + cat embeddings + MLP
  (c) LightFTT   -- lightweight FT-Transformer: per-feature linear tokenizer
                     for numeric scalars + cat embeddings as tokens, CLS token,
                     1 TransformerEncoder layer, small d_token=16 (kept tiny to
                     stay CPU-feasible, no MPS -- report 161 stall history)

Scope: INNER folds ONLY (val=2022, val=2023). GBDT reference is built ONCE per
fold (single seed=7, identical recipe to agent6/agent7) and reused across all
three DL candidates within that fold, to save compute. The 2024 (outer) fold
is never built or touched by this script -- only the single most promising
candidate (by inner-only shared-weight gain) should go to a SEPARATE outer
confirm script afterward, applied exactly once, no re-tuning.

Row-independence: all categorical encoding goes through dl_common.to_tensors,
which builds vocab from the TRAIN split only and applies it via a fixed
per-row .map() -- structurally row-independent (same pattern already used and
empirically verified for SimpleMLP in agent7's 5-seed script). No new
category-encoding code is introduced here.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score
from agent2_asof_decomp2 import AsofDecomposer2
import dl_common as dlc

DEVICE = torch.device('cpu')  # force CPU -- MPS stall history (report 161)
SEED = 7
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


# ---------------------------------------------------------------------------
# Candidate DL architectures
# ---------------------------------------------------------------------------

class DeepMLP(nn.Module):
    """Wider/deeper cat-embedding MLP than SimpleMLP: 256-128-64 with BatchNorm."""
    def __init__(self, num_dim, cat_cardinalities, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.cat_embedder = dlc.CatEmbedder(cat_cardinalities, emb_dim=8, max_emb_dim=24)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


class PLEMLP(nn.Module):
    """PLE-encoded numeric features (raw, NOT z-scored) + cat embeddings -> MLP."""
    def __init__(self, cat_cardinalities, bin_edges_list, hidden=(128, 64), dropout=0.15):
        super().__init__()
        self.ple = dlc.PLEEncoder(bin_edges_list)
        self.cat_embedder = dlc.CatEmbedder(cat_cardinalities)
        in_dim = self.ple.out_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num_raw, x_cat):
        x_ple = self.ple(x_num_raw)
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_ple, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


class LightFTT(nn.Module):
    """Lightweight FT-Transformer: per-feature linear tokenizer for numeric
    scalars (each -> d_token vector via its own weight/bias row) + categorical
    embeddings as tokens, CLS token prepended, 1 TransformerEncoder layer,
    small d_token to stay CPU-feasible."""
    def __init__(self, num_dim, cat_cardinalities, d_token=16, n_heads=2, n_layers=1, dropout=0.1):
        super().__init__()
        self.num_dim = num_dim
        self.num_weight = nn.Parameter(torch.randn(num_dim, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(num_dim, d_token))
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinalities])
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                                                dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
        if len(self.cat_embs) > 0:
            cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
            tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        else:
            tokens = num_tokens
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.transformer(seq)
        return self.head(out[:, 0, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Shared fold-building (identical to agent6/agent7's asof_dec recipe)
# ---------------------------------------------------------------------------

def add_asof_dec_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    val_season = fold_max_season + 1
    dec = AsofDecomposer2().fit(df_tr_f, val_season=val_season)
    tr_feats = dec.transform(df_tr_f)
    val_feats = dec.transform(df_val_f)
    tr_feats.index = X_tr_f.index
    val_feats.index = X_val_f.index
    return pd.concat([X_tr_f, tr_feats], axis=1), pd.concat([X_val_f, val_feats], axis=1)


def build_asofdec_fold_frames(df_train, fold):
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)
    dlc.add_count_x_base(df_tr_f, X_tr_f)
    dlc.add_count_x_base(df_val_f, X_val_f)
    cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
    X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_tr_f, X_val_f = add_asof_dec_features(df_tr_f, df_val_f, fold.fold_max_season, X_tr_f, X_val_f)
    y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)
    y_val_f = df_val_f[config.TARGET_COL].values.astype(np.float32)
    return X_tr_f, X_val_f, y_tr_f, y_val_f


def fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f):
    cat_cols = [c for c in X_tr_f.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS
                or c == config.TRACKMAN_MATCH_FLAG_COL or c == 'count_x_base']
    cat_idx = [X_tr_f.columns.get_loc(c) for c in cat_cols if c in X_tr_f.columns]

    m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    m_lgb.fit(X_tr_f, y_tr_f, categorical_feature=cat_idx)
    p_lgb = np.clip(m_lgb.predict_proba(X_val_f)[:, 1] + SHIFTS['lgb'], 1e-6, 1 - 1e-6)

    X_tr_cb, X_val_cb = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(int).astype(str)
        X_val_cb[c] = X_val_cb[c].astype(int).astype(str)
    for c in [col for col in X_tr_cb.columns if col not in cat_cols]:
        X_tr_cb[c] = X_tr_cb[c].astype(np.float32)
        X_val_cb[c] = X_val_cb[c].astype(np.float32)
    m_cb = CatBoostClassifier(cat_features=cat_cols, **CB_PARAMS)
    m_cb.fit(X_tr_cb, y_tr_f)
    p_cb = np.clip(m_cb.predict_proba(X_val_cb)[:, 1] + SHIFTS['cb'], 1e-6, 1 - 1e-6)

    # NOTE: value-1 fixed transform (report 203 fix), NOT batch-dependent .cat.codes
    X_tr_x, X_val_x = X_tr_f.copy(), X_val_f.copy()
    for c in cat_cols:
        if c == 'count_x_base':
            X_tr_x[c] = X_tr_x[c].astype(np.float32)
            X_val_x[c] = X_val_x[c].astype(np.float32)
        else:
            X_tr_x[c] = (X_tr_x[c].astype(np.float32) - 1.0)
            X_val_x[c] = (X_val_x[c].astype(np.float32) - 1.0)
    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(X_tr_x.astype(np.float32), y_tr_f)
    p_xgb = np.clip(m_xgb.predict_proba(X_val_x.astype(np.float32))[:, 1] + SHIFTS['xgb'], 1e-6, 1 - 1e-6)

    w_lgb, w_cb, w_xgb = WEIGHTS
    p_ens = np.clip(w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb, 1e-6, 1 - 1e-6)
    return p_ens


def best_shared_weight(results, vs_list):
    """Single w (0..0.5 grid) maximizing avg skill across vs_list -- inner-only, no per-fold cherry-pick."""
    gbdt_only_avg = float(np.mean([results[vs]['sk_gbdt'] for vs in vs_list]))
    best_w, best_avg = 0.0, gbdt_only_avg
    for w in np.linspace(0, 0.5, 26):
        sks = []
        for vs in vs_list:
            r = results[vs]
            p_blend = np.clip((1 - w) * r['p_gbdt'] + w * r['p_dl'], 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(r['y'], p_blend)
            sks.append(sk)
        avg = float(np.mean(sks))
        if avg > best_avg:
            best_avg, best_w = avg, float(w)
    return best_w, best_avg, gbdt_only_avg


def main():
    t_start = time.time()
    df_train = pd.read_csv(config.TRAIN_PATH)
    folds = get_cv_folds(df_train)
    inner_folds = [f for f in folds if f.val_season in (2022, 2023)]

    candidates = ['DeepMLP', 'PLEMLP', 'LightFTT']
    all_results = {c: {} for c in candidates}

    for fold in inner_folds:
        vs = fold.val_season
        log(f"=== fold val={vs}: building asof_dec (v2) feature frames ===")
        t0 = time.time()
        X_tr_f, X_val_f, y_tr_f, y_val_f = build_asofdec_fold_frames(df_train, fold)
        log(f"fold val={vs}: X_tr={X_tr_f.shape} X_val={X_val_f.shape} built in {time.time()-t0:.1f}s")

        log(f"fold val={vs}: fitting GBDT reference (single seed={SEED}, shared across all 3 candidates) ...")
        t0 = time.time()
        p_gbdt = fit_predict_gbdt(X_tr_f, X_val_f, y_tr_f)
        sk_gbdt, br_gbdt, _, _ = calc_brier_skill_score(y_val_f, p_gbdt)
        log(f"fold val={vs}: GBDT alone skill={sk_gbdt:.2f} (raw brier={br_gbdt:.6f}) in {time.time()-t0:.1f}s")

        tens = dlc.to_tensors(X_tr_f, X_val_f)
        num_tr, num_val = tens['num_tr'], tens['num_val']
        num_tr_raw, num_val_raw = tens['num_tr_raw'], tens['num_val_raw']
        cat_tr, cat_val = tens['cat_tr'], tens['cat_val']
        y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)
        num_dim = num_tr.shape[1]
        cat_cardinalities = tens['cat_cardinalities']

        # ---- (a) DeepMLP ----
        log(f"fold val={vs}: training DeepMLP (256-128-64, BatchNorm) ...")
        t0 = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = DeepMLP(num_dim, cat_cardinalities)
        model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=10, lr=1e-3,
                                          batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                          verbose_prefix=f"[DeepMLP val={vs}] ")
        p_deep = dlc.predict(model, num_val, cat_val, DEVICE, shift)
        sk_deep, *_ = calc_brier_skill_score(y_val_f, p_deep)
        corr_deep = float(np.corrcoef(p_deep, p_gbdt)[0, 1])
        log(f"fold val={vs}: DeepMLP alone skill={sk_deep:.2f} corr(vs GBDT)={corr_deep:.4f} ({time.time()-t0:.1f}s)")
        all_results['DeepMLP'][vs] = dict(sk_dl=sk_deep, sk_gbdt=sk_gbdt, corr=corr_deep,
                                           p_dl=p_deep, p_gbdt=p_gbdt, y=y_val_f)

        # ---- (b) PLEMLP ----
        log(f"fold val={vs}: training PLEMLP (16 quantile bins, train-only edges) ...")
        t0 = time.time()
        bin_edges = dlc.compute_ple_bin_edges(num_tr_raw.numpy(), n_bins=16)
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = PLEMLP(cat_cardinalities, bin_edges)
        model, shift = dlc.train_generic(model, num_tr_raw, cat_tr, y_tr_t, epochs=10, lr=1e-3,
                                          batch_size=8192, device=DEVICE, weight_decay=1e-5,
                                          verbose_prefix=f"[PLEMLP val={vs}] ")
        p_ple = dlc.predict(model, num_val_raw, cat_val, DEVICE, shift)
        sk_ple, *_ = calc_brier_skill_score(y_val_f, p_ple)
        corr_ple = float(np.corrcoef(p_ple, p_gbdt)[0, 1])
        log(f"fold val={vs}: PLEMLP alone skill={sk_ple:.2f} corr(vs GBDT)={corr_ple:.4f} ({time.time()-t0:.1f}s)")
        all_results['PLEMLP'][vs] = dict(sk_dl=sk_ple, sk_gbdt=sk_gbdt, corr=corr_ple,
                                          p_dl=p_ple, p_gbdt=p_gbdt, y=y_val_f)

        # ---- (c) LightFTT ----
        log(f"fold val={vs}: training LightFTT (d_token=16, 1 transformer layer) ...")
        t0 = time.time()
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = LightFTT(num_dim, cat_cardinalities, d_token=16, n_heads=2, n_layers=1)
        model, shift = dlc.train_generic(model, num_tr, cat_tr, y_tr_t, epochs=8, lr=1e-3,
                                          batch_size=4096, device=DEVICE, weight_decay=1e-5,
                                          verbose_prefix=f"[LightFTT val={vs}] ")
        p_ftt = dlc.predict(model, num_val, cat_val, DEVICE, shift)
        sk_ftt, *_ = calc_brier_skill_score(y_val_f, p_ftt)
        corr_ftt = float(np.corrcoef(p_ftt, p_gbdt)[0, 1])
        log(f"fold val={vs}: LightFTT alone skill={sk_ftt:.2f} corr(vs GBDT)={corr_ftt:.4f} ({time.time()-t0:.1f}s)")
        all_results['LightFTT'][vs] = dict(sk_dl=sk_ftt, sk_gbdt=sk_gbdt, corr=corr_ftt,
                                            p_dl=p_ftt, p_gbdt=p_gbdt, y=y_val_f)

    log("\n=== SUMMARY (INNER ONLY -- 2024/outer never built) ===")
    summary = {}
    for c in candidates:
        w, avg, gbdt_avg = best_shared_weight(all_results[c], (2022, 2023))
        gain = avg - gbdt_avg
        summary[c] = dict(best_w=w, inner_avg=avg, gbdt_avg=gbdt_avg, gain=gain)
        log(f"{c}: shared best_w={w:.2f} -> inner avg skill={avg:.2f} (GBDT-alone inner avg={gbdt_avg:.2f}, "
            f"gain={gain:+.2f})")
        for vs in (2022, 2023):
            r = all_results[c][vs]
            log(f"    val={vs}: {c} alone={r['sk_dl']:.2f} | GBDT alone={r['sk_gbdt']:.2f} | corr={r['corr']:.4f}")

    best_candidate = max(summary, key=lambda c: summary[c]['gain'])
    log(f"\n=== BEST CANDIDATE BY INNER-ONLY GAIN: {best_candidate} "
        f"(gain={summary[best_candidate]['gain']:+.2f}, w={summary[best_candidate]['best_w']:.2f}) ===")

    np.savez('/tmp/agent8_dl_diversify_screen.npz',
              **{f'p_{c}_{vs}': all_results[c][vs]['p_dl'] for c in candidates for vs in (2022, 2023)},
              **{f'p_gbdt_{vs}': all_results[candidates[0]][vs]['p_gbdt'] for vs in (2022, 2023)},
              **{f'y_{vs}': all_results[candidates[0]][vs]['y'] for vs in (2022, 2023)},
              summary_json=str(summary), best_candidate=best_candidate)
    log(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
