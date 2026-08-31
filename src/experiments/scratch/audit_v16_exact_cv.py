"""Exact, row-independent temporal CV for the deployed submit_v16/v23 recipe.

This deliberately does not reuse cache_final: that cache is a different,
128-feature experiment.  It mirrors train_submit_v16.py feature construction
and model parameters, while using only seasons strictly before the holdout.
"""
import argparse
import os
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

ROOT = os.path.expanduser("~/LG_data")
PKG = os.path.join(ROOT, "work", "submit_v16")
sys.path.insert(0, PKG)
import config  # noqa: E402
from preprocessing import PitchPreprocessor  # noqa: E402
from agent2_asof_decomp2 import AsofDecomposer2  # noqa: E402

SEEDS = [7, 123, 2025, 31415, 8675309]


class CatEmbedder(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(16, max(2, int(card ** 0.25 * 8))))
            for card in cards
        ])
        self.out_dim = sum(x.embedding_dim for x in self.embs)

    def forward(self, x):
        return torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class SimpleMLP(nn.Module):
    def __init__(self, n_num, cards):
        super().__init__()
        self.emb = CatEmbedder(cards)
        self.net = nn.Sequential(
            nn.Linear(n_num + self.emb.out_dim, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        return self.net(torch.cat([x_num, self.emb(x_cat)], 1)).squeeze(1)


def add_features(df_train, df_val, val_season):
    # The package persists intermediate artifacts during fit.  Redirect them
    # to an audit-only location so no submit package artifact can be altered.
    config.ARTIFACTS_DIR = os.path.join(ROOT, "scratch", "audit_v16_artifacts", str(val_season))
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    prep = PitchPreprocessor().fit(
        df_train, as_of_season=val_season - 1, is_final=False,
        trackman_path=os.path.join(ROOT, "open", "data", "trackman_history.csv"),
    )
    xtr, xva = prep.transform(df_train), prep.transform(df_val)
    # transform's merge resets the index; explicit positional assignment makes
    # CV identical to fresh test.csv inference rather than relying on index luck.
    for raw, out in ((df_train, xtr), (df_val, xva)):
        base = ((raw.runner_on_1b.fillna(0) > 0).astype(int).astype(str) + "_" +
                (raw.runner_on_2b.fillna(0) > 0).astype(int).astype(str) + "_" +
                (raw.runner_on_3b.fillna(0) > 0).astype(int).astype(str))
        count = raw.balls_before.fillna(0).astype(int).astype(str) + "_" + raw.strikes_before.fillna(0).astype(int).astype(str)
        out["count_x_base"] = (count + "_" + base).to_numpy()
    cmap = {v: i for i, v in enumerate(xtr["count_x_base"].unique())}
    xtr["count_x_base"] = xtr["count_x_base"].map(cmap).fillna(-1).astype(int)
    xva["count_x_base"] = xva["count_x_base"].map(cmap).fillna(-1).astype(int)
    for out in (xtr, xva):
        v0 = out.tkm_rel_speed_mean.clip(lower=60.0) * 1.46667
        tf = (60.5 - out.tkm_extension_mean.clip(lower=4.0, upper=8.0)) / v0
        rt = (tf - 0.15).clip(lower=0.01) / tf
        dt = np.sqrt((out.tkm_rel_side_mean + out.tkm_horz_break_mean / 12.0 * rt) ** 2 +
                     (out.tkm_rel_height_mean + out.tkm_induced_vert_break_mean / 12.0 * rt) ** 2)
        dp = np.sqrt((out.tkm_rel_side_mean + out.tkm_horz_break_mean / 12.0) ** 2 +
                     (out.tkm_rel_height_mean + out.tkm_induced_vert_break_mean / 12.0) ** 2)
        out["tkm_tunnel_dist_015s"] = dt.astype(np.float32)
        out["tkm_plate_break_divergence"] = ((dp - dt) / 0.15).astype(np.float32)
        out["tkm_deception_index"] = (dp / (dt + 0.1)).astype(np.float32)
    dec = AsofDecomposer2().fit(df_train, val_season)
    xtr = pd.concat([xtr, dec.transform(df_train).reset_index(drop=True)], axis=1)
    xva = pd.concat([xva, dec.transform(df_val).reset_index(drop=True)], axis=1)
    assert xtr.shape[1] == 119 and list(xtr.columns) == list(xva.columns), xtr.shape
    return xtr, xva


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"))
    tr, va = df[df.season < args.season].copy(), df[df.season == args.season].copy()
    xtr, xva = add_features(tr, va, args.season)
    ytr, yva = tr.control_success.to_numpy(), va.control_success.to_numpy()
    cats = [c for c in xtr if c in ["top_bottom", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "count_code", "platoon_matchup", "tkm_match", "count_x_base"]]
    pred = np.zeros((3, len(va)), dtype=np.float64)
    xcb_tr, xcb_va = xtr.copy(), xva.copy()
    for c in cats:
        xcb_tr[c] = xcb_tr[c].astype(int).astype(str); xcb_va[c] = xcb_va[c].astype(int).astype(str)
    for c in xtr.columns.difference(cats):
        xcb_tr[c] = xcb_tr[c].astype(np.float32); xcb_va[c] = xcb_va[c].astype(np.float32)
    xx_tr, xx_va = xtr.copy(), xva.copy()
    for c in cats:
        xx_tr[c] = xx_tr[c].astype(np.float32) if c == "count_x_base" else (xx_tr[c] - 1).astype(np.float32)
        xx_va[c] = xx_va[c].astype(np.float32) if c == "count_x_base" else (xx_va[c] - 1).astype(np.float32)
    for seed in SEEDS:
        print(f"seed {seed}", flush=True)
        ml = lgb.train(dict(objective="regression", metric="rmse", learning_rate=.05, num_leaves=31, seed=seed, verbose=-1, n_estimators=300, min_child_samples=50, subsample=.8, colsample_bytree=.8), lgb.Dataset(xtr, label=ytr))
        pred[0] += ml.predict(xva)
        mc = CatBoostClassifier(iterations=300, learning_rate=.06, depth=6, cat_features=cats, random_seed=seed, verbose=0)
        mc.fit(xcb_tr, ytr); pred[1] += mc.predict_proba(xcb_va)[:, 1]
        mx = xgb.XGBClassifier(n_estimators=250, learning_rate=.05, max_depth=5, random_state=seed, tree_method="hist", subsample=.8, colsample_bytree=.8, eval_metric="logloss")
        mx.fit(xx_tr.astype(np.float32), ytr); pred[2] += mx.predict_proba(xx_va.astype(np.float32))[:, 1]
    pred /= len(SEEDS)
    out = {"y": yva.astype(np.int8), "p_lgb": pred[0].astype(np.float32), "p_cb": pred[1].astype(np.float32), "p_xgb": pred[2].astype(np.float32), "n_features": np.array(119), "val_season": np.array(args.season)}
    if not args.skip_mlp:
        nums = [c for c in xtr if c not in cats]
        mean, std = xtr[nums].to_numpy(np.float32).mean(0), xtr[nums].to_numpy(np.float32).std(0)
        std[std < 1e-6] = 1
        vocabs = [{v: i for i, v in enumerate(sorted(xtr[c].astype(str).unique()))} for c in cats]
        cards = [len(v) + 1 for v in vocabs]
        ntr = torch.tensor(np.nan_to_num((xtr[nums].to_numpy(np.float32) - mean) / std), dtype=torch.float32)
        nva = torch.tensor(np.nan_to_num((xva[nums].to_numpy(np.float32) - mean) / std), dtype=torch.float32)
        def enc(d): return torch.tensor(np.stack([d[c].astype(str).map(v).fillna(len(v)).to_numpy(np.int64) for c, v in zip(cats, vocabs)], 1))
        ctr, cva, yt = enc(xtr), enc(xva), torch.tensor(ytr, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ntr, ctr, yt), batch_size=2048, shuffle=True)
        pm = np.zeros(len(va))
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            m = SimpleMLP(len(nums), cards); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
            for _ in range(5):
                m.train()
                for bn, bc, by in loader:
                    opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(m(bn, bc), by); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): pm += torch.sigmoid(m(nva, cva)).numpy()
        out["p_mlp"] = (pm / len(SEEDS)).astype(np.float32)
    dest = os.path.join(ROOT, "scratch", "audit_v16_exact")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, f"val{args.season}.npz")
    np.savez_compressed(path, **out)
    print(f"saved {path}; elapsed={(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
