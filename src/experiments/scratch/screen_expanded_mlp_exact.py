"""Strict temporal screen for a capacity-expanded tabular MLP.

This is deliberately separate from cache_final and the deployable packages.
It reuses only the exact v16/v33 feature builder, fits each season using
strictly earlier seasons, and never reads test.csv.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.expanduser("~/LG_data")
sys.path.insert(0, os.path.join(ROOT, "scratch"))
from audit_v16_exact_cv import add_features


class CatEmbedder(nn.Module):
    def __init__(self, cards):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(card, min(24, max(3, int(card ** 0.25 * 12))))
            for card in cards
        ])
        self.out_dim = sum(m.embedding_dim for m in self.embs)

    def forward(self, x):
        return torch.cat([m(x[:, i]) for i, m in enumerate(self.embs)], 1)


class ExpandedMLP(nn.Module):
    def __init__(self, n_num, cards):
        super().__init__()
        self.emb = CatEmbedder(cards)
        d = n_num + self.emb.out_dim
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(.12),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(.12),
            nn.Linear(128, 1),
        )

    def forward(self, num, cat):
        return self.net(torch.cat([num, self.emb(cat)], 1)).squeeze(1)


def skill(y, p):
    brier = float(np.mean((y - p) ** 2))
    base = float(np.mean(y) * (1 - np.mean(y)))
    return 100000 * (1 - brier / base), brier


def run(season, seeds):
    raw = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"))
    tr, va = raw[raw.season < season].copy(), raw[raw.season == season].copy()
    xtr, xva = add_features(tr, va, season)
    ytr = tr.control_success.to_numpy(np.float32)
    yva = va.control_success.to_numpy(np.float32)
    cats = [c for c in xtr if c in ["top_bottom", "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "count_code", "platoon_matchup", "tkm_match", "count_x_base"]]
    nums = [c for c in xtr if c not in cats]
    mean = xtr[nums].to_numpy(np.float32).mean(0)
    std = xtr[nums].to_numpy(np.float32).std(0); std[std < 1e-6] = 1
    ntr = torch.tensor(np.nan_to_num((xtr[nums].to_numpy(np.float32)-mean)/std), dtype=torch.float32)
    nva = torch.tensor(np.nan_to_num((xva[nums].to_numpy(np.float32)-mean)/std), dtype=torch.float32)
    vocabs = [{v:i for i,v in enumerate(sorted(xtr[c].astype(str).unique()))} for c in cats]
    cards = [len(v)+1 for v in vocabs]
    def encode(x):
        return torch.tensor(np.stack([x[c].astype(str).map(v).fillna(len(v)).to_numpy(np.int64) for c,v in zip(cats,vocabs)], 1))
    ctr, cva = encode(xtr), encode(xva)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ntr, ctr, torch.tensor(ytr)), batch_size=2048, shuffle=True)
    out = np.zeros(len(va), dtype=np.float64)
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = ExpandedMLP(len(nums), cards)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-5)
        for epoch in range(8):
            model.train()
            for bn, bc, by in loader:
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(bn, bc), by)
                loss.backward(); opt.step()
            print(f"season={season} seed={seed} epoch={epoch+1}", flush=True)
        model.eval()
        with torch.no_grad(): out += torch.sigmoid(model(nva, cva)).numpy()
    return yva, out / len(seeds)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023]); args = ap.parse_args()
    result = {}
    for season in args.seasons:
        y, p_new = run(season, [7, 2025])
        base = np.load(os.path.join(ROOT, "scratch", "audit_v16_exact", f"val{season}.npz"))
        p_gbdt = .15*base["p_lgb"] + .75*base["p_cb"] + .10*base["p_xgb"]
        p_old = base["p_mlp"]
        rows = []
        for replace in [0, .25, .5, .75, 1]:
            p_mlp = (1-replace)*p_old + replace*p_new
            p = np.clip(.5 + 1.10*((1-.35)*p_gbdt + .35*p_mlp - .5) - .0045192086, 1e-6, 1-1e-6)
            sk, br = skill(y, p); rows.append({"replace":replace, "skill":sk, "brier":br})
        result[str(season)] = rows
        print(json.dumps({season:rows}, indent=2), flush=True)
    outpath = os.path.join(ROOT, "scratch", "expanded_mlp_exact_results.json")
    with open(outpath, "w") as f: json.dump(result, f, indent=2)
    print(outpath, flush=True)


if __name__ == "__main__":
    main()
