#!/usr/bin/env python3
"""hunt_multitask.py — Hypothesis U: auxiliary outcome supervision.

WHERE THE HYPOTHESIS COMES FROM (a measurement, not an invention)
-----------------------------------------------------------------
train.csv is exactly chronological within (pitcher_id, season): the diff of
`asof_pitcher_n` inside every such group is 1.0 for 1,472,832 of 1,475,092 rows and
NaN (group boundary) for the rest -- never anything else.  The asof rate columns are
therefore cumulative counters, and

    label_i = n_{i+1} * rate_{i+1} - n_i * rate_i

recovers the per-pitch outcome flag EXACTLY.  Verified on the one flag we can check:
the recovered success flag equals `control_success` on 1,472,040/1,472,040 rows
(match 1.000000, max |residual| before rounding 0.0146).

That gives four extra per-pitch binary labels the pipeline has never used:
`reverse`, `middle`, `ball`, `strike` -- rates .2291 / .1496 / .3698 / .4436.
They are not redundant with the target: succ=1 implies rev=0 and mid=0, but the
NEGATIVE class splits into four heterogeneous subclasses (rev only 287,063; mid only
170,000; both 50,208; neither 194,010), and the orthogonal zone class carries
P(succ | ball)=.424 vs P(succ | strike)=.588 vs P(succ | neither)=.567.

WHY THIS IS A DIFFERENT AXIS
----------------------------
All 17 closed axes changed the FEATURES, the MODEL CLASS, the BLEND or the
POST-HOC CALIBRATION.  None of them changed the TARGET.  This adds supervision,
which is the one thing feature engineering cannot buy: it constrains the shared
representation with 4 labels that have far more signal than control_success.
Distinct from both jobs running concurrently (v69 features; causal in-game
sequence teacher/student).

RULE 4 (row independence)
-------------------------
The auxiliary labels are computed from TRAIN rows only and are used only as
training targets.  At inference the network is read at head 0 exactly as today; no
test row, and no statistic of the test batch, enters any computation.  The label
of train row i uses train row i+1 -- both are train rows, both are in the past
relative to the evaluation season of every fold.

EFFECTIVE WEIGHT
----------------
COMPONENT = mlp, effective weight 0.50 -> an MLP-alone effect of +X is worth
+0.50X overall.  To clear the LB noise floor (12 pts) the MLP-alone effect must be
at least +24.  Reported both ways.

PRE-REGISTERED PASS CRITERION (fixed before the first run, not revisited)
------------------------------------------------------------------------
inner folds 2021/2022/2023 x seeds 7,123,2025,31415,8675309, paired (fold,seed)
cells, scored with production-identical prediction bagging:
    ALL THREE fold means positive  AND  t > 2.5  over the 15 cells.
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
import build_cache as bc
import exp_template as ET          # reuse the protocol's scoring verbatim

BASE = os.path.join(LG, 'harness/cache')
FOLDS = [2021, 2022, 2023]
SEEDS = [7, 123, 2025, 31415, 8675309]
AUX = [('asof_pitcher_reverse_rate', 'rev'), ('asof_pitcher_middle_rate', 'mid'),
       ('asof_pitcher_ball_rate', 'ball'), ('asof_pitcher_strike_rate', 'strike')]


def recover_aux(df):
    """Per-pitch auxiliary labels by differencing the asof counters. Train rows only."""
    g = df.groupby(['pitcher_id', 'season'], sort=False)
    n = df['asof_pitcher_n'].astype(np.float64)
    n_next = g['asof_pitcher_n'].shift(-1).astype(np.float64)
    out, w = {}, None
    for col, name in AUX:
        r = df[col].astype(np.float64)
        lab = n_next * g[col].shift(-1).astype(np.float64) - n * r
        m = lab.notna().values
        w = m if w is None else (w & m)
        out[name] = np.clip(np.nan_to_num(np.round(lab.values), nan=0.0), 0.0, 1.0)
    A = np.stack([out[nm] for _, nm in AUX], 1).astype(np.float32)
    return A, w.astype(np.float32)


class MultiMLP(nn.Module):
    """Identical trunk to build_cache.SimpleMLP_MSE; head widened 1 -> 1+len(AUX)."""
    def __init__(self, num_dim, cards, n_aux, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = bc.CatEmbedder(cards)
        layers, prev = [], num_dim + self.cat_embedder.out_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1 + n_aux), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, xn, xc):
        return self.net(torch.cat([xn, self.cat_embedder(xc)], dim=1))


def train_mlp(seed, nz_tr, ca_tr, ytr, aux_tr, wt_tr, nz_va, ca_va, cards, lam):
    torch.manual_seed(seed)
    net = MultiMLP(nz_tr.shape[1], cards, aux_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(nz_tr), torch.tensor(ca_tr),
        torch.tensor(ytr, dtype=torch.float32),
        torch.tensor(aux_tr), torch.tensor(wt_tr))
    dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
    net.train()
    for _ in range(5):
        for bn, bcat, by, ba, bw in dl:
            opt.zero_grad()
            o = net(bn, bcat)
            loss = ((o[:, 0] - by) ** 2).mean()
            if lam > 0:
                aux = ((o[:, 1:] - ba) ** 2).mean(1)
                loss = loss + lam * (aux * bw).sum() / bw.sum().clamp(min=1.0)
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(nz_va), torch.tensor(ca_va))[:, 0].numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--years', type=int, nargs='+', default=FOLDS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    a = ap.parse_args()
    print(__doc__)
    print(f'\n>>> lambda = {a.lam}   tag = {a.tag}')
    print('>>> effective weight mlp = 0.50 -> MLP-alone must reach +24 to clear the '
          'LB noise floor of 12\n', flush=True)

    t0 = time.time()
    torch.set_num_threads(2)
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]
    AUXL, AUXW = recover_aux(df)
    print(f'aux labels recovered: rates {AUXL.mean(0).round(4).tolist()}  '
          f'usable {AUXW.mean():.5f}  ({time.time()-t0:.0f}s)', flush=True)

    cdir = os.path.join(LG, f'harness/cache_{a.tag}')
    os.makedirs(cdir, exist_ok=True)
    for y in a.years:
        need = [s for s in a.seeds if not os.path.exists(os.path.join(cdir, f'pred_{y}_{s}.npz'))]
        if not need:
            print(f'=== eval {y}: all seeds cached, skip ===', flush=True); continue
        past, va, prep, dec, cat_map = ET.fold_data(df, y)
        pidx, vidx = past.index.values, va.index.values
        _, Xpa133 = bc.build_features(past, prep, dec, cat_map)
        _, Xva133 = bc.build_features(va, prep, dec, cat_map)
        ypa = past['control_success'].values.astype(np.float64)
        nz_tr, ca_tr, art = bc.mlp_arrays(Xpa133)
        nz_va, ca_va, _ = bc.mlp_arrays(Xva133, art)
        del Xpa133, Xva133
        aux_tr, wt_tr = AUXL[pidx], AUXW[pidx]
        print(f'\n=== eval {y}: past {len(past):,} val {len(va):,}  aux cover '
              f'{wt_tr.mean():.5f}  ({time.time()-t0:.0f}s) ===', flush=True)
        for s in need:
            t1 = time.time()
            src = dict(np.load(os.path.join(BASE, f'pred_{y}_{s}.npz')))
            out = dict(src)
            out['mlp'] = train_mlp(s, nz_tr, ca_tr, ypa, aux_tr, wt_tr,
                                   nz_va, ca_va, art['cards'], a.lam)
            np.savez_compressed(os.path.join(cdir, f'pred_{y}_{s}.npz'), **out)
            yv = np.load(os.path.join(BASE, f'y_{y}.npy'))
            from evaluate import skill
            print(f'  seed {s}: done ({time.time()-t1:.0f}s)  MLP-alone '
                  f'{skill(src["mlp"], yv):.1f} -> {skill(out["mlp"], yv):.1f}', flush=True)
        del nz_tr, ca_tr, nz_va, ca_va, past, va

    ET.score(cdir)
    print(f'\n총 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
