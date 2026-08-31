"""
exp_embed.py — experiment: learned pitcher/batter entity embeddings in the neural component.

THE GAP THIS TARGETS
--------------------
Two facts that sit badly together:

  1. The pitcher effect is by far the dominant signal in this dataset. outputs/501
     measured the pitcher season-to-date channel at ~604 points on 2024, against
     ~49 for count, ~47 for batter, and exactly 0.000000 for pitcher x count.

  2. `pitcher_id` and `batter_id` are NOT model features. config.py excludes them
     as "IDs", so every component sees pitcher identity only through pre-aggregated
     asof_* rates. The model literally cannot tell two pitchers apart beyond their
     summary statistics.

Cardinality here is unusually friendly: 792 pitchers and 830 batters over 1.47M rows,
about 1,862 rows per pitcher. That is not a high-cardinality problem -- it is close to
the ideal regime for a learned embedding, where each entity has enough data to
estimate a real vector.

WHY THE EARLIER NEGATIVE RESULT DOES NOT APPLY
-----------------------------------------------
An earlier probe fed pitcher_id/batter_id to LightGBM as raw categoricals and the
score collapsed (519.7 -> 197.2 on 2024). That is a real result about GREEDY
CATEGORICAL SPLITTING, which shatters 792 levels into arbitrary subsets and overfits
immediately. A weight-decayed embedding trained by SGD is a different mechanism with
different failure modes: it shrinks unused dimensions toward zero rather than carving
the level set. The tree result is not evidence against the embedding.

Also relevant: the just-completed capacity sweep showed this problem punishes added
capacity hard (L3 seed_sd exploded 5.5 -> 93.1). So embeddings are deliberately kept
small and strongly regularised, and embedding weight decay is swept as its own axis
rather than assumed.

LEVELS
------
    E0  baseline MLP, no entity embeddings          <- already in harness/cache
    E1  pitcher embedding only,      dim 16, wd 1e-4
    E2  pitcher + batter embeddings, dim 16, wd 1e-4
    E3  pitcher + batter embeddings, dim 32, wd 1e-3  (higher capacity, harder shrinkage)

GBDT predictions are copied verbatim from the baseline cache: the entity embedding is
the single manipulated variable. Inner years only for selection.

    export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 KMP_DUPLICATE_LIB_OK=TRUE
    venv311/bin/python3 harness/exp_embed.py --levels E1 E2 E3
"""
import os, sys, time, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
LG = os.path.expanduser('~/LG_data')
sys.path.insert(0, os.path.join(LG, 'harness'))
sys.path.insert(0, os.path.join(LG, 'work/submit_v42'))
sys.path.insert(0, LG)

import torch
import torch.nn as nn

from build_cache import build_features, mlp_arrays, CACHE, CAT_COLS
from preprocessing import PitchPreprocessor
from agent2_asof_decomp2 import AsofDecomposer2

LEVELS = {
    # E1 came back +12.0 on inner with only 2 seeds (noise 19.4) -- formally a miss, but
    # positive in BOTH years and with the pitcher-only variant beating pitcher+batter,
    # which matches the measured 12:1 pitcher:batter signal ratio. Too coherent to
    # discard, too noisy to trust. E1b/E1c/E1d probe the neighbourhood; the real test is
    # rerunning E1 itself at 5 seeds, where the noise floor drops by ~sqrt(n).
    'E1': dict(use_pitcher=True,  use_batter=False, emb_dim=16, emb_wd=1e-4, epochs=6),
    'E1b': dict(use_pitcher=True, use_batter=False, emb_dim=8,  emb_wd=1e-4, epochs=6),
    'E1c': dict(use_pitcher=True, use_batter=False, emb_dim=24, emb_wd=1e-4, epochs=6),
    'E1d': dict(use_pitcher=True, use_batter=False, emb_dim=16, emb_wd=1e-3, epochs=6),
    'E2': dict(use_pitcher=True,  use_batter=True,  emb_dim=16, emb_wd=1e-4, epochs=6),
    'E3': dict(use_pitcher=True,  use_batter=True,  emb_dim=32, emb_wd=1e-3, epochs=6),
}


class CatEmbedder(nn.Module):
    def __init__(self, cards, emb_dim=8, max_emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(c, min(max_emb_dim, max(2, int(c ** 0.25 * emb_dim)))) for c in cards])
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x):
        if not len(self.embs):
            return torch.zeros(x.shape[0], 0)
        return torch.cat([e(x[:, i]) for i, e in enumerate(self.embs)], dim=1)


class EntityMLP(nn.Module):
    """Production SimpleMLP_MSE plus optional pitcher/batter embedding towers.

    The entity embeddings are returned as a separate parameter group so they can carry
    their own weight decay -- the whole point is to regularise identity harder than the
    rest of the network, since identity is where overfitting would show up first."""
    def __init__(self, num_dim, cards, n_pitcher=0, n_batter=0, emb_dim=16,
                 hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cards)
        self.pit_emb = nn.Embedding(n_pitcher, emb_dim) if n_pitcher else None
        self.bat_emb = nn.Embedding(n_batter, emb_dim) if n_batter else None
        for e in (self.pit_emb, self.bat_emb):
            if e is not None:
                nn.init.normal_(e.weight, std=0.01)   # start near zero: no prior belief
        extra = (emb_dim if n_pitcher else 0) + (emb_dim if n_batter else 0)
        layers, prev = [], num_dim + self.cat_embedder.out_dim + extra
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def entity_params(self):
        return [e.weight for e in (self.pit_emb, self.bat_emb) if e is not None]

    def other_params(self):
        ent = {id(p) for p in self.entity_params()}
        return [p for p in self.parameters() if id(p) not in ent]

    def forward(self, xn, xc, xp=None, xb=None):
        parts = [xn, self.cat_embedder(xc)]
        if self.pit_emb is not None:
            parts.append(self.pit_emb(xp))
        if self.bat_emb is not None:
            parts.append(self.bat_emb(xb))
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


def entity_index(train_ids, val_ids):
    """Vocabulary fitted on TRAIN ONLY; anything unseen at eval time maps to a shared
    UNK slot whose embedding is learned from nothing and stays near its zero init."""
    vocab = {v: i for i, v in enumerate(sorted(pd.unique(train_ids)))}
    unk = len(vocab)
    tr = np.array([vocab[v] for v in train_ids], dtype=np.int64)
    va = np.array([vocab.get(v, unk) for v in val_ids], dtype=np.int64)
    return tr, va, unk + 1


def run(df, year, seeds, level, out_dir):
    cfg = LEVELS[level]
    os.makedirs(out_dir, exist_ok=True)
    need = [s for s in seeds if not os.path.exists(os.path.join(out_dir, f'pred_{year}_{s}.npz'))]
    if not need:
        print(f'  {level} year={year}: 전부 캐시됨, 생략', flush=True)
        return

    tr = df[df.season < year]
    va = df[df.season == year]
    print(f'\n=== embed {level} eval={year}  train {len(tr):,}  cfg={cfg} ===', flush=True)
    t0 = time.time()

    prep = PitchPreprocessor()
    prep.fit(tr, as_of_season=year - 1, is_final=False,
             trackman_path=os.path.join(LG, 'open/data/trackman_history.csv'))
    bs = ((tr['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
          (tr['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cs = (tr['balls_before'].fillna(0).astype(int).astype(str) + '_' +
          tr['strikes_before'].fillna(0).astype(int).astype(str))
    cat_map = {v: i for i, v in enumerate((cs + '_' + bs).unique())}
    dec = AsofDecomposer2(); dec.fit(tr, val_season=year)

    _, Xtr133 = build_features(tr, prep, dec, cat_map)
    _, Xva133 = build_features(va, prep, dec, cat_map)
    ytr = tr['control_success'].values.astype(np.float64)
    nz_tr, ca_tr, art = mlp_arrays(Xtr133)
    nz_va, ca_va, _ = mlp_arrays(Xva133, art)

    pit_tr, pit_va, n_pit = entity_index(tr['pitcher_id'].values, va['pitcher_id'].values)
    bat_tr, bat_va, n_bat = entity_index(tr['batter_id'].values, va['batter_id'].values)
    unseen_p = float((pit_va == n_pit - 1).mean())
    unseen_b = float((bat_va == n_bat - 1).mean())
    print(f'  features built ({time.time()-t0:.0f}s)  투수 {n_pit-1}명 / 타자 {n_bat-1}명, '
          f'val 미학습 투수 {unseen_p:.2%} 타자 {unseen_b:.2%}', flush=True)

    Tn, Tc = torch.tensor(nz_tr), torch.tensor(ca_tr)
    Tp, Tb = torch.tensor(pit_tr), torch.tensor(bat_tr)
    Ty = torch.tensor(ytr, dtype=torch.float32)
    Vn, Vc = torch.tensor(nz_va), torch.tensor(ca_va)
    Vp, Vb = torch.tensor(pit_va), torch.tensor(bat_va)

    for seed in need:
        t1 = time.time()
        torch.manual_seed(seed)
        net = EntityMLP(len(art['num_cols']), art['cards'],
                        n_pitcher=n_pit if cfg['use_pitcher'] else 0,
                        n_batter=n_bat if cfg['use_batter'] else 0,
                        emb_dim=cfg['emb_dim'])
        opt = torch.optim.Adam([
            {'params': net.other_params(),  'weight_decay': 1e-5},
            {'params': net.entity_params(), 'weight_decay': cfg['emb_wd']},
        ], lr=1e-3)
        crit = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(Tn, Tc, Tp, Tb, Ty)
        dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
        net.train()
        for ep in range(cfg['epochs']):
            tot = 0.0
            for bn, bc, bp, bb, by in dl:
                opt.zero_grad()
                loss = crit(net(bn, bc, bp, bb), by)
                loss.backward(); opt.step()
                tot += float(loss) * len(by)
            print(f'    epoch {ep+1}/{cfg["epochs"]} mse={tot/len(ytr):.6f} ({time.time()-t1:.0f}s)', flush=True)
        net.eval()
        with torch.no_grad():
            mlp_pred = net(Vn, Vc, Vp, Vb).numpy().astype(np.float64)

        base = dict(np.load(os.path.join(CACHE, f'pred_{year}_{seed}.npz')))
        base['mlp'] = mlp_pred
        np.savez_compressed(os.path.join(out_dir, f'pred_{year}_{seed}.npz'), **base)
        print(f'  seed {seed}: {level} 완료 ({time.time()-t1:.0f}s)', flush=True)
    print(f'=== {level} {year} 완료 {(time.time()-t0)/60:.1f}분 ===', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2022, 2023])
    ap.add_argument('--seeds', type=int, nargs='+', default=[7, 123])
    ap.add_argument('--levels', nargs='+', default=['E1', 'E2', 'E3'])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(LG, 'open/data/train.csv'))
    df.columns = [c.replace('﻿', '') for c in df.columns]

    from exp_capacity import score_dir

    dirs = {'E0': CACHE}
    for lv in a.levels:
        d = os.path.join(LG, f'harness/cache_emb_{lv}')
        for y in a.years:
            run(df, y, a.seeds, lv, d)
        dirs[lv] = d

    print('\n=== 엔티티 임베딩 스캔 결과 (inner 전용) ===', flush=True)
    res = {}
    for lv, d in dirs.items():
        r = score_dir(d, a.years, a.seeds)
        if r:
            res[lv] = r
            print(f'  {lv}: inner={r["inner"]:8.1f}  연도별={ {k: round(v,1) for k,v in r["season_mean"].items()} }  seed_sd={r["seed_sd"]:.1f}', flush=True)
    if 'E0' in res and len(res) > 1:
        b = res['E0']['inner']
        noise = float(np.mean([v['seed_sd'] for v in res.values() if not np.isnan(v['seed_sd'])]))
        best = max(res, key=lambda k: res[k]['inner'])
        print(f'\n  → 최고={best}  E0(현행) 대비 델타={res[best]["inner"]-b:+.1f}  노이즈={noise:.1f}')
        print(f'  → 신뢰가능={bool(res[best]["inner"]-b > noise and best != "E0")}')
