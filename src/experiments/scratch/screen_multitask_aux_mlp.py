"""Inner-only screen of legal train-label auxiliary multitask learning.

Recovered post-pitch flags are used only as TRAINING TARGETS.  Inference takes
the same row-independent pre-pitch features as the production SimpleMLP and
returns only the control-success head.
"""
import json
import sys
import time

sys.path[:0] = ["~/LG_data/scratch", os.path.expanduser("~/LG_data")]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import dl_common as dlc
from agent2_asof_decomp2 import AsofDecomposer2
from agent2_recover_labels import recover
from core.eval_utils import calc_brier_skill_score
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor

DEVICE = torch.device("cpu")
SEEDS = [7, 123, 2025]
AUX_COLS = ["lab_reverse", "lab_middle", "lab_ball", "lab_strike",
            "lab_fastball", "lab_breaking", "lab_offspeed"]
OUT = "~/LG_data/scratch/multitask_aux_mlp_results.json"
PRED_DIR = "~/LG_data/scratch/multitask_aux_preds"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def frames(df, fold):
    tr = df.iloc[fold.train_idx].copy(); va = df.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor().fit(tr, as_of_season=fold.fold_max_season, is_final=False)
    xt, xv = prep.transform(tr), prep.transform(va)
    dlc.add_count_x_base(tr, xt); dlc.add_count_x_base(va, xv)
    mp = {v: i for i, v in enumerate(xt["count_x_base"].unique())}
    xt["count_x_base"] = xt["count_x_base"].map(mp).fillna(-1).astype(int)
    xv["count_x_base"] = xv["count_x_base"].map(mp).fillna(-1).astype(int)
    dec = AsofDecomposer2().fit(tr, fold.val_season)
    at, av = dec.transform(tr), dec.transform(va)
    at.index, av.index = xt.index, xv.index
    return tr, va, pd.concat([xt, at], axis=1), pd.concat([xv, av], axis=1)


class MultiTaskMLP(nn.Module):
    def __init__(self, num_dim, cards):
        super().__init__()
        self.emb = dlc.CatEmbedder(cards)
        self.trunk = nn.Sequential(nn.Linear(num_dim + self.emb.out_dim, 128), nn.ReLU(),
                                   nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU(),
                                   nn.Dropout(.15))
        self.main = nn.Linear(64, 1)
        self.aux = nn.Linear(64, len(AUX_COLS))

    def forward(self, xn, xc):
        h = self.trunk(torch.cat([xn, self.emb(xc)], dim=1))
        return self.main(h).squeeze(1), self.aux(h)


def train_predict(tens, y, aux, aux_weight, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiTaskMLP(tens["num_tr"].shape[1], tens["cat_cardinalities"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    n = len(y); rng = np.random.RandomState(seed + 1); order = rng.permutation(n)
    dev = order[:int(.05*n)]; train = order[int(.05*n):]
    y_t = torch.tensor(y, dtype=torch.float32)
    a_t = torch.tensor(np.nan_to_num(aux, nan=0.0), dtype=torch.float32)
    m_t = torch.tensor(np.isfinite(aux), dtype=torch.float32)
    best, state, bad = 1e9, None, 0
    for epoch in range(8):
        model.train(); perm = train[np.random.permutation(len(train))]
        for start in range(0, len(perm), 8192):
            ix = perm[start:start+8192]
            main, other = model(tens["num_tr"][ix], tens["cat_tr"][ix])
            loss = bce(main, y_t[ix])
            if aux_weight:
                raw = nn.functional.binary_cross_entropy_with_logits(other, a_t[ix], reduction="none")
                loss = loss + aux_weight * (raw*m_t[ix]).sum()/m_t[ix].sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            main, _ = model(tens["num_tr"][dev], tens["cat_tr"][dev])
            dl = bce(main, y_t[dev]).item()
            log(f"seed={seed} aux_w={aux_weight} epoch={epoch+1} dev_main_bce={dl:.6f}")
        if dl < best-1e-5:
            best=dl; state={k:v.clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad += 1
            if bad >= 2: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        pdv = torch.sigmoid(model(tens["num_tr"][dev], tens["cat_tr"][dev])[0]).numpy()
        pva = torch.sigmoid(model(tens["num_val"], tens["cat_val"])[0]).numpy()
    shift = dlc.search_best_shift(y[dev], pdv)
    return np.clip(pva + shift, 1e-6, 1-1e-6), float(shift)


def main():
    df = pd.read_csv(config.TRAIN_PATH); labels = recover(df); result = {}
    import os
    os.makedirs(PRED_DIR, exist_ok=True)
    folds = get_cv_folds(df)
    for fold in folds:
        tr, va, xt, xv = frames(df, fold); tens = dlc.to_tensors(xt, xv)
        ytr = tr[config.TARGET_COL].to_numpy(np.float32); yva = va[config.TARGET_COL].to_numpy(np.float32)
        aux = labels.iloc[fold.train_idx][AUX_COLS].to_numpy(np.float32)
        result[str(fold.val_season)] = {}
        for weight in (1.0,):
            predictions, shifts, seed_skills = [], [], []
            for seed in SEEDS:
                p, shift = train_predict(tens, ytr, aux, weight, seed)
                predictions.append(p); shifts.append(shift)
                seed_skills.append(calc_brier_skill_score(yva, p)[0])
            p = np.mean(predictions, axis=0)
            skill, brier, _, _ = calc_brier_skill_score(yva, p)
            result[str(fold.val_season)][str(weight)] = {
                "skill":skill,"brier":brier,"shifts":shifts,"seed_skills":seed_skills}
            np.savez_compressed(
                f"{PRED_DIR}/val{fold.val_season}.npz",
                y=yva.astype(np.int8), p=p.astype(np.float32),
                **{f"p_seed_{seed}": pred.astype(np.float32)
                   for seed, pred in zip(SEEDS, predictions)})
            log(f"RESULT val={fold.val_season} aux_w={weight}: bagged_skill={skill:.2f} seeds={seed_skills}")
        with open(OUT,"w") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
