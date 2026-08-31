"""
dl_common.py — 딥러닝 4개 트랙(TabR/ModernNCA/TabM/PLE-MLP) 공용 유틸리티.

시간 예산상 5-seed가 아닌 2-seed 배깅을 표준으로 사용 (사용자 지시의 "2시간 초과 시 현실적으로
조정" 조항 적용, 각 보고서에 명시). strict_as_of=True, 동일 70개 피처(count_x_base 포함) 사용.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
TARGET = config.TARGET_COL

_FAISS_WORKER = '~/LG_data/scratch/faiss_search_worker.py'
_faiss_call_counter = [0]


def faiss_search(candidates, queries, k):
    """faiss L2 search via a fully separate subprocess (no torch in that process) —
    calling faiss.search() directly in a process where torch is already imported
    segfaults on this macOS/arm64 setup (OpenMP/BLAS conflict, not fixed by
    KMP_DUPLICATE_LIB_OK). Returns (D, I) numpy arrays, same as faiss.Index.search."""
    import subprocess
    _faiss_call_counter[0] += 1
    tag = _faiss_call_counter[0]
    cand_path = f'/tmp/_faiss_cand_{tag}.npy'
    query_path = f'/tmp/_faiss_query_{tag}.npy'
    out_path = f'/tmp/_faiss_out_{tag}.npz'
    np.save(cand_path, candidates.astype(np.float32))
    np.save(query_path, queries.astype(np.float32))
    subprocess.run([sys.executable, _FAISS_WORKER, cand_path, query_path, str(k), out_path],
                    check=True, capture_output=True, text=True)
    result = np.load(out_path)
    D, I = result['D'], result['I']
    for p in (cand_path, query_path, out_path):
        try:
            os.remove(p)
        except OSError:
            pass
    return D, I

CAT_COLS_BASE = config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS + [config.TRACKMAN_MATCH_FLAG_COL, 'count_x_base']

DEBIASED_SEEDS_FULL = [7, 123, 2025, 31415, 8675309]
DL_SEEDS = [7, 123]  # 2-seed subset for DL tracks (time budget)


def add_count_x_base(df_src, X_dst):
    b_str = ((df_src['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
             (df_src['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
             (df_src['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    c_str = (df_src['balls_before'].fillna(0).astype(int).astype(str) + '_' +
             df_src['strikes_before'].fillna(0).astype(int).astype(str))
    X_dst['count_x_base'] = (c_str + '_' + b_str)
    return X_dst


def build_fold_frames(df_train, fold):
    """Matches the GBDT pipeline exactly: PitchPreprocessor + count_x_base."""
    df_tr_f = df_train.iloc[fold.train_idx].copy()
    df_val_f = df_train.iloc[fold.val_idx].copy()
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold.fold_max_season, is_final=False)
    X_tr_f = prep.transform(df_tr_f)
    X_val_f = prep.transform(df_val_f)
    add_count_x_base(df_tr_f, X_tr_f)
    add_count_x_base(df_val_f, X_val_f)
    cat_map = {v: i for i, v in enumerate(X_tr_f['count_x_base'].unique())}
    X_tr_f['count_x_base'] = X_tr_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    X_val_f['count_x_base'] = X_val_f['count_x_base'].map(cat_map).fillna(-1).astype(int)
    y_tr_f = df_tr_f[TARGET].values.astype(np.float32)
    y_val_f = df_val_f[TARGET].values.astype(np.float32)
    return X_tr_f, X_val_f, y_tr_f, y_val_f


def to_tensors(X_tr_f, X_val_f, cat_cols=None):
    """Numeric -> z-scored float32 tensor. Categorical -> factorized int64 tensor
    (train-only vocab, unseen val categories map to a reserved 'unknown' index)."""
    if cat_cols is None:
        cat_cols = [c for c in CAT_COLS_BASE if c in X_tr_f.columns]
    num_cols = [c for c in X_tr_f.columns if c not in cat_cols]

    num_tr_raw = X_tr_f[num_cols].astype(np.float32).values
    num_val_raw = X_val_f[num_cols].astype(np.float32).values
    mean = np.nanmean(num_tr_raw, axis=0)
    std = np.nanstd(num_tr_raw, axis=0)
    std[std < 1e-8] = 1.0
    num_tr = np.nan_to_num((num_tr_raw - mean) / std, nan=0.0)
    num_val = np.nan_to_num((num_val_raw - mean) / std, nan=0.0)
    num_tr_raw = np.nan_to_num(num_tr_raw, nan=np.nanmedian(num_tr_raw))
    num_val_raw = np.nan_to_num(num_val_raw, nan=np.nanmedian(num_tr_raw))

    cat_cardinalities = []
    cat_tr_cols, cat_val_cols = [], []
    for c in cat_cols:
        train_vals = X_tr_f[c].astype(str)
        vocab = {v: i for i, v in enumerate(pd.unique(train_vals))}
        unk_idx = len(vocab)
        cat_tr_cols.append(train_vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
        val_vals = X_val_f[c].astype(str)
        cat_val_cols.append(val_vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
        cat_cardinalities.append(unk_idx + 1)

    cat_tr = np.stack(cat_tr_cols, axis=1) if cat_tr_cols else np.zeros((len(X_tr_f), 0), dtype=np.int64)
    cat_val = np.stack(cat_val_cols, axis=1) if cat_val_cols else np.zeros((len(X_val_f), 0), dtype=np.int64)

    return dict(
        num_tr=torch.tensor(num_tr, dtype=torch.float32),
        num_val=torch.tensor(num_val, dtype=torch.float32),
        num_tr_raw=torch.tensor(num_tr_raw, dtype=torch.float32),
        num_val_raw=torch.tensor(num_val_raw, dtype=torch.float32),
        cat_tr=torch.tensor(cat_tr, dtype=torch.int64),
        cat_val=torch.tensor(cat_val, dtype=torch.int64),
        cat_cardinalities=cat_cardinalities,
        num_cols=num_cols, cat_cols=cat_cols,
    )


def make_dev_split(n, dev_frac=0.05, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_dev = int(n * dev_frac)
    return idx[n_dev:], idx[:n_dev]  # train_idx, dev_idx


def search_best_shift(y_dev, p_dev, grid=None):
    if grid is None:
        grid = np.linspace(-0.05, 0.05, 41)
    best_s, best_b = 0.0, calc_raw_brier(y_dev, np.clip(p_dev, 1e-6, 1 - 1e-6))
    for s in grid:
        p_shift = np.clip(p_dev + s, 1e-6, 1 - 1e-6)
        b = calc_raw_brier(y_dev, p_shift)
        if b < best_b:
            best_b, best_s = b, s
    return best_s


def train_generic(model, num_tr, cat_tr, y_tr, epochs, lr, batch_size, device, weight_decay=1e-5, verbose_prefix=""):
    """Standard training loop: BCEWithLogits + Adam, with an internal dev-split
    carved from num_tr/cat_tr/y_tr for early stopping + post-hoc shift search."""
    n = len(y_tr)
    tr_idx, dev_idx = make_dev_split(n, dev_frac=0.05, seed=1)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    num_tr_t, cat_tr_t, y_tr_t = num_tr[tr_idx], cat_tr[tr_idx], y_tr[tr_idx]
    num_dev_t, cat_dev_t, y_dev_np = num_tr[dev_idx], cat_tr[dev_idx], y_tr[dev_idx].numpy()

    n_train = len(tr_idx)
    best_dev_loss = float('inf')
    best_state = None
    patience, bad_epochs = 2, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb_num = num_tr_t[idx].to(device)
            xb_cat = cat_tr_t[idx].to(device)
            yb = y_tr_t[idx].to(device)
            opt.zero_grad()
            logits = model(xb_num, xb_cat)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            dev_logits = model(num_dev_t.to(device), cat_dev_t.to(device)).cpu()
            dev_loss = loss_fn(dev_logits, torch.tensor(y_dev_np)).item()
        print(f"{verbose_prefix}Epoch {epoch+1}/{epochs}: train_loss={train_loss:.5f} dev_loss={dev_loss:.5f}")

        if dev_loss < best_dev_loss - 1e-5:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{verbose_prefix}Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p_dev = torch.sigmoid(model(num_dev_t.to(device), cat_dev_t.to(device))).cpu().numpy()
    shift = search_best_shift(y_dev_np, p_dev)
    return model, shift


def predict(model, num_x, cat_x, device, shift, batch_size=32768):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(num_x), batch_size):
            xb_num = num_x[i:i + batch_size].to(device)
            xb_cat = cat_x[i:i + batch_size].to(device)
            logits = model(xb_num, xb_cat)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(preds)
    return np.clip(p + shift, 1e-6, 1 - 1e-6)


class CatEmbedder(nn.Module):
    """Concatenated embeddings for a set of categorical columns."""
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


class SimpleMLP(nn.Module):
    """Baseline MLP: z-scored numeric + categorical embeddings -> MLP head."""
    def __init__(self, num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


class PLEEncoder(nn.Module):
    """Piecewise Linear Encoding for numeric features (Gorishniy et al. style).
    Bin boundaries are precomputed (quantile-based) and passed in as a buffer;
    this module just does the piecewise-linear lookup, differentiable end-to-end
    only through the linear projection that follows (encoding itself is a fixed
    deterministic transform of the input value, matching the paper's PLE-Q variant)."""
    def __init__(self, bin_edges_list):
        # bin_edges_list: list of 1D np arrays, one per numeric feature, each length n_bins+1
        super().__init__()
        self.n_features = len(bin_edges_list)
        self.n_bins = bin_edges_list[0].shape[0] - 1 if self.n_features > 0 else 0
        max_bins = max((e.shape[0] - 1 for e in bin_edges_list), default=1)
        self.n_bins = max_bins
        padded = np.zeros((self.n_features, max_bins + 1), dtype=np.float32)
        for i, e in enumerate(bin_edges_list):
            nb = e.shape[0] - 1
            padded[i, :nb + 1] = e
            if nb < max_bins:
                padded[i, nb + 1:] = e[-1]
        self.register_buffer('edges', torch.tensor(padded, dtype=torch.float32))  # (F, B+1)
        self.out_dim = self.n_features * max_bins

    def forward(self, x_num):
        # x_num: (N, F) raw (NOT z-scored) numeric values
        edges = self.edges  # (F, B+1)
        x = x_num.unsqueeze(-1)  # (N, F, 1)
        left = edges[:, :-1].unsqueeze(0)   # (1, F, B)
        right = edges[:, 1:].unsqueeze(0)   # (1, F, B)
        width = (right - left).clamp(min=1e-6)
        frac = ((x - left) / width).clamp(0.0, 1.0)  # (N, F, B): fraction within each bin
        is_between = ((x >= left) & (x < right)).float()
        is_above = (x >= right).float()
        # bins fully below x -> 1.0; the bin containing x -> fractional position; bins above -> 0.0
        enc = is_above * 1.0 + is_between * frac
        return enc.reshape(x_num.shape[0], -1)


def compute_ple_bin_edges(num_tr_raw, n_bins=16):
    """Quantile-based bin edges per numeric feature, computed from TRAIN only."""
    edges = []
    for j in range(num_tr_raw.shape[1]):
        col = num_tr_raw[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            edges.append(np.array([0.0, 1.0], dtype=np.float32))
            continue
        qs = np.linspace(0, 1, n_bins + 1)
        e = np.unique(np.quantile(col, qs)).astype(np.float32)
        if len(e) < 2:
            e = np.array([col.min(), col.max() + 1e-6], dtype=np.float32)
        edges.append(e)
    return edges


def run_dl_track(model_factory, track_name, epochs=10, lr=1e-3, batch_size=8192, seeds=None,
                  log_fn=print, use_raw_num=False, extra_tens_fn=None):
    """Full 3-fold x N-seed bagging harness for a DL track.
    model_factory(num_dim, cat_cardinalities, seed, tens) -> nn.Module
    use_raw_num: if True, pass RAW (unscaled) numeric tensors to the model (for PLE-style
    encoders that need raw values, not z-scored ones).
    extra_tens_fn(tens) -> dict of extra kwargs merged into model_factory call, for tracks
    that need extra precomputed artifacts (e.g. PLE bin edges) per fold.
    Returns dict(mean_skill, overall_brier, fold_details, oof, val_idx_all)
    """
    if seeds is None:
        seeds = DL_SEEDS
    df_train = pd.read_csv(config.TRAIN_PATH)
    folds = get_cv_folds(df_train)
    val_idx_all = np.concatenate([f.val_idx for f in folds])
    oof = np.zeros(len(df_train))
    fold_details = []
    per_seed_fold_skills = []  # individual (unbagged) seed x fold results, for stability/collapse checks

    for k, fold in enumerate(folds):
        t0 = time.time()
        X_tr_f, X_val_f, y_tr_f, y_val_f = build_fold_frames(df_train, fold)
        tens = to_tensors(X_tr_f, X_val_f)
        if use_raw_num:
            num_tr, num_val = tens['num_tr_raw'], tens['num_val_raw']
        else:
            num_tr, num_val = tens['num_tr'], tens['num_val']
        cat_tr, cat_val = tens['cat_tr'], tens['cat_val']
        y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)
        num_dim = num_tr.shape[1]
        cat_cardinalities = tens['cat_cardinalities']
        extra_kwargs = extra_tens_fn(tens) if extra_tens_fn is not None else {}

        p_sum = np.zeros(len(y_val_f))
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_factory(num_dim, cat_cardinalities, seed, **extra_kwargs)
            model, shift = train_generic(model, num_tr, cat_tr, y_tr_t, epochs, lr, batch_size, DEVICE,
                                          verbose_prefix=f"[{track_name} fold{k+1} seed{seed}] ")
            p_val = predict(model, num_val, cat_val, DEVICE, shift)
            p_sum += p_val

            sk_seed, br_seed, _, _ = calc_brier_skill_score(y_val_f, p_val)
            per_seed_fold_skills.append({'fold': k + 1, 'val_season': fold.val_season, 'seed': seed,
                                          'skill_k': sk_seed, 'raw_brier_k': br_seed})
            collapse_flag = " ⚠️ COLLAPSE (skill<=0)" if sk_seed <= 0 else ""
            log_fn(f"[{track_name}] Fold {k+1} seed={seed} done: skill={sk_seed:.2f}{collapse_flag} "
                   f"({time.time()-t0:.1f}s cumulative)")

        p_bagged = p_sum / len(seeds)
        oof[fold.val_idx] = p_bagged
        sk, br, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})
        log_fn(f"[{track_name}] === Fold {k+1} ({fold.val_season}) COMPLETE (bagged): Skill={sk:.2f} "
               f"Brier={br:.6f} ({time.time()-t0:.1f}s) ===")

    mean_skill = evaluate_fold_skills(fold_details)
    overall_brier = calc_raw_brier(df_train.iloc[val_idx_all][TARGET].values, oof[val_idx_all])
    n_collapsed = sum(1 for r in per_seed_fold_skills if r['skill_k'] <= 0)
    if n_collapsed > 0:
        log_fn(f"[{track_name}] ⚠️⚠️ {n_collapsed}/{len(per_seed_fold_skills)} seed x fold runs COLLAPSED (skill<=0)!")
    return dict(mean_skill=mean_skill, overall_brier=overall_brier, fold_details=fold_details,
                oof=oof, val_idx_all=val_idx_all, df_train=df_train,
                per_seed_fold_skills=per_seed_fold_skills)


def analyze_vs_gbdt(dl_result, gbdt_ref_path='/tmp/gbdt_reference_oof.npz'):
    """Correlation + ensemble-value grid search against the cached GBDT reference OOF.
    Skill is computed the PROJECT-STANDARD way (per-fold, then averaged) — never pooled."""
    ref = np.load(gbdt_ref_path)
    gbdt_val_idx = ref['val_idx']
    p_gbdt_full = np.full(int(gbdt_val_idx.max()) + 1, np.nan)
    p_gbdt_full[gbdt_val_idx] = ref['p_ens']

    val_idx_all = dl_result['val_idx_all']
    df_train = dl_result['df_train']
    p_dl_full = dl_result['oof']

    folds = get_cv_folds(df_train)
    y_full = df_train[TARGET].values

    corr = float(np.corrcoef(p_dl_full[val_idx_all], p_gbdt_full[val_idx_all])[0, 1])

    best_w, best_skill = 0.0, None
    grid_results = []
    for w in np.linspace(0, 0.5, 26):  # DL weight 0..0.5 in 0.02 steps
        fold_skills = []
        for fold in folds:
            vi = fold.val_idx
            p_blend = np.clip((1 - w) * p_gbdt_full[vi] + w * p_dl_full[vi], 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(y_full[vi], p_blend)
            fold_skills.append(sk)
        mean_sk = float(np.mean(fold_skills))
        grid_results.append((float(w), mean_sk))
        if best_skill is None or mean_sk > best_skill:
            best_skill, best_w = mean_sk, float(w)

    gbdt_only_skill = float(ref['skill'])
    return dict(corr=corr, best_w=best_w, best_skill=best_skill,
                gbdt_only_skill_reference=gbdt_only_skill, grid_results=grid_results)


def analyze_vs_gbdt_nested(dl_result, gbdt_ref_path='/tmp/gbdt_reference_oof.npz'):
    """Nested-honest version of analyze_vs_gbdt: select the blend weight w using ONLY
    inner folds (2022, 2023), then apply that frozen w to the full 3-fold set (incl.
    outer 2024) for a genuinely out-of-sample evaluation. Mirrors 144번's methodology."""
    ref = np.load(gbdt_ref_path)
    gbdt_val_idx = ref['val_idx']
    p_gbdt_full = np.full(int(gbdt_val_idx.max()) + 1, np.nan)
    p_gbdt_full[gbdt_val_idx] = ref['p_ens']

    df_train = dl_result['df_train']
    p_dl_full = dl_result['oof']
    folds = get_cv_folds(df_train)
    y_full = df_train[TARGET].values
    inner_folds = [f for f in folds if f.val_season in (2022, 2023)]

    def fold_skill_for_w(w, fold_list):
        skills = []
        for fold in fold_list:
            vi = fold.val_idx
            p_blend = np.clip((1 - w) * p_gbdt_full[vi] + w * p_dl_full[vi], 1e-6, 1 - 1e-6)
            sk, _, _, _ = calc_brier_skill_score(y_full[vi], p_blend)
            skills.append(sk)
        return float(np.mean(skills))

    best_w_inner, best_inner_skill = 0.0, -1
    for w in np.linspace(0, 0.5, 26):
        sk = fold_skill_for_w(w, inner_folds)
        if sk > best_inner_skill:
            best_inner_skill, best_w_inner = sk, float(w)

    honest_full_skill = fold_skill_for_w(best_w_inner, folds)
    outer_fold = [f for f in folds if f.val_season == 2024][0]
    outer_only_skill = fold_skill_for_w(best_w_inner, [outer_fold])

    # circular (for comparison, matching the original analyze_vs_gbdt logic)
    best_w_circular, best_circular_skill = 0.0, -1
    for w in np.linspace(0, 0.5, 26):
        sk = fold_skill_for_w(w, folds)
        if sk > best_circular_skill:
            best_circular_skill, best_w_circular = sk, float(w)

    return dict(
        best_w_inner=best_w_inner, best_inner_skill=best_inner_skill,
        honest_full_skill=honest_full_skill, outer_only_skill=outer_only_skill,
        best_w_circular=best_w_circular, best_circular_skill=best_circular_skill,
        gbdt_only_skill_reference=float(ref['skill']),
        circularity_gap=best_circular_skill - honest_full_skill,
    )
