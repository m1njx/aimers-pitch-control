"""track1_model.py — TabR-lite retrieval-context builder, importable without side effects.
Uses dl_common.faiss_search (subprocess-isolated) instead of calling faiss directly in-process,
since faiss.search() segfaults when torch has already been imported in the same process
(macOS/arm64 OpenMP/BLAS conflict, not fixed by KMP_DUPLICATE_LIB_OK)."""
import numpy as np
import torch
from dl_common import faiss_search

K_NEIGHBORS = 32


def build_retrieval_context(num_tr, num_val, y_tr, k=K_NEIGHBORS):
    """Leave-one-out neighbor context for train rows; train-only neighbor pool for val rows.
    Returns context arrays to concatenate onto num_tr/num_val:
    [mean neighbor y, std neighbor y, inverse-distance-weighted mean neighbor y].

    Vectorized (no per-row Python loop): assumes the row's own self-match is the FIRST
    (distance=0) result of a k+1 search against itself, which holds for the vast majority
    of rows (only fails for exact-duplicate feature vectors, an acceptable approximation
    for this practical TabR-lite simplification — mismatch rate is logged)."""
    xb = np.ascontiguousarray(num_tr.numpy().astype(np.float32))
    y_np = y_tr.numpy() if torch.is_tensor(y_tr) else y_tr
    n = xb.shape[0]

    D_tr, I_tr = faiss_search(xb, xb, k + 1)  # search against itself, then drop self below
    self_mismatch = float(np.mean(I_tr[:, 0] != np.arange(n)))
    if self_mismatch > 0.01:
        print(f"[build_retrieval_context] WARNING: self-match-at-position-0 assumption "
              f"violated for {self_mismatch*100:.2f}% of rows (likely duplicate feature vectors)")

    neighbors = I_tr[:, 1:k + 1]  # drop assumed self (column 0), keep next k
    dists = D_tr[:, 1:k + 1]
    ny = y_np[neighbors]  # (N, k)
    ctx_tr = np.zeros((n, 3), dtype=np.float32)
    ctx_tr[:, 0] = ny.mean(axis=1)
    ctx_tr[:, 1] = ny.std(axis=1)
    w = 1.0 / (dists + 1e-3)
    ctx_tr[:, 2] = (ny * w).sum(axis=1) / w.sum(axis=1)

    xq = np.ascontiguousarray(num_val.numpy().astype(np.float32))
    D_val, I_val = faiss_search(xb, xq, k)
    ny_val = y_np[I_val]
    ctx_val = np.zeros((xq.shape[0], 3), dtype=np.float32)
    ctx_val[:, 0] = ny_val.mean(axis=1)
    ctx_val[:, 1] = ny_val.std(axis=1)
    w = 1.0 / (D_val + 1e-3)
    ctx_val[:, 2] = (ny_val * w).sum(axis=1) / w.sum(axis=1)

    return torch.tensor(ctx_tr, dtype=torch.float32), torch.tensor(ctx_val, dtype=torch.float32)
