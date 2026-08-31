import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch

import config
from cv_utils import get_cv_folds
from dl_common import (build_fold_frames, to_tensors, train_generic, predict, SimpleMLP,
                        PLEEncoder, compute_ple_bin_edges, CatEmbedder, DEVICE)

print(f"Device: {DEVICE}")

df_train = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train)
fold = folds[0]

# Subsample for smoke test speed
sub_train_idx = np.random.RandomState(0).choice(fold.train_idx, size=5000, replace=False)
sub_val_idx = np.random.RandomState(1).choice(fold.val_idx, size=1000, replace=False)

class FakeFold:
    train_idx = sub_train_idx
    val_idx = sub_val_idx
    fold_max_season = fold.fold_max_season
    val_season = fold.val_season

X_tr_f, X_val_f, y_tr_f, y_val_f = build_fold_frames(df_train, FakeFold())
print(f"X_tr_f shape: {X_tr_f.shape}, X_val_f shape: {X_val_f.shape}")

tens = to_tensors(X_tr_f, X_val_f)
print(f"num_tr: {tens['num_tr'].shape}, cat_tr: {tens['cat_tr'].shape}, cardinalities: {tens['cat_cardinalities']}")

y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)

# Test SimpleMLP
print("\n--- SimpleMLP smoke test ---")
model = SimpleMLP(tens['num_tr'].shape[1], tens['cat_cardinalities'])
model, shift = train_generic(model, tens['num_tr'], tens['cat_tr'], y_tr_t, epochs=2, lr=1e-3,
                              batch_size=512, device=DEVICE, verbose_prefix="[smoke] ")
p = predict(model, tens['num_val'], tens['cat_val'], DEVICE, shift)
print(f"SimpleMLP OK, pred shape={p.shape}, mean={p.mean():.4f}")

# Test PLE
print("\n--- PLE smoke test ---")
edges = compute_ple_bin_edges(tens['num_tr_raw'].numpy(), n_bins=8)
ple = PLEEncoder(edges)
out = ple(tens['num_tr_raw'][:10])
print(f"PLE output shape: {out.shape} (expect 10 x {ple.out_dim})")

# Test TabM
print("\n--- TabM smoke test ---")
sys.path.insert(0, '~/LG_data/scratch')
from track3_model import TabM
tabm = TabM(tens['num_tr'].shape[1], tens['cat_cardinalities'], seed=0, k_members=4)
out = tabm(tens['num_tr'][:16], tens['cat_tr'][:16])
print(f"TabM output shape: {out.shape} (expect 16,)")
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y_tr_t[:16])
loss.backward()
print(f"TabM backward OK, loss={loss.item():.4f}")

# Test faiss retrieval context (TabR)
print("\n--- faiss retrieval smoke test ---")
import faiss
from track1_model import build_retrieval_context
ctx_tr, ctx_val = build_retrieval_context(tens['num_tr'], tens['num_val'], y_tr_t, k=8)
print(f"ctx_tr shape: {ctx_tr.shape}, ctx_val shape: {ctx_val.shape}")

# Test ModernNCA train_nca (few steps)
print("\n--- ModernNCA smoke test ---")
from track2_model import NCAEncoder, train_nca, infer_nca
enc = NCAEncoder(tens['num_tr'].shape[1], tens['cat_cardinalities'])
enc = train_nca(enc, tens['num_tr'], tens['cat_tr'], y_tr_t, DEVICE, epochs=1,
                 pool_size=256, anchor_batch=128, prefix="[smoke-nca] ")
p_nca = infer_nca(enc, tens['num_tr'], tens['cat_tr'], y_tr_t, tens['num_val'], tens['cat_val'], DEVICE, k=8)
print(f"ModernNCA pred shape: {p_nca.shape}, mean={p_nca.mean():.4f}")

print("\n=== ALL SMOKE TESTS PASSED ===")
