"""track2_model.py — ModernNCA encoder + train/infer functions, importable without side effects."""
import numpy as np
import torch
import torch.nn as nn

from dl_common import CatEmbedder, faiss_search

EMB_DIM = 32
POOL_SIZE = 4096
ANCHOR_BATCH = 1024
EPOCHS = 6
TEMPERATURE = 4.0
INFER_K = 64


class NCAEncoder(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, emb_dim=EMB_DIM):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x)


def train_nca(encoder, num_tr, cat_tr, y_tr, device, epochs=EPOCHS, lr=1e-3,
              pool_size=POOL_SIZE, anchor_batch=ANCHOR_BATCH, temperature=TEMPERATURE,
              prefix="", log_fn=print):
    n = len(y_tr)
    rng = np.random.RandomState(1)
    perm_dev = rng.permutation(n)
    n_dev = int(n * 0.05)
    dev_idx = perm_dev[:n_dev]
    train_pool_idx = perm_dev[n_dev:]  # anchors + candidate pool both drawn from this set only

    encoder.to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    n_train_pool = len(train_pool_idx)
    y_tr_np = y_tr.numpy()

    best_dev_loss = float('inf')
    best_state = None
    patience, bad_epochs = 2, 0

    for epoch in range(epochs):
        encoder.train()
        perm = np.random.permutation(n_train_pool)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_train_pool, anchor_batch):
            anchor_pos = train_pool_idx[perm[i:i + anchor_batch]]
            pool_pos = train_pool_idx[np.random.choice(n_train_pool, size=min(pool_size, n_train_pool), replace=False)]
            mask_keep = ~np.isin(pool_pos, anchor_pos)
            pool_pos = pool_pos[mask_keep]
            if len(pool_pos) < 8:
                continue

            xa_num = num_tr[anchor_pos].to(device)
            xa_cat = cat_tr[anchor_pos].to(device)
            xp_num = num_tr[pool_pos].to(device)
            xp_cat = cat_tr[pool_pos].to(device)
            ya = torch.tensor(y_tr_np[anchor_pos], dtype=torch.float32, device=device)
            yp = torch.tensor(y_tr_np[pool_pos], dtype=torch.float32, device=device)

            emb_a = encoder(xa_num, xa_cat)
            emb_p = encoder(xp_num, xp_cat)
            dist = torch.cdist(emb_a, emb_p, p=2)
            attn = torch.softmax(-dist / temperature, dim=1)
            pred = (attn * yp.unsqueeze(0)).sum(dim=1)
            pred = pred.clamp(1e-5, 1 - 1e-5)
            loss = -(ya * torch.log(pred) + (1 - ya) * torch.log(1 - pred)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        encoder.eval()
        with torch.no_grad():
            eval_pool_pos = train_pool_idx[np.random.RandomState(42).choice(n_train_pool, size=min(8192, n_train_pool), replace=False)]
            xp_num = num_tr[eval_pool_pos].to(device)
            xp_cat = cat_tr[eval_pool_pos].to(device)
            emb_p = encoder(xp_num, xp_cat)
            yp = torch.tensor(y_tr_np[eval_pool_pos], dtype=torch.float32, device=device)

            xa_num = num_tr[dev_idx].to(device)
            xa_cat = cat_tr[dev_idx].to(device)
            ya = torch.tensor(y_tr_np[dev_idx], dtype=torch.float32, device=device)
            emb_a = encoder(xa_num, xa_cat)
            dist = torch.cdist(emb_a, emb_p, p=2)
            attn = torch.softmax(-dist / temperature, dim=1)
            pred = (attn * yp.unsqueeze(0)).sum(dim=1).clamp(1e-5, 1 - 1e-5)
            dev_loss = -(ya * torch.log(pred) + (1 - ya) * torch.log(1 - pred)).mean().item()

        log_fn(f"{prefix}Epoch {epoch+1}/{epochs}: train_loss={train_loss:.5f} dev_loss={dev_loss:.5f}")
        if dev_loss < best_dev_loss - 1e-5:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log_fn(f"{prefix}Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder


def infer_nca(encoder, num_tr, cat_tr, y_tr, num_val, cat_val, device, k=INFER_K, batch_size=8192,
              temperature=TEMPERATURE):
    encoder.eval()
    y_tr_np = y_tr.numpy()
    with torch.no_grad():
        emb_tr_list = []
        for i in range(0, len(y_tr_np), batch_size):
            emb_tr_list.append(encoder(num_tr[i:i + batch_size].to(device), cat_tr[i:i + batch_size].to(device)).cpu().numpy())
        emb_tr = np.ascontiguousarray(np.concatenate(emb_tr_list).astype(np.float32))

        emb_v_list = []
        for i in range(0, num_val.shape[0], batch_size):
            emb_v_list.append(encoder(num_val[i:i + batch_size].to(device), cat_val[i:i + batch_size].to(device)).cpu().numpy())
        emb_v = np.ascontiguousarray(np.concatenate(emb_v_list).astype(np.float32))

    D, I = faiss_search(emb_tr, emb_v, k)
    w = np.exp(-D / temperature)
    w = w / w.sum(axis=1, keepdims=True)
    ny = y_tr_np[I]
    p = (w * ny).sum(axis=1)
    return np.clip(p, 1e-6, 1 - 1e-6)
