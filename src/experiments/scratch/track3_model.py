"""track3_model.py — TabM (BatchEnsemble MLP) model definition, importable without side effects."""
import torch
import torch.nn as nn
from dl_common import CatEmbedder

K_MEMBERS = 8


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_dim, out_dim, k_members):
        super().__init__()
        self.k = k_members
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.r = nn.Parameter(torch.randn(k_members, in_dim) * 0.1 + 1.0)
        self.s = nn.Parameter(torch.randn(k_members, out_dim) * 0.1 + 1.0)

    def forward(self, x):
        x_scaled = x * self.r.unsqueeze(1)
        out = self.W(x_scaled)
        out = out * self.s.unsqueeze(1)
        return out


class TabM(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, seed, k_members=K_MEMBERS, hidden=(64, 32)):
        super().__init__()
        self.k = k_members
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        dims = [in_dim] + list(hidden)
        self.layers = nn.ModuleList([
            BatchEnsembleLinear(dims[i], dims[i + 1], k_members) for i in range(len(dims) - 1)
        ])
        self.head = BatchEnsembleLinear(dims[-1], 1, k_members)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        x = x.unsqueeze(0).expand(self.k, -1, -1)
        for layer in self.layers:
            x = self.dropout(self.act(layer(x)))
        out = self.head(x).squeeze(-1)
        return out.mean(dim=0)


def tabm_factory(num_dim, cat_cardinalities, seed):
    torch.manual_seed(seed)
    return TabM(num_dim, cat_cardinalities, seed)
