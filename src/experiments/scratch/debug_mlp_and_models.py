import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.expanduser('~/LG_data')
test_df = pd.read_csv(os.path.join(BASE_DIR, 'open', 'data', 'test.csv'))

# Let's inspect the models in v50 vs v51
print("Checking model probabilities on test sample:")

# v50 MLP
class CatEmbedder(nn.Module):
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

class SimpleMLP_50(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden=(128, 64), dropout=0.12):
        super().__init__()
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = num_dim + self.cat_embedder.out_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x_num, x_cat):
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)

# Check v50 MLP mean on test
# Load artifacts
art_50 = joblib.load(os.path.join(BASE_DIR, 'work', 'submit_v50', 'model', 'mlp_artifacts.pkl'))
art_51 = joblib.load(os.path.join(BASE_DIR, 'work', 'submit_v51', 'model', 'mlp_artifacts.pkl'))

print(f"v50 num_dim={art_50['num_dim']}, v51 num_dim={art_51['num_dim']}")
