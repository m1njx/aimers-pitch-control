import sys, os, joblib
import pandas as pd, numpy as np
import torch
import torch.nn as nn
print("5. Training Multi-Task MLP Models (3 Seeds)...", flush=True)
v55_dir = '~/LG_data/work/submit_v55'
X = pd.read_pickle(os.path.join(v55_dir, 'model', 'X_train.pkl'))
y_df = pd.read_pickle(os.path.join(v55_dir, 'model', 'y_train.pkl'))

X_num = X.select_dtypes(include=[np.number]).fillna(0).astype(np.float32).values
joblib.dump({'num_cols': list(X.select_dtypes(include=[np.number]).columns)}, os.path.join(v55_dir, 'model', 'mlp_cols.pkl'))

y_main_t = torch.tensor(y_df['control_success'].values, dtype=torch.float32)
aux_cols = ["lab_reverse", "lab_middle", "lab_ball", "lab_strike", "lab_fastball", "lab_breaking", "lab_offspeed"]
y_aux_t = torch.tensor(y_df[aux_cols].fillna(0).values, dtype=torch.float32)
X_t = torch.tensor(X_num, dtype=torch.float32)

class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64), nn.ReLU())
        self.head_main = nn.Linear(64, 1)
        self.head_aux = nn.Linear(64, 7)
    def forward(self, x):
        feat = self.net(x)
        return torch.sigmoid(self.head_main(feat)).squeeze(), torch.sigmoid(self.head_aux(feat))

for s in [7, 123, 2025]:
    torch.manual_seed(s)
    model = MultiTaskMLP(X_num.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    crit = nn.BCELoss()
    for epoch in range(2):
        opt.zero_grad()
        p_main, p_aux = model(X_t)
        loss = crit(p_main, y_main_t) + 0.5 * crit(p_aux, y_aux_t)
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), os.path.join(v55_dir, 'model', f'mlp_seed{s}.pt'))
    print(f"MLP Seed {s} done.", flush=True)
