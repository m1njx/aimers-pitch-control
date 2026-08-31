"""
178_entity_embedding_into_gbdt.py
"1000점대는 임베딩을 잘 만든다"는 힌트 대응. 지금까지 임베딩(TabR/TabM/175번 IRT)은
전부 그 자체 모델 안에서만 쓰이고 끝났음 - GBDT(최고 성능, 843.69)에는 한번도
주입 안 해봄. pitcher_id/batter_id를 작은 신경망(2-tower)으로 사전학습해서 얻은
저차원 dense 임베딩(각 8차원)을 GBDT의 추가 피처로 주입 -> asof_* 집계통계보다
더 풍부한 개체 표현을 GBDT가 활용할 수 있는지 확인.
매 fold마다 해당 fold의 학습분(df_tr_f)만으로 임베딩을 새로 학습(리키지 방지),
val의 미학습 개체는 평균 임베딩으로 폴백.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from core.eval_utils import run_standard_sota_evaluation

EMB_DIM = 8
DEVICE = torch.device('cpu')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class EntityEmbedNet(nn.Module):
    def __init__(self, n_pitchers, n_batters, emb_dim, n_ctx):
        super().__init__()
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, emb_dim, padding_idx=0)
        self.batter_emb = nn.Embedding(n_batters + 1, emb_dim, padding_idx=0)
        self.ctx_proj = nn.Linear(n_ctx, emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim * 3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, pid, bid, ctx):
        pe = self.pitcher_emb(pid)
        be = self.batter_emb(bid)
        ce = self.ctx_proj(ctx)
        h = torch.cat([pe, be, ce], dim=1)
        return self.head(h).squeeze(-1)


def train_embeddings(df_tr_f, emb_dim=EMB_DIM, epochs=3):
    pitcher_vocab = {pid: i + 1 for i, pid in enumerate(df_tr_f['pitcher_id'].unique())}
    batter_vocab = {bid: i + 1 for i, bid in enumerate(df_tr_f['batter_id'].unique())}
    n_pitchers, n_batters = len(pitcher_vocab), len(batter_vocab)

    ctx_cols = ['balls_before', 'strikes_before', 'outs_before', 'li', 'run_total_before', 'score_diff_home']
    ctx = df_tr_f[ctx_cols].fillna(0).astype(np.float32).values
    ctx_mean, ctx_std = ctx.mean(0), ctx.std(0) + 1e-6
    ctx = (ctx - ctx_mean) / ctx_std

    pid = df_tr_f['pitcher_id'].map(pitcher_vocab).astype(np.int64).values
    bid = df_tr_f['batter_id'].map(batter_vocab).astype(np.int64).values
    y = df_tr_f[config.TARGET_COL].values.astype(np.float32)

    pid_t = torch.tensor(pid, dtype=torch.int64)
    bid_t = torch.tensor(bid, dtype=torch.int64)
    ctx_t = torch.tensor(ctx, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = EntityEmbedNet(n_pitchers, n_batters, emb_dim, len(ctx_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(y)
    batch_size = 8192
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx_b = perm[i:i + batch_size]
            opt.zero_grad()
            logit = model(pid_t[idx_b], bid_t[idx_b], ctx_t[idx_b])
            loss = loss_fn(logit, y_t[idx_b])
            loss.backward()
            opt.step()

    pitcher_emb_w = model.pitcher_emb.weight.detach().numpy()  # (n_pitchers+1, emb_dim), row0=UNK/pad
    batter_emb_w = model.batter_emb.weight.detach().numpy()
    return pitcher_vocab, batter_vocab, pitcher_emb_w, batter_emb_w


def add_entity_embedding_features(df_tr_f, df_val_f, fold_max_season, X_tr_f, X_val_f):
    pitcher_vocab, batter_vocab, p_emb, b_emb = train_embeddings(df_tr_f)

    for df_src, X_dst in [(df_tr_f, X_tr_f), (df_val_f, X_val_f)]:
        pid_idx = df_src['pitcher_id'].map(pitcher_vocab).fillna(0).astype(int).values
        bid_idx = df_src['batter_id'].map(batter_vocab).fillna(0).astype(int).values
        p_vecs = p_emb[pid_idx]  # (n, emb_dim)
        b_vecs = b_emb[bid_idx]
        for d in range(EMB_DIM):
            X_dst[f'pitcher_emb_{d}'] = p_vecs[:, d].astype(np.float32)
            X_dst[f'batter_emb_{d}'] = b_vecs[:, d].astype(np.float32)
    return X_tr_f, X_val_f


df_train = pd.read_csv(config.TRAIN_PATH)
BASE_MP = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
WEIGHTS = (0.15, 0.75, 0.10)
SHIFTS = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SCREEN_SEEDS = [7, 123]
FULL_SEEDS = [7, 123, 2025, 31415, 8675309]
BASELINE_REF = 843.69

log("=== 178: 사전학습 pitcher/batter 임베딩(8차원) -> GBDT 피처 주입, 2-seed 스크리닝 ===")
t0 = time.time()
r = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                  weights=WEIGHTS, shifts=SHIFTS, random_seeds=SCREEN_SEEDS,
                                  extra_feature_fn=add_entity_embedding_features)
dt = (time.time() - t0) / 60
log(f"[+entity_embedding] 2-seed skill={r['mean_fold_skill']:.2f} "
    f"(delta vs {BASELINE_REF}={r['mean_fold_skill']-BASELINE_REF:+.2f}) "
    f"folds={[round(fd['skill_k'],2) for fd in r['fold_details']]} ({dt:.1f}min)")

result = {'screen_skill': r['mean_fold_skill'], 'delta': r['mean_fold_skill'] - BASELINE_REF,
          'fold_details': r['fold_details'], 'minutes': dt}

if result['delta'] > -10.0:
    log("\n노이즈 바닥 근접/양수 -> 5-seed 정식 확인")
    t0 = time.time()
    r_full = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=BASE_MP,
                                           weights=WEIGHTS, shifts=SHIFTS, random_seeds=FULL_SEEDS,
                                           extra_feature_fn=add_entity_embedding_features)
    log(f"[FULL 5-seed +entity_embedding] skill={r_full['mean_fold_skill']:.2f} "
        f"(delta vs {BASELINE_REF}={r_full['mean_fold_skill']-BASELINE_REF:+.2f}) "
        f"folds={[round(fd['skill_k'],2) for fd in r_full['fold_details']]} ({(time.time()-t0)/60:.1f}min)")
    result['full_5seed_skill'] = r_full['mean_fold_skill']
    result['full_fold_details'] = r_full['fold_details']
else:
    log(f"\nDelta ({result['delta']:+.2f}) 너무 나쁨, 5-seed 생략")

with open('/tmp/178_result.json', 'w') as f:
    json.dump(result, f, indent=2)
log("\n=== 178 DONE ===")
