"""
175_irt_ability_difficulty.py
"완전히 새로운 판" 1번: 심리측정학의 문항반응이론(IRT/Rasch 모델) 구조를 차용.
지금까지 모든 모델(GBDT/TabR/TabM/PLE-MLP)은 asof_pitcher_* 같은 "집계 통계"를
투수 능력의 대리변수로 써서 다른 상황 피처들과 똑같이 트리/신경망이 자유롭게 섞도록
맡겼음. 이 실험은 정반대: 투수 능력을 명시적 잠재 스칼라(임베딩)로 분리하고,
나머지 모든 상황 피처(카운트/베이스/레버리지/이닝 등, asof_*/tkm_* 등 능력성
피처는 제외)를 "난이도" 함수로 묶어서, logit = 능력 - 난이도 + bias 로 결합.
GBDT/일반신경망과 귀납적 편향(inductive bias) 자체가 다른 구조.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor
from core.eval_utils import calc_brier_skill_score, evaluate_fold_skills

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SEEDS = [7, 123, 2025, 31415, 8675309]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


ABILITY_PREFIXES = ('asof_pitcher', 'asof_batter', 'tkm_', 'pitcher_success_trend')


class IRTModel(nn.Module):
    def __init__(self, n_pitchers, diff_cat_cardinalities, diff_num_dim, emb_dim=8):
        super().__init__()
        self.pitcher_ability = nn.Embedding(n_pitchers + 1, 1, padding_idx=0)
        nn.init.zeros_(self.pitcher_ability.weight)
        self.cat_embs = nn.ModuleList([nn.Embedding(card + 1, emb_dim) for card in diff_cat_cardinalities])
        diff_in = emb_dim * len(diff_cat_cardinalities) + diff_num_dim
        self.difficulty_net = nn.Sequential(
            nn.Linear(diff_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, pitcher_idx, diff_cat, diff_num):
        ability = self.pitcher_ability(pitcher_idx).squeeze(-1)
        cat_parts = [emb(diff_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
        diff_in = torch.cat(cat_parts + [diff_num], dim=1)
        difficulty = self.difficulty_net(diff_in).squeeze(-1)
        logit = ability - difficulty + self.bias
        return logit


def build_fold_tensors(df_tr_f, df_val_f, fold_max_season):
    prep = PitchPreprocessor()
    prep.fit(df_tr_f, as_of_season=fold_max_season, is_final=False)
    X_tr = prep.transform(df_tr_f)
    X_val = prep.transform(df_val_f)

    diff_cols = [c for c in X_tr.columns if not c.startswith(ABILITY_PREFIXES)]
    cat_cols = [c for c in diff_cols if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS]
    num_cols = [c for c in diff_cols if c not in cat_cols]

    # pitcher_id vocab (train-only, unseen -> 0/UNK)
    pitcher_vocab = {pid: i + 1 for i, pid in enumerate(df_tr_f['pitcher_id'].unique())}
    n_pitchers = len(pitcher_vocab)
    pid_tr = df_tr_f['pitcher_id'].map(pitcher_vocab).fillna(0).astype(np.int64).values
    pid_val = df_val_f['pitcher_id'].map(pitcher_vocab).fillna(0).astype(np.int64).values

    cat_vocabs, cat_cardinalities = [], []
    cat_tr_arrs, cat_val_arrs = [], []
    for c in cat_cols:
        vocab = {v: i + 1 for i, v in enumerate(X_tr[c].astype(str).unique())}
        cat_vocabs.append(vocab)
        cat_cardinalities.append(len(vocab))
        cat_tr_arrs.append(X_tr[c].astype(str).map(vocab).fillna(0).astype(np.int64).values)
        cat_val_arrs.append(X_val[c].astype(str).map(vocab).fillna(0).astype(np.int64).values)
    cat_tr = np.stack(cat_tr_arrs, axis=1) if cat_cols else np.zeros((len(X_tr), 0), dtype=np.int64)
    cat_val = np.stack(cat_val_arrs, axis=1) if cat_cols else np.zeros((len(X_val), 0), dtype=np.int64)

    num_tr_raw = X_tr[num_cols].astype(np.float32).values
    num_val_raw = X_val[num_cols].astype(np.float32).values
    num_mean, num_std = num_tr_raw.mean(0), num_tr_raw.std(0) + 1e-6
    num_tr = np.nan_to_num((num_tr_raw - num_mean) / num_std, nan=0.0)
    num_val = np.nan_to_num((num_val_raw - num_mean) / num_std, nan=0.0)

    return {
        'n_pitchers': n_pitchers, 'cat_cardinalities': cat_cardinalities, 'num_dim': len(num_cols),
        'pid_tr': pid_tr, 'pid_val': pid_val, 'cat_tr': cat_tr, 'cat_val': cat_val,
        'num_tr': num_tr, 'num_val': num_val,
    }


def run_irt_track(df_train, seeds):
    folds = get_cv_folds(df_train)
    y_full = df_train[config.TARGET_COL].values
    n = len(df_train)
    oof = np.zeros(n)
    fold_details = []
    per_seed_fold = []

    for k, fold in enumerate(folds):
        t0 = time.time()
        df_tr_f = df_train.iloc[fold.train_idx].copy()
        df_val_f = df_train.iloc[fold.val_idx].copy()
        y_tr_f = df_tr_f[config.TARGET_COL].values.astype(np.float32)
        y_val_f = df_val_f[config.TARGET_COL].values

        tens = build_fold_tensors(df_tr_f, df_val_f, fold.fold_max_season)
        pid_tr_t = torch.tensor(tens['pid_tr'], dtype=torch.int64)
        pid_val_t = torch.tensor(tens['pid_val'], dtype=torch.int64)
        cat_tr_t = torch.tensor(tens['cat_tr'], dtype=torch.int64)
        cat_val_t = torch.tensor(tens['cat_val'], dtype=torch.int64)
        num_tr_t = torch.tensor(tens['num_tr'], dtype=torch.float32)
        num_val_t = torch.tensor(tens['num_val'], dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)

        p_sum = np.zeros(len(y_val_f))
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = IRTModel(tens['n_pitchers'], tens['cat_cardinalities'], tens['num_dim']).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()

            n_tr = len(y_tr_f)
            perm = np.random.permutation(n_tr)
            n_dev = int(n_tr * 0.05)
            dev_idx, train_idx = perm[:n_dev], perm[n_dev:]

            best_dev_loss, best_state, patience, bad = 1e9, None, 3, 0
            batch_size = 4096
            for epoch in range(10):
                model.train()
                ep_perm = np.random.permutation(train_idx)
                for i in range(0, len(ep_perm), batch_size):
                    idx_b = ep_perm[i:i + batch_size]
                    opt.zero_grad()
                    logit = model(pid_tr_t[idx_b].to(DEVICE), cat_tr_t[idx_b].to(DEVICE), num_tr_t[idx_b].to(DEVICE))
                    loss = loss_fn(logit, y_tr_t[idx_b].to(DEVICE))
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    dev_logit = model(pid_tr_t[dev_idx].to(DEVICE), cat_tr_t[dev_idx].to(DEVICE), num_tr_t[dev_idx].to(DEVICE))
                    dev_loss = loss_fn(dev_logit, y_tr_t[dev_idx].to(DEVICE)).item()
                if dev_loss < best_dev_loss:
                    best_dev_loss, best_state, bad = dev_loss, {k2: v.clone() for k2, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                p_val = torch.sigmoid(model(pid_val_t.to(DEVICE), cat_val_t.to(DEVICE), num_val_t.to(DEVICE))).cpu().numpy()
            p_sum += p_val
            sk_seed, _, _, _ = calc_brier_skill_score(y_val_f, np.clip(p_val, 1e-6, 1 - 1e-6))
            per_seed_fold.append({'fold': k + 1, 'val_season': fold.val_season, 'seed': seed, 'skill_k': sk_seed})
            log(f"  fold{k+1}({fold.val_season}) seed={seed}: skill={sk_seed:.2f}")

        p_bagged = np.clip(p_sum / len(seeds), 1e-6, 1 - 1e-6)
        oof[fold.val_idx] = p_bagged
        sk, _, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
        fold_details.append({'fold': k + 1, 'val_season': fold.val_season, 'skill_k': sk})
        log(f"[IRT] === Fold {k+1} ({fold.val_season}) COMPLETE: Skill={sk:.2f} ({time.time()-t0:.1f}s) ===")

    return {'mean_skill': evaluate_fold_skills(fold_details), 'fold_details': fold_details,
            'per_seed_fold': per_seed_fold, 'oof': oof}


df_train = pd.read_csv(config.TRAIN_PATH)
log(f"=== 175: IRT 능력-난이도 잠재모델 (device={DEVICE}, 5-seed) ===")
t0 = time.time()
result = run_irt_track(df_train, SEEDS)
log(f"\nIRT 모델 최종: Skill={result['mean_skill']:.2f}점 ({(time.time()-t0)/60:.1f}min)")
log(f"참고 - GBDT SSOT: 843.69, TabM(main-env): 790.00, TabR: 784.34, MLP: 798.55")

with open('/tmp/175_result.json', 'w') as f:
    json.dump({'mean_skill': result['mean_skill'], 'fold_details': result['fold_details'],
                'per_seed_fold': result['per_seed_fold']}, f, indent=2)
np.save('/tmp/175_oof.npy', result['oof'])
log("=== 175 DONE ===")
