"""
reverify_5seed_tabr.py
TabR-lite(152번)를 150번 정식 42-제외 5-seed(7,123,2025,31415,8675309) 표준으로 재검증.
track1_model.py의 벡터화 수정(원래 65분 -> 훨씬 빠를 것으로 기대) 반영판.
OOF를 디스크에 저장.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch

import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills
from dl_common import build_fold_frames, to_tensors, train_generic, predict, SimpleMLP, DEBIASED_SEEDS_FULL, DEVICE, TARGET
from track1_model import build_retrieval_context, K_NEIGHBORS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"=== 5-seed re-verification: TabR-lite (device={DEVICE}) ===")
log(f"Seeds: {DEBIASED_SEEDS_FULL}")

t_start = time.time()
df_train_global = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train_global)
val_idx_all = np.concatenate([f.val_idx for f in folds])
oof = np.zeros(len(df_train_global))
fold_details = []
per_seed_fold_skills = []

for k_fold, fold in enumerate(folds):
    t0 = time.time()
    X_tr_f, X_val_f, y_tr_f, y_val_f = build_fold_frames(df_train_global, fold)
    tens = to_tensors(X_tr_f, X_val_f)
    y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)

    log(f"[Fold {k_fold+1}] Building retrieval context (vectorized)...")
    t_ctx0 = time.time()
    ctx_tr, ctx_val = build_retrieval_context(tens['num_tr'], tens['num_val'], y_tr_t, k=K_NEIGHBORS)
    log(f"[Fold {k_fold+1}] Retrieval context built in {time.time()-t_ctx0:.1f}s")

    num_tr_aug = torch.cat([tens['num_tr'], ctx_tr], dim=1)
    num_val_aug = torch.cat([tens['num_val'], ctx_val], dim=1)
    num_dim = num_tr_aug.shape[1]
    cat_cardinalities = tens['cat_cardinalities']

    p_sum = np.zeros(len(y_val_f))
    for seed in DEBIASED_SEEDS_FULL:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SimpleMLP(num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1)
        model, shift = train_generic(model, num_tr_aug, tens['cat_tr'], y_tr_t, epochs=10, lr=1e-3,
                                      batch_size=8192, device=DEVICE,
                                      verbose_prefix=f"[T1-5seed fold{k_fold+1} seed{seed}] ")
        p_val = predict(model, num_val_aug, tens['cat_val'], DEVICE, shift)
        p_sum += p_val
        sk_seed, br_seed, _, _ = calc_brier_skill_score(y_val_f, p_val)
        per_seed_fold_skills.append({'fold': k_fold + 1, 'val_season': fold.val_season, 'seed': seed,
                                      'skill_k': sk_seed, 'raw_brier_k': br_seed})
        collapse_flag = " ⚠️ COLLAPSE (skill<=0)" if sk_seed <= 0 else ""
        log(f"[T1-5seed] Fold {k_fold+1} seed={seed} done: skill={sk_seed:.2f}{collapse_flag} "
            f"({time.time()-t0:.1f}s cumulative)")

    p_bagged = p_sum / len(DEBIASED_SEEDS_FULL)
    oof[fold.val_idx] = p_bagged
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
    fold_details.append({'fold': k_fold + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})
    log(f"[T1-5seed] === Fold {k_fold+1} ({fold.val_season}) COMPLETE: Skill={sk:.2f} "
        f"Brier={br:.6f} ({time.time()-t0:.1f}s) ===")

mean_skill = evaluate_fold_skills(fold_details)
overall_brier = calc_raw_brier(df_train_global.iloc[val_idx_all][TARGET].values, oof[val_idx_all])
log(f"\nTabR-lite (5-seed): Skill={mean_skill:.2f}점 Brier={overall_brier:.6f}")

n_collapsed = sum(1 for r in per_seed_fold_skills if r['skill_k'] <= 0)
log(f"Per-seed-fold stability: {len(per_seed_fold_skills)} runs, {n_collapsed} collapsed (skill<=0)")
for r in per_seed_fold_skills:
    flag = " ⚠️ COLLAPSE" if r['skill_k'] <= 0 else ""
    log(f"  fold{r['fold']}({r['val_season']}) seed={r['seed']}: skill={r['skill_k']:.2f}{flag}")

np.savez('/tmp/tabr_5seed_oof.npz', oof=oof, val_idx_all=val_idx_all, mean_skill=mean_skill,
          overall_brier=overall_brier, n_collapsed=n_collapsed)

t_elapsed = time.time() - t_start
log(f"\n=== DONE in {t_elapsed/60:.1f} min ===")
