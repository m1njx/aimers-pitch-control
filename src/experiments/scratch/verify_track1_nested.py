"""
verify_track1_nested.py
152번(TabR-lite) 블렌딩 개선폭(+41.76)의 nested-honest 재검증. TabR은 학습이 65분으로 길어
별도 스크립트로 분리. OOF를 디스크에 저장해 향후 재검증 시 재학습이 필요 없게 함.
결과: outputs/156b_tabr_nested_verification.md
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills
from dl_common import (build_fold_frames, to_tensors, train_generic, predict, SimpleMLP,
                        analyze_vs_gbdt_nested, DL_SEEDS, DEVICE, TARGET)
from track1_model import build_retrieval_context, K_NEIGHBORS

OUTPUTS_DIR = Path('~/LG_data/outputs')
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

t_start = time.time()
print(f"=== TRACK1 NESTED RE-VERIFICATION (device={DEVICE}) ===")

df_train_global = pd.read_csv(config.TRAIN_PATH)
folds = get_cv_folds(df_train_global)
val_idx_all = np.concatenate([f.val_idx for f in folds])
oof = np.zeros(len(df_train_global))
fold_details = []

for k_fold, fold in enumerate(folds):
    t0 = time.time()
    X_tr_f, X_val_f, y_tr_f, y_val_f = build_fold_frames(df_train_global, fold)
    tens = to_tensors(X_tr_f, X_val_f)
    y_tr_t = torch.tensor(y_tr_f, dtype=torch.float32)

    print(f"[Fold {k_fold+1}] Building retrieval context...")
    ctx_tr, ctx_val = build_retrieval_context(tens['num_tr'], tens['num_val'], y_tr_t, k=K_NEIGHBORS)
    num_tr_aug = torch.cat([tens['num_tr'], ctx_tr], dim=1)
    num_val_aug = torch.cat([tens['num_val'], ctx_val], dim=1)
    num_dim = num_tr_aug.shape[1]
    cat_cardinalities = tens['cat_cardinalities']

    p_sum = np.zeros(len(y_val_f))
    for seed in DL_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SimpleMLP(num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1)
        model, shift = train_generic(model, num_tr_aug, tens['cat_tr'], y_tr_t, epochs=10, lr=1e-3,
                                      batch_size=8192, device=DEVICE,
                                      verbose_prefix=f"[T1-reverify fold{k_fold+1} seed{seed}] ")
        p_val = predict(model, num_val_aug, tens['cat_val'], DEVICE, shift)
        p_sum += p_val
        print(f"[T1-reverify] Fold {k_fold+1} seed={seed} done ({time.time()-t0:.1f}s cumulative)")

    p_bagged = p_sum / len(DL_SEEDS)
    oof[fold.val_idx] = p_bagged
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
    fold_details.append({'fold': k_fold + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})
    print(f"[T1-reverify] === Fold {k_fold+1} ({fold.val_season}) COMPLETE: Skill={sk:.2f} "
          f"Brier={br:.6f} ({time.time()-t0:.1f}s) ===")

mean_skill = evaluate_fold_skills(fold_details)
overall_brier = calc_raw_brier(df_train_global.iloc[val_idx_all][TARGET].values, oof[val_idx_all])
result = dict(mean_skill=mean_skill, overall_brier=overall_brier, fold_details=fold_details,
              oof=oof, val_idx_all=val_idx_all, df_train=df_train_global)
print(f"\nTabR-lite (reverify): Skill={mean_skill:.2f}점 Brier={overall_brier:.6f}")

# Save OOF for future reuse (avoid needing to retrain again)
np.savez('/tmp/tabr_oof_cache.npz', oof=oof, val_idx_all=val_idx_all, mean_skill=mean_skill,
          overall_brier=overall_brier)

print("\n--- Nested-honest analysis ---")
nested = analyze_vs_gbdt_nested(result)
GBDT_REF = nested['gbdt_only_skill_reference']
print(f"TabR nested: best_w_inner={nested['best_w_inner']:.2f} "
      f"honest_full_skill={nested['honest_full_skill']:.2f} "
      f"(circular={nested['best_circular_skill']:.2f}, gap={nested['circularity_gap']:.2f}) "
      f"outer_only={nested['outer_only_skill']:.2f}")

t_elapsed = time.time() - t_start
print(f"\n=== DONE in {t_elapsed/60:.1f} min ===")

NOISE_FLOOR = 15.10
lines = [
    "# 156b. TabR-lite 블렌딩 개선폭 Nested 정직 재검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    "- **목적**: 152번의 블렌딩 개선폭(+41.76)이 outer fold(2024) 포함 순환검증에 의한 착시인지 확인.\n",
    "---\n",
    "| 방식 | 가중치 선택 기준 | Skill | GBDT 단독 대비 |",
    "|:---|:---|:---:|:---:|",
    f"| 순환검증 (152번 원본과 동일) | outer 포함 3-fold 전체 | `{nested['best_circular_skill']:.2f}점` | `{nested['best_circular_skill']-GBDT_REF:+.2f}점` |",
    f"| **Nested 정직 검증** | inner(2022,23)만 → outer 최초 적용 | **`{nested['honest_full_skill']:.2f}점`** | **`{nested['honest_full_skill']-GBDT_REF:+.2f}점`** |",
    f"\n- 순환검증-정직검증 격차: `{nested['circularity_gap']:.2f}점`",
    f"- inner-선택 가중치(`{nested['best_w_inner']:.2f}`)의 outer(2024) 단독 성과: `{nested['outer_only_skill']:.2f}점`",
    f"- **판정**: {'✅ Noise floor 초과, 진짜 개선' if nested['honest_full_skill']-GBDT_REF > NOISE_FLOOR else '❌ Noise floor 이내 — 순환검증 착시였음'}",
]

with open(OUTPUTS_DIR / '156b_tabr_nested_verification.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Report 156b written!")
