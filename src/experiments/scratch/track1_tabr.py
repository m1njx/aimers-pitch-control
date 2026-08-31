"""
TRACK 1: TabR-lite (검색 증강 MLP)
결과: outputs/152_tabr.md, 로그: outputs/152_tabr_progress.log

전체 TabR 논문의 미분 가능한 candidate retrieval 대신, faiss 기반 k-NN 검색으로 이웃의
타겟/피처를 컨텍스트 벡터로 만들어 MLP에 추가 입력하는 실용적 간소화 버전을 구현함(명시).
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
import torch.nn as nn

import config
from cv_utils import get_cv_folds
from core.eval_utils import calc_raw_brier, calc_brier_skill_score, evaluate_fold_skills
from dl_common import (build_fold_frames, to_tensors, train_generic, predict, SimpleMLP,
                        analyze_vs_gbdt, DL_SEEDS, DEVICE, TARGET)
from track1_model import build_retrieval_context, K_NEIGHBORS

OUTPUTS_DIR = Path('~/LG_data/outputs')
LOG_PATH = OUTPUTS_DIR / '152_tabr_progress.log'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_lines = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


log(f"=== TRACK 1 START (device={DEVICE}) ===")
t_start = time.time()


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

    log(f"[Fold {k_fold+1}] Building faiss retrieval context (k={K_NEIGHBORS}, "
        f"train_n={tens['num_tr'].shape[0]:,})...")
    t_ctx0 = time.time()
    ctx_tr, ctx_val = build_retrieval_context(tens['num_tr'], tens['num_val'], y_tr_t, k=K_NEIGHBORS)
    log(f"[Fold {k_fold+1}] Retrieval context built in {time.time()-t_ctx0:.1f}s")

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
                                      verbose_prefix=f"[T1-TabR fold{k_fold+1} seed{seed}] ")
        p_val = predict(model, num_val_aug, tens['cat_val'], DEVICE, shift)
        p_sum += p_val
        log(f"[T1-TabR] Fold {k_fold+1} seed={seed} done ({time.time()-t0:.1f}s cumulative)")

    p_bagged = p_sum / len(DL_SEEDS)
    oof[fold.val_idx] = p_bagged
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
    fold_details.append({'fold': k_fold + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})
    log(f"[T1-TabR] === Fold {k_fold+1} ({fold.val_season}) COMPLETE: Skill={sk:.2f} "
        f"Brier={br:.6f} ({time.time()-t0:.1f}s) ===")

mean_skill = evaluate_fold_skills(fold_details)
overall_brier = calc_raw_brier(df_train_global.iloc[val_idx_all][TARGET].values, oof[val_idx_all])
result = dict(mean_skill=mean_skill, overall_brier=overall_brier, fold_details=fold_details,
              oof=oof, val_idx_all=val_idx_all, df_train=df_train_global)
log(f"\nTabR-lite: Skill={mean_skill:.2f}점 Brier={overall_brier:.6f}")

log("\n--- Correlation & ensemble value vs GBDT reference ---")
analysis = analyze_vs_gbdt(result)
log(f"vs GBDT: corr={analysis['corr']:.4f}, best_w={analysis['best_w']:.2f}, "
    f"best_blend_skill={analysis['best_skill']:.2f} (GBDT-only ref={analysis['gbdt_only_skill_reference']:.2f})")

t_elapsed = time.time() - t_start
log(f"\n=== TRACK 1 COMPUTATION DONE in {t_elapsed/60:.1f} min ===")

lines = [
    "# 152. TRACK 1 — TabR-lite (검색 증강 MLP) 최종 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    f"- **검증**: `strict_as_of=True`, 42-제외 2-seed 배깅(`{DL_SEEDS}`, 시간 예산상 축소, 명시)",
    f"- **구현 방식(중요 — 원 논문과의 차이 명시)**: TabR 원 논문(Gorishniy et al. 2023)의 미분 가능한 candidate retrieval 대신, "
    f"faiss `IndexFlatL2`로 z-scored 수치 피처 공간에서 k={K_NEIGHBORS} 최근접 이웃을 검색하고 "
    f"(이웃 타겟의 평균/표준편차/역거리가중평균) 3개 컨텍스트 피처를 만들어 원래 피처에 이어붙인 뒤 표준 MLP로 학습하는 "
    f"**실용적 간소화 버전(TabR-lite)**. 학습 행은 자기 자신을 이웃에서 제외(leave-one-out, 134번 self-inclusion leakage 교훈 반영), "
    f"검증 행은 학습셋에서만 검색(leakage 없음).",
    f"- **기대치**: 83번(단순 MLP) 로컬 320.50점 대비 폭발적 개선 기대하지 않음.\n",
    "---\n",
    "## 1. 단독 성능\n",
    "| 지표 | 값 |",
    "|:---|:---:|",
    f"| Raw Brier | `{overall_brier:.6f}` |",
    f"| 3→2-seed 배깅 Skill | `{mean_skill:.2f}점` |",
    f"| 83번(320.50) 대비 | `{mean_skill-320.50:+.2f}점` |",
    "\n### Fold별 상세\n",
    "| Fold | 검증시즌 | Raw Brier | Skill |",
    "|:---:|:---:|:---:|:---:|",
]
for fd in fold_details:
    lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")

lines.extend([
    "\n---\n",
    "## 2. GBDT 3종과의 상관관계 및 앙상블 가치\n",
    f"- **corr(TabR-lite, GBDT 앙상블)**: `{analysis['corr']:.4f}`",
    f"- **GBDT 단독(seed=7 참조) Skill**: `{analysis['gbdt_only_skill_reference']:.2f}점`",
    f"- **최적 블렌딩 가중치(TabR 비중)**: `{analysis['best_w']:.2f}`, 그때 Skill: **`{analysis['best_skill']:.2f}점`**",
    f"- **GBDT 단독 대비 블렌딩 개선폭**: `{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`",
    "\n---\n",
    "## 3. 최종 결론\n",
])
delta_blend = analysis['best_skill'] - analysis['gbdt_only_skill_reference']
if delta_blend > 15.10:
    lines.append(f"> ✅ **ACCEPT (앙상블 가치 있음)**: GBDT와 블렌딩 시 노이즈 바닥을 초과하는 개선(`{delta_blend:+.2f}점`)이 확인됐다.")
else:
    lines.append(f"> ❌ **REJECT**: 단독 성능(`{mean_skill:.2f}점`)이 GBDT(800점대)에 크게 못 미치고, 블렌딩해도 노이즈 바닥 이내(`{delta_blend:+.2f}점`)로 앙상블 가치가 확인되지 않았다.")

with open(OUTPUTS_DIR / '152_tabr.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

log("Report 152 written! TRACK 1 COMPLETE.")
