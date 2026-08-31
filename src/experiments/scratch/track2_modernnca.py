"""
TRACK 2: ModernNCA (경량 소프트 최근접 이웃, Neighborhood Component Analysis 계열)
결과: outputs/153_modernnca.md, 로그: outputs/153_modernnca_progress.log

학습: 미니배치 anchor를 무작위 후보 pool과 임베딩 공간에서 비교, softmax(-거리/T)로
후보 라벨을 가중평균한 값을 예측치로 삼아 BCE로 encoder를 end-to-end 학습(NCA 목적함수).
추론: 전체 학습셋 임베딩으로 faiss 인덱스를 구축, top-K 이웃의 라벨을 동일한 방식으로 가중평균.
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
from dl_common import build_fold_frames, to_tensors, CatEmbedder, analyze_vs_gbdt, DL_SEEDS, DEVICE, TARGET
from track2_model import NCAEncoder, train_nca, infer_nca, EMB_DIM, POOL_SIZE, ANCHOR_BATCH, EPOCHS, TEMPERATURE, INFER_K

OUTPUTS_DIR = Path('~/LG_data/outputs')
LOG_PATH = OUTPUTS_DIR / '153_modernnca_progress.log'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_lines = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


log(f"=== TRACK 2 START (device={DEVICE}) ===")
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
    num_dim = tens['num_tr'].shape[1]
    cat_cardinalities = tens['cat_cardinalities']

    p_sum = np.zeros(len(y_val_f))
    for seed in DL_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        encoder = NCAEncoder(num_dim, cat_cardinalities)
        encoder = train_nca(encoder, tens['num_tr'], tens['cat_tr'], y_tr_t, DEVICE,
                             prefix=f"[T2-NCA fold{k_fold+1} seed{seed}] ", log_fn=log)
        p_val = infer_nca(encoder, tens['num_tr'], tens['cat_tr'], y_tr_t,
                           tens['num_val'], tens['cat_val'], DEVICE)
        p_sum += p_val
        log(f"[T2-NCA] Fold {k_fold+1} seed={seed} done ({time.time()-t0:.1f}s cumulative)")

    p_bagged = p_sum / len(DL_SEEDS)
    oof[fold.val_idx] = p_bagged
    sk, br, _, _ = calc_brier_skill_score(y_val_f, p_bagged)
    fold_details.append({'fold': k_fold + 1, 'val_season': fold.val_season, 'skill_k': sk, 'raw_brier_k': br})
    log(f"[T2-NCA] === Fold {k_fold+1} ({fold.val_season}) COMPLETE: Skill={sk:.2f} "
        f"Brier={br:.6f} ({time.time()-t0:.1f}s) ===")

mean_skill = evaluate_fold_skills(fold_details)
overall_brier = calc_raw_brier(df_train_global.iloc[val_idx_all][TARGET].values, oof[val_idx_all])
result = dict(mean_skill=mean_skill, overall_brier=overall_brier, fold_details=fold_details,
              oof=oof, val_idx_all=val_idx_all, df_train=df_train_global)
log(f"\nModernNCA: Skill={mean_skill:.2f}점 Brier={overall_brier:.6f}")

log("\n--- Correlation & ensemble value vs GBDT reference ---")
analysis = analyze_vs_gbdt(result)
log(f"vs GBDT: corr={analysis['corr']:.4f}, best_w={analysis['best_w']:.2f}, "
    f"best_blend_skill={analysis['best_skill']:.2f} (GBDT-only ref={analysis['gbdt_only_skill_reference']:.2f})")

t_elapsed = time.time() - t_start
log(f"\n=== TRACK 2 COMPUTATION DONE in {t_elapsed/60:.1f} min ===")

lines = [
    "# 153. TRACK 2 — ModernNCA (경량 소프트 최근접 이웃) 최종 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    f"- **검증**: `strict_as_of=True`, 42-제외 2-seed 배깅(`{DL_SEEDS}`, 시간 예산상 축소, 명시)",
    f"- **구현**: 미니배치 anchor 대 무작위 후보 pool(size={POOL_SIZE})의 임베딩 거리에 softmax(-거리/T={TEMPERATURE})를 취해 "
    f"후보 라벨을 가중평균한 값을 예측치로 BCE 학습(NCA 목적함수) — encoder를 end-to-end 학습. "
    f"추론 시엔 전체 학습셋 임베딩으로 faiss 인덱스를 구축해 top-{INFER_K} 이웃 가중평균으로 예측.",
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
    f"- **corr(ModernNCA, GBDT 앙상블)**: `{analysis['corr']:.4f}`",
    f"- **GBDT 단독(seed=7 참조) Skill**: `{analysis['gbdt_only_skill_reference']:.2f}점`",
    f"- **최적 블렌딩 가중치(NCA 비중)**: `{analysis['best_w']:.2f}`, 그때 Skill: **`{analysis['best_skill']:.2f}점`**",
    f"- **GBDT 단독 대비 블렌딩 개선폭**: `{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`",
    "\n---\n",
    "## 3. 최종 결론\n",
])
delta_blend = analysis['best_skill'] - analysis['gbdt_only_skill_reference']
if delta_blend > 15.10:
    lines.append(f"> ✅ **ACCEPT (앙상블 가치 있음)**: GBDT와 블렌딩 시 노이즈 바닥을 초과하는 개선(`{delta_blend:+.2f}점`)이 확인됐다.")
else:
    lines.append(f"> ❌ **REJECT**: 단독 성능(`{mean_skill:.2f}점`)이 GBDT(800점대)에 크게 못 미치고, 블렌딩해도 노이즈 바닥 이내(`{delta_blend:+.2f}점`)로 앙상블 가치가 확인되지 않았다.")

with open(OUTPUTS_DIR / '153_modernnca.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

log("Report 153 written! TRACK 2 COMPLETE.")
