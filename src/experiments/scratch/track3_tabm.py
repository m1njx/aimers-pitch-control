"""
TRACK 3: TabM (배치 앙상블 MLP, BatchEnsemble style)
결과: outputs/154_tabm.md, 로그: outputs/154_tabm_progress.log
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

from dl_common import run_dl_track, analyze_vs_gbdt, CatEmbedder, DL_SEEDS, DEVICE
from track3_model import BatchEnsembleLinear, TabM, tabm_factory, K_MEMBERS

OUTPUTS_DIR = Path('~/LG_data/outputs')
LOG_PATH = OUTPUTS_DIR / '154_tabm_progress.log'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_lines = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


log(f"=== TRACK 3 START (device={DEVICE}) ===")
t_start = time.time()

log(f"\n--- Training TabM (K={K_MEMBERS} virtual batch-ensemble members) ---")
t0 = time.time()
result = run_dl_track(tabm_factory, "T3-TabM", epochs=10, lr=1e-3, batch_size=4096,
                       seeds=DL_SEEDS, log_fn=log)
log(f"TabM: Skill={result['mean_skill']:.2f}점 Brier={result['overall_brier']:.6f} "
    f"({(time.time()-t0)/60:.1f}min)")

log("\n--- Correlation & ensemble value vs GBDT reference ---")
analysis = analyze_vs_gbdt(result)
log(f"vs GBDT: corr={analysis['corr']:.4f}, best_w={analysis['best_w']:.2f}, "
    f"best_blend_skill={analysis['best_skill']:.2f} (GBDT-only ref={analysis['gbdt_only_skill_reference']:.2f})")

t_elapsed = time.time() - t_start
log(f"\n=== TRACK 3 COMPUTATION DONE in {t_elapsed/60:.1f} min ===")

lines = [
    "# 154. TRACK 3 — TabM (배치 앙상블 MLP) 최종 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    f"- **검증**: `strict_as_of=True`, 42-제외 2-seed 배깅(`{DL_SEEDS}`, 시간 예산상 축소, 명시)",
    f"- **구현**: BatchEnsemble 스타일 — 공유 가중치 W에 시드-멤버별 rank-1 스케일 벡터(r_k 입력, s_k 출력)를 곱해 {K_MEMBERS}개 가상 앙상블 멤버를 동시에 학습, 최종 예측은 K개 멤버 로짓 평균. 파라미터 공유로 K개 독립 MLP보다 훨씬 적은 비용으로 앙상블 다양성 확보.",
    f"- **기대치**: 83번(단순 MLP) 로컬 320.50점 대비 폭발적 개선 기대하지 않음.\n",
    "---\n",
    "## 1. 단독 성능\n",
    "| 지표 | 값 |",
    "|:---|:---:|",
    f"| Raw Brier | `{result['overall_brier']:.6f}` |",
    f"| 3→2-seed 배깅 Skill | `{result['mean_skill']:.2f}점` |",
    f"| 83번(320.50) 대비 | `{result['mean_skill']-320.50:+.2f}점` |",
    "\n### Fold별 상세\n",
    "| Fold | 검증시즌 | Raw Brier | Skill |",
    "|:---:|:---:|:---:|:---:|",
]
for fd in result['fold_details']:
    lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")

lines.extend([
    "\n---\n",
    "## 2. GBDT 3종과의 상관관계 및 앙상블 가치\n",
    f"- **corr(TabM, GBDT 앙상블)**: `{analysis['corr']:.4f}`",
    f"- **GBDT 단독(seed=7 참조) Skill**: `{analysis['gbdt_only_skill_reference']:.2f}점`",
    f"- **최적 블렌딩 가중치(TabM 비중)**: `{analysis['best_w']:.2f}`, 그때 Skill: **`{analysis['best_skill']:.2f}점`**",
    f"- **GBDT 단독 대비 블렌딩 개선폭**: `{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`",
    "\n---\n",
    "## 3. 최종 결론\n",
])
delta_blend = analysis['best_skill'] - analysis['gbdt_only_skill_reference']
if delta_blend > 15.10:
    lines.append(f"> ✅ **ACCEPT (앙상블 가치 있음)**: GBDT와 블렌딩 시 노이즈 바닥을 초과하는 개선(`{delta_blend:+.2f}점`)이 확인됐다.")
else:
    lines.append(f"> ❌ **REJECT**: 단독 성능(`{result['mean_skill']:.2f}점`)이 GBDT(800점대)에 크게 못 미치고, 블렌딩해도 노이즈 바닥 이내(`{delta_blend:+.2f}점`)로 앙상블 가치가 확인되지 않았다.")

with open(OUTPUTS_DIR / '154_tabm.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

log("Report 154 written! TRACK 3 COMPLETE.")
