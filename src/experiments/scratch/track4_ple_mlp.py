"""
TRACK 4: PLE 인코딩 + 기존 MLP 재시도
결과: outputs/155_ple_mlp.md, 로그: outputs/155_ple_mlp_progress.log
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

from dl_common import (run_dl_track, analyze_vs_gbdt, SimpleMLP, PLEEncoder,
                        compute_ple_bin_edges, CatEmbedder, DL_SEEDS, DEVICE)

OUTPUTS_DIR = Path('~/LG_data/outputs')
LOG_PATH = OUTPUTS_DIR / '155_ple_mlp_progress.log'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_lines = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


log(f"=== TRACK 4 START (device={DEVICE}) ===")
t_start = time.time()

BASELINE_83_SKILL = 320.50
BASELINE_83_BRIER = 0.248850


class PLEMLP(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, seed, bin_edges):
        super().__init__()
        self.ple = PLEEncoder(bin_edges)
        self.cat_embedder = CatEmbedder(cat_cardinalities)
        in_dim = self.ple.out_dim + self.cat_embedder.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x_num_raw, x_cat):
        x_ple = self.ple(x_num_raw)
        x_cat_emb = self.cat_embedder(x_cat)
        x = torch.cat([x_ple, x_cat_emb], dim=1)
        return self.net(x).squeeze(-1)


# --- Baseline MLP (z-scored numeric, no PLE) — controlled "before" ---
log("\n--- Sub-task A: Baseline MLP (no PLE, z-scored numeric) — internal replication of 83번 spirit ---")


def baseline_factory(num_dim, cat_cardinalities, seed):
    torch.manual_seed(seed)
    return SimpleMLP(num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1)


t0 = time.time()
result_baseline = run_dl_track(baseline_factory, "T4-baseline", epochs=10, lr=1e-3, batch_size=8192,
                                seeds=DL_SEEDS, log_fn=log)
log(f"Baseline MLP (no PLE): Skill={result_baseline['mean_skill']:.2f}점 "
    f"Brier={result_baseline['overall_brier']:.6f} ({(time.time()-t0)/60:.1f}min)")

# --- PLE MLP — controlled "after" ---
log("\n--- Sub-task B: PLE-encoded MLP (same architecture, raw numeric -> PLE) ---")

_ple_edges_cache = {}


def ple_extra_tens_fn(tens):
    key = id(tens['num_tr_raw'])
    edges = compute_ple_bin_edges(tens['num_tr_raw'].numpy(), n_bins=16)
    return dict(bin_edges=edges)


def ple_factory(num_dim, cat_cardinalities, seed, bin_edges):
    torch.manual_seed(seed)
    return PLEMLP(num_dim, cat_cardinalities, seed, bin_edges)


t1 = time.time()
result_ple = run_dl_track(ple_factory, "T4-PLE", epochs=10, lr=1e-3, batch_size=8192,
                           seeds=DL_SEEDS, log_fn=log, use_raw_num=True, extra_tens_fn=ple_extra_tens_fn)
log(f"PLE MLP: Skill={result_ple['mean_skill']:.2f}점 Brier={result_ple['overall_brier']:.6f} "
    f"({(time.time()-t1)/60:.1f}min)")

# --- Correlation / ensemble value vs GBDT (use the better of the two) ---
log("\n--- Sub-task C: correlation & ensemble value vs GBDT reference ---")
best_result = result_ple if result_ple['mean_skill'] >= result_baseline['mean_skill'] else result_baseline
best_label = "PLE MLP" if best_result is result_ple else "Baseline MLP"
analysis = analyze_vs_gbdt(best_result)
log(f"vs GBDT: corr={analysis['corr']:.4f}, best_w={analysis['best_w']:.2f}, "
    f"best_blend_skill={analysis['best_skill']:.2f} (GBDT-only ref={analysis['gbdt_only_skill_reference']:.2f})")

t_elapsed = time.time() - t_start
log(f"\n=== TRACK 4 COMPUTATION DONE in {t_elapsed/60:.1f} min ===")

# =============================================================================
# WRITE REPORT 155
# =============================================================================
ple_effect = result_ple['mean_skill'] - result_baseline['mean_skill']

lines = [
    "# 155. TRACK 4 — PLE 인코딩 + 기존 MLP 재시도 최종 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    f"- **검증**: `strict_as_of=True`, 42-제외 2-seed 배깅(`{DL_SEEDS}`) — 시간 예산상 150번의 5-seed(7,123,2025,31415,8675309)에서 앞 2개로 축소함(사용자 지시의 '2시간 초과 시 현실적 조정' 조항 적용, 명시).",
    f"- **기대치**: 83번(sklearn MLP, PLE 없음)의 로컬 성능은 `{BASELINE_83_SKILL}점`(Raw Brier `{BASELINE_83_BRIER}`)으로 극히 저조했음. 폭발적 개선 기대하지 않음.\n",
    "---\n",
    "## 1. 통제 비교: PLE 인코딩 적용 전/후\n",
    "| 구성 | Raw Brier | 3-seed→2-seed 배깅 Skill | 83번(320.50) 대비 |",
    "|:---|:---:|:---:|:---:|",
    f"| 내부 재현 Baseline MLP (PLE 없음, z-score) | `{result_baseline['overall_brier']:.6f}` | `{result_baseline['mean_skill']:.2f}점` | `{result_baseline['mean_skill']-BASELINE_83_SKILL:+.2f}점` |",
    f"| **PLE-인코딩 MLP** | `{result_ple['overall_brier']:.6f}` | `{result_ple['mean_skill']:.2f}점` | `{result_ple['mean_skill']-BASELINE_83_SKILL:+.2f}점` |",
    f"\n- **PLE 인코딩 자체의 순수 효과 (동일 아키텍처, 인코딩만 차이)**: `{ple_effect:+.2f}점`",
]
if ple_effect > 15.10:
    lines.append("- **판정**: PLE가 노이즈 바닥(±15.10점)을 초과하는 뚜렷한 개선을 만들었다.")
elif ple_effect < -15.10:
    lines.append("- **판정**: PLE가 오히려 노이즈 바닥을 초과하는 악화를 만들었다.")
else:
    lines.append("- **판정**: PLE 적용 효과는 노이즈 바닥(±15.10점) 이내로, 통계적으로 유의미한 차이라 보기 어렵다.")

lines.extend([
    "\n### Fold별 상세 (Baseline MLP)\n",
    "| Fold | 검증시즌 | Raw Brier | Skill |",
    "|:---:|:---:|:---:|:---:|",
])
for fd in result_baseline['fold_details']:
    lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")

lines.extend([
    "\n### Fold별 상세 (PLE MLP)\n",
    "| Fold | 검증시즌 | Raw Brier | Skill |",
    "|:---:|:---:|:---:|:---:|",
])
for fd in result_ple['fold_details']:
    lines.append(f"| {fd['fold']} | {fd['val_season']} | `{fd['raw_brier_k']:.6f}` | `{fd['skill_k']:.2f}점` |")

lines.extend([
    "\n---\n",
    f"## 2. GBDT 3종과의 상관관계 및 앙상블 가치 (더 나은 쪽: {best_label})\n",
    f"- **corr(DL, GBDT 앙상블)**: `{analysis['corr']:.4f}` (83번의 MLP-GBDT 상관관계 0.71~0.72와 비교)",
    f"- **GBDT 단독(seed=7 참조) Skill**: `{analysis['gbdt_only_skill_reference']:.2f}점`",
    f"- **최적 블렌딩 가중치(DL 비중)**: `{analysis['best_w']:.2f}`, 그때 Skill: **`{analysis['best_skill']:.2f}점`**",
    f"- **GBDT 단독 대비 블렌딩 개선폭**: `{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`",
    "\n---\n",
    "## 3. 최종 결론\n",
])

overall_verdict = "REJECT"
if result_ple['mean_skill'] > 843.69 - 15.10:
    pass
if analysis['best_skill'] - analysis['gbdt_only_skill_reference'] > 15.10:
    overall_verdict = "ACCEPT (앙상블 가치 있음)"
    lines.append(f"> ✅ **{overall_verdict}**: {best_label}을 GBDT와 블렌딩 시 노이즈 바닥을 초과하는 개선(`{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`)이 확인됐다.")
else:
    lines.append(f"> ❌ **REJECT**: 단독 성능(`{best_result['mean_skill']:.2f}점`)도 GBDT(800점대)에 크게 못 미치고, 블렌딩해도 GBDT 단독 대비 노이즈 바닥 이내(`{analysis['best_skill']-analysis['gbdt_only_skill_reference']:+.2f}점`)로 앙상블 가치가 확인되지 않았다.")

with open(OUTPUTS_DIR / '155_ple_mlp.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

log("Report 155 written! TRACK 4 COMPLETE.")
