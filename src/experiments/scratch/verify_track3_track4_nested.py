"""
verify_track3_track4_nested.py
154번(TabM)/155번(Baseline MLP) 블렌딩 개선폭이 outer fold(2024)까지 포함한 순환검증이었을
가능성이 있어, inner(2022,2023)만으로 블렌딩 가중치를 선택 후 outer에 최초 적용하는
정직한 재검증을 수행. 결과: outputs/156_nested_honest_dl_verification.md
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
from pathlib import Path
from datetime import datetime

from dl_common import run_dl_track, analyze_vs_gbdt_nested, SimpleMLP, DL_SEEDS, DEVICE
from track3_model import tabm_factory

OUTPUTS_DIR = Path('~/LG_data/outputs')
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

t_start = time.time()
print(f"=== NESTED-HONEST RE-VERIFICATION (device={DEVICE}) ===")

print("\n--- Re-training TabM (154번) ---")
t0 = time.time()
result_tabm = run_dl_track(tabm_factory, "T3-TabM-reverify", epochs=10, lr=1e-3, batch_size=4096, seeds=DL_SEEDS)
print(f"TabM: Skill={result_tabm['mean_skill']:.2f}점 ({(time.time()-t0)/60:.1f}min)")


def baseline_factory(num_dim, cat_cardinalities, seed):
    import torch
    torch.manual_seed(seed)
    return SimpleMLP(num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1)


print("\n--- Re-training Baseline MLP (155번) ---")
t1 = time.time()
result_baseline = run_dl_track(baseline_factory, "T4-baseline-reverify", epochs=10, lr=1e-3, batch_size=8192, seeds=DL_SEEDS)
print(f"Baseline MLP: Skill={result_baseline['mean_skill']:.2f}점 ({(time.time()-t1)/60:.1f}min)")

print("\n--- Nested-honest analysis: TabM ---")
nested_tabm = analyze_vs_gbdt_nested(result_tabm)
print(f"TabM nested: best_w_inner={nested_tabm['best_w_inner']:.2f} "
      f"honest_full_skill={nested_tabm['honest_full_skill']:.2f} "
      f"(circular={nested_tabm['best_circular_skill']:.2f}, gap={nested_tabm['circularity_gap']:.2f}) "
      f"outer_only={nested_tabm['outer_only_skill']:.2f}")

print("\n--- Nested-honest analysis: Baseline MLP ---")
nested_baseline = analyze_vs_gbdt_nested(result_baseline)
print(f"Baseline MLP nested: best_w_inner={nested_baseline['best_w_inner']:.2f} "
      f"honest_full_skill={nested_baseline['honest_full_skill']:.2f} "
      f"(circular={nested_baseline['best_circular_skill']:.2f}, gap={nested_baseline['circularity_gap']:.2f}) "
      f"outer_only={nested_baseline['outer_only_skill']:.2f}")

t_elapsed = time.time() - t_start
print(f"\n=== DONE in {t_elapsed/60:.1f} min ===")

NOISE_FLOOR = 15.10
GBDT_REF = nested_tabm['gbdt_only_skill_reference']

lines = [
    "# 156a. TabM / Baseline-MLP 블렌딩 개선폭 Nested 정직 재검증 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **목적**: 154(TabM)/155(Baseline MLP)번의 블렌딩 개선폭(+48.20 / +36.23)이 outer fold(2024) 포함 "
    f"순환검증에 의한 착시인지 확인. 144번과 동일 방법론(inner 2022/23만으로 가중치 선택 → outer 2024 최초 적용).\n",
    "---\n",
    "## 1. TabM (154번)\n",
    "| 방식 | 가중치 선택 기준 | Skill | GBDT 단독 대비 |",
    "|:---|:---|:---:|:---:|",
    f"| 순환검증 (154번 원본과 동일) | outer 포함 3-fold 전체 | `{nested_tabm['best_circular_skill']:.2f}점` | `{nested_tabm['best_circular_skill']-GBDT_REF:+.2f}점` |",
    f"| **Nested 정직 검증** | inner(2022,23)만 → outer 최초 적용 | **`{nested_tabm['honest_full_skill']:.2f}점`** | **`{nested_tabm['honest_full_skill']-GBDT_REF:+.2f}점`** |",
    f"\n- 순환검증-정직검증 격차: `{nested_tabm['circularity_gap']:.2f}점`",
    f"- inner-선택 가중치(`{nested_tabm['best_w_inner']:.2f}`)의 outer(2024) 단독 성과: `{nested_tabm['outer_only_skill']:.2f}점`",
    f"- **판정**: {'✅ Noise floor 초과, 진짜 개선 가능성 있음' if nested_tabm['honest_full_skill']-GBDT_REF > NOISE_FLOOR else '❌ Noise floor 이내 — 순환검증 착시였음'}",
    "\n---\n",
    "## 2. Baseline MLP (155번)\n",
    "| 방식 | 가중치 선택 기준 | Skill | GBDT 단독 대비 |",
    "|:---|:---|:---:|:---:|",
    f"| 순환검증 (155번 원본과 동일) | outer 포함 3-fold 전체 | `{nested_baseline['best_circular_skill']:.2f}점` | `{nested_baseline['best_circular_skill']-GBDT_REF:+.2f}점` |",
    f"| **Nested 정직 검증** | inner(2022,23)만 → outer 최초 적용 | **`{nested_baseline['honest_full_skill']:.2f}점`** | **`{nested_baseline['honest_full_skill']-GBDT_REF:+.2f}점`** |",
    f"\n- 순환검증-정직검증 격차: `{nested_baseline['circularity_gap']:.2f}점`",
    f"- inner-선택 가중치(`{nested_baseline['best_w_inner']:.2f}`)의 outer(2024) 단독 성과: `{nested_baseline['outer_only_skill']:.2f}점`",
    f"- **판정**: {'✅ Noise floor 초과, 진짜 개선 가능성 있음' if nested_baseline['honest_full_skill']-GBDT_REF > NOISE_FLOOR else '❌ Noise floor 이내 — 순환검증 착시였음'}",
]

with open(OUTPUTS_DIR / '156a_nested_honest_dl_verification.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Report 156a written!")
