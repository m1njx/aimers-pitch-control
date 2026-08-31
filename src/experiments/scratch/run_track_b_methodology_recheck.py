"""
run_track_b_methodology_recheck.py
TRACK B: 방법론 엄격함 재검토
결과: outputs/148_track_b_methodology_recheck.md, 진행로그: outputs/148_track_b_progress.log
"""
import sys, os, time, re, warnings, subprocess
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

import config
from core.eval_utils import run_standard_sota_evaluation

OUTPUTS_DIR = Path('~/LG_data/outputs')
LOG_PATH = OUTPUTS_DIR / '148_track_b_progress.log'
NOW_STR = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

log_lines = []


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_lines.append(line)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


log("=== TRACK B START: 방법론 엄격함 재검토 ===")
t_start = time.time()

TARGET_CUTOFF = 1014.0
CURRENT_BEST_LOCAL = 843.69  # 145번, seed42 미포함 5-seed bagged, 가장 실전에 근접했던 추정치
CURRENT_BEST_LB = 840.76     # 5차 제출, 실전 역대 최고 (113번, 정정 반영)

df_train = pd.read_csv(config.TRAIN_PATH)
sota_mp = {
    'lgb': {'colsample_bytree': 0.7, 'subsample': 0.7},
    'cb': {'iterations': 250, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0},
    'xgb': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.8}
}
sota_weights = (0.15, 0.75, 0.10)
sota_shifts = {'lgb': -0.007, 'cb': -0.008, 'xgb': -0.006}
SEEDS = [42, 100, 2024]

report_sections = {}

# =============================================================================
# SUB-TASK 1: strict_as_of=True vs False 비교, 실제 제출 이력과 대조
# =============================================================================
log("\n--- Sub-task 1: strict_as_of=True vs False comparison (current SSOT, 3-seed bagged) ---")

log("[Run] strict_as_of=True (현재 공식 방식), 3-seed bagged...")
t0 = time.time()
r_strict_true = run_standard_sota_evaluation(df_train, strict_as_of=True, model_params=sota_mp,
                                              weights=sota_weights, shifts=sota_shifts, random_seeds=SEEDS)
log(f"  strict=True: Skill={r_strict_true['mean_fold_skill']:.2f}점 Brier={r_strict_true['overall_raw_brier']:.6f} "
    f"({(time.time()-t0)/60:.1f}min)")

log("[Run] strict_as_of=False (as_of_season=2023 고정, '덜 엄격'), 3-seed bagged...")
t1 = time.time()
r_strict_false = run_standard_sota_evaluation(df_train, strict_as_of=False, model_params=sota_mp,
                                               weights=sota_weights, shifts=sota_shifts, random_seeds=SEEDS)
log(f"  strict=False: Skill={r_strict_false['mean_fold_skill']:.2f}점 Brier={r_strict_false['overall_raw_brier']:.6f} "
    f"({(time.time()-t1)/60:.1f}min)")

gap_true = CURRENT_BEST_LB - r_strict_true['mean_fold_skill']
gap_false = CURRENT_BEST_LB - r_strict_false['mean_fold_skill']
log(f"6차 제출 실전(839.60) 대비: strict=True gap={839.6025545093 - r_strict_true['mean_fold_skill']:+.2f}, "
    f"strict=False gap={839.6025545093 - r_strict_false['mean_fold_skill']:+.2f}")
log(f"5차 제출 실전(840.76) 대비(참고, 다른 config): strict=True gap={840.76 - 850.09:+.2f} (113번), "
    f"strict=False gap={840.76 - 843.42:+.2f} (113번)")

report_sections['subtask1'] = dict(
    strict_true_skill=r_strict_true['mean_fold_skill'], strict_true_brier=r_strict_true['overall_raw_brier'],
    strict_false_skill=r_strict_false['mean_fold_skill'], strict_false_brier=r_strict_false['overall_raw_brier'],
)

# =============================================================================
# SUB-TASK 2: 과거 REJECT 후보 중 노이즈 바닥 근처였던 것들 컴파일
# =============================================================================
log("\n--- Sub-task 2: compiling past near-noise-floor REJECT candidates ---")

reject_candidates = []
report_files = sorted(OUTPUTS_DIR.glob('*.md'))
pattern = re.compile(r'([+-]\d+\.\d{1,2})\s*점')
for rf in report_files:
    try:
        text = rf.read_text(encoding='utf-8')
    except Exception:
        continue
    if 'REJECT' in text or '기각' in text or '판별불가' in text:
        deltas = [float(m) for m in pattern.findall(text)]
        near_noise = [d for d in deltas if 0 < abs(d) <= 20]
        if near_noise:
            reject_candidates.append((rf.name, near_noise[:5]))

log(f"Reports mentioning REJECT/기각/판별불가 with a delta in (0, 20]점 range: {len(reject_candidates)}개 보고서")
for name, deltas in reject_candidates[:30]:
    log(f"  {name}: {deltas}")

report_sections['subtask2'] = dict(reject_candidates=reject_candidates)

t_elapsed = time.time() - t_start
log(f"\n=== TRACK B COMPUTATION DONE in {t_elapsed/60:.1f} min. Writing report... ===")

# =============================================================================
# WRITE REPORT 148
# =============================================================================
s1 = report_sections['subtask1']
s2 = report_sections['subtask2']

lines = [
    "# 148. TRACK B — 방법론 엄격함 재검토 최종 보고서\n",
    f"- **작성 일시**: {NOW_STR}",
    f"- **소요 시간**: {t_elapsed/60:.1f}분",
    f"- **목표**: 1014점(리더보드 100등 커트라인) 도달을 위해, 지금까지 지켜온 검증 원칙 중 과도하게 보수적인 것이 있는지 재검토\n",
    "---\n",
    "## 1. strict_as_of=True가 정말 최적인가? — 실전 데이터로 재검토\n",
    "### 1.1 현재 SSOT(15/75/10) 구성에서 strict=True vs False 3-seed 배깅 비교\n",
    "| 방식 | Skill | Raw Brier |",
    "|:---|:---:|:---:|",
    f"| **strict_as_of=True (현재 공식)** | `{s1['strict_true_skill']:.2f}점` | `{s1['strict_true_brier']:.6f}` |",
    f"| strict_as_of=False (as_of=2023 고정, 덜 엄격) | `{s1['strict_false_skill']:.2f}점` | `{s1['strict_false_brier']:.6f}` |",
    f"\n- 6차 제출 실전 점수(839.60)와의 거리: strict=True `{839.6025545093 - s1['strict_true_skill']:+.2f}점` vs strict=False `{839.6025545093 - s1['strict_false_skill']:+.2f}점`",
    "\n### 1.2 113번 보고서의 과거 관측 (5차 제출, 구 20/70/10 구성)\n",
    "| 방식 | 로컬 Skill | 5차 실전(840.76)과의 거리 |",
    "|:---|:---:|:---:|",
    "| strict_as_of=True | `850.09점` | `-9.33점` |",
    "| strict_as_of=False | `843.42점` | `-2.66점` (더 근접) |",
    "\n### 1.3 종합 판단\n",
]

# Decide verdict based on actual numbers
if abs(839.6025545093 - s1['strict_false_skill']) < abs(839.6025545093 - s1['strict_true_skill']):
    verdict_1 = ("strict=False가 6차 제출 실전 점수에도 더 근접했다 — 113번(5차)에 이어 두 번째 사례에서도 "
                 "같은 패턴이 재현되었다. **strict_as_of=False가 실전 예측력 면에서는 오히려 더 나은 신호일 가능성이 있다.**")
else:
    verdict_1 = ("이번엔 strict=True가 6차 제출 실전 점수에 더 근접해, 113번(5차)의 패턴이 재현되지 않았다. "
                 "표본이 2건뿐이라 어느 쪽이 실전을 더 잘 예측하는지 아직 확정할 수 없다.")
lines.append(f"- {verdict_1}")
lines.append(
    "- **다만 중요한 구분**: `strict_as_of=False`는 **Fold 1/2에 미래(2023년) 데이터가 섞이는 진짜 leakage가 있는 방식**이다(113번이 이미 지적). "
    "이걸 새 공식 검증 기준으로 승격하는 것은 위험하다 — leakage가 있는 지표로 모델을 고르면 실전에서 예측 못한 방식으로 무너질 수 있다(2차 제출의 교훈과 유사한 유형).\n"
)
lines.append(
    "- **권고**: strict_as_of=False를 '공식 채택/기각 기준'으로 승격하지는 않되, **참고 지표로 나란히 보고**하는 관행을 유지한다(이미 137번 이후 스크립트들이 이렇게 하고 있음). "
    "즉 strict=True는 여전히 '틀리지 않은 방향'이지만, strict=False와의 괴리가 클 때는 두 지표 모두 제시해 사용자가 리스크를 판단할 수 있게 하는 것이 최선이다.\n"
)

lines.extend([
    "---\n",
    "## 2. Nested Validation이 실전 개선 기회를 놓치고 있는가?\n",
    f"- 노이즈 바닥(±15.1점 혹은 그 이전 ±1.7점) 근처에서 REJECT 판정을 받은 후보를 다룬 보고서: **{len(s2['reject_candidates'])}개**",
    "- 대표 사례 (보고서명, 근처 delta 값들):",
])
for name, deltas in s2['reject_candidates'][:20]:
    lines.append(f"  - `{name}`: {deltas}")
lines.extend([
    "\n### 2.1 핵심 사실: 이 중 실제로 제출까지 이어진 사례는 0건\n",
    "- 지금까지 6번의 실제 제출은 전부 로컬 CV에서 '명확히 채택'된 구성(공식 SSOT 갱신 시점)이었고, "
    "노이즈 바닥 근처의 애매한 REJECT 후보를 일부러 실제 제출해서 검증한 사례는 없다.",
    "- 3차 제출(로컬 783.46 → 실전 796.84, +13.38)이 유일하게 '로컬이 과소평가했던' 사례이고, "
    "2차 제출(로컬 823.95 → 실전 684.98, -138.97)은 반대로 '로컬이 크게 과대평가'했던 재앙적 사례다. "
    "즉 로컬 CV의 방향성 오차 자체가 양쪽으로 다 발생한 전례가 있어, **노이즈 근처 후보를 채택 안 하는 보수적 태도가 반드시 손해인지는 증거가 엇갈린다.**",
    "\n### 2.2 판단\n",
    "- 지금까지의 REJECT 판정 대부분(구조 전환, 매치업, 태깅 불일치, 단조제약 앙상블 등)은 노이즈 바닥을 **명확히** 벗어난 큰 폭 악화였고, "
    "진짜로 '애매한 경계선(노이즈 바닥 ±15점 이내에서 근소하게 양수)'에 있던 후보는 드물다 — 대표적으로 135번 4-feature 단조제약(LightGBM +3.69점 등) 정도.",
    "- **결론**: nested validation 원칙 자체가 실전 개선 기회를 체계적으로 놓치고 있다는 증거는 약하다. 다만 "
    "1014점이라는 명확한 목표가 생긴 지금은, '노이즈 바닥 이내지만 근소하게 양수'인 후보들을 그냥 버리기보다 "
    "**낮은 리스크로 실제 제출해 실전 데이터를 쌓는 전략**은 남은 시간이 충분하다면 시도할 가치가 있다(2.1절 데이터가 부족해서 방향 판단 자체가 어려운 상태이기 때문).\n",
    "---\n",
    "## 3. 1014점 커트라인을 감안한 전략 재검토: 보수적 개별 채택 vs 리스크 조합 일괄 시도\n",
    f"- 현재 가장 신뢰할 만한 로컬 추정치(145번, 시드42 미포함 5-seed 배깅): **{CURRENT_BEST_LOCAL}점**",
    f"- 현재 실전 최고 기록(5차, 113번, 정정 반영): **{CURRENT_BEST_LB}점**",
    f"- 목표까지 남은 거리: 로컬 기준 약 **{TARGET_CUTOFF - CURRENT_BEST_LOCAL:.2f}점**, 실전 기준 약 **{TARGET_CUTOFF - CURRENT_BEST_LB:.2f}점**",
    "",
    "### 3.1 지금까지의 탐색 밀도",
    "- 8~146번(140여 개 보고서)에 걸쳐 하이퍼파라미터, 피처 엔지니어링, calibration, 구조 전환, 단조제약, 앙상블 재탐색을 "
    "이미 광범위하게 시도했고 대부분 REJECT였다. 노이즈 바닥을 명확히 초과하는 단일 후보는 사실상 고갈된 상태로 보인다(147번 TRACK A도 동일 결론).",
    "",
    "### 3.2 전략 판단",
    f"- 목표(1014점)와 현재 실전 최고(840.76점)의 격차(**{TARGET_CUTOFF-CURRENT_BEST_LB:.2f}점**)는, "
    "지금까지 관측된 어떤 단일 실험의 개선폭(대부분 노이즈 바닥 ±15점 근처이거나 그 이하)보다도 압도적으로 크다.",
    "- 따라서 **'보수적으로 하나씩 검증해서 채택'하는 방식으로는 남은 시간 내에 이 격차를 메울 가능성이 낮다**는 것이 정직한 평가다.",
    "- 그렇다고 '검증 안 된 리스크 후보들을 한꺼번에 조합해서 제출'하는 것이 합리적인 것도 아니다 — 개별 후보가 이미 노이즈 바닥 근처이거나 REJECT였던 것들을 "
    "여러 개 조합한다고 해서 그 효과가 선형적으로 합산되지 않고, 오히려 서로 상쇄되거나(상관관계가 있는 후보들) 2차 제출처럼 예상 못 한 방식으로 무너질 위험이 있다.",
    "- **가장 현실적인 결론**: 1014점은 현재 데이터/피처셋과 지금까지 시도한 방법론의 범위 안에서는 도달하기 매우 어려운 목표로 보인다. "
    "남은 자원은 (a) 이미 검증된 유일한 실질 개선인 배깅 방법론을 다듬는 것(시드 42 편향 제거 등, 145번), "
    "(b) 정말 새로운 정보원(TRACK A에서도 찾지 못함)이 없다면 확률적으로 낮은 기대값을 인정하고 안정적인 최고 기록 유지에 집중하는 것 중 택해야 한다.",
])

with open(OUTPUTS_DIR / '148_track_b_methodology_recheck.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

log("Report 148 written! TRACK B COMPLETE.")
