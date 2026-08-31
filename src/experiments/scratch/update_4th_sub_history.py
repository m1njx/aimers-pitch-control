"""
update_4th_sub_history.py — Record 4th Submission Official LB Score (837.20) and update all logs
"""

import sys, os, json
sys.path.insert(0, os.path.expanduser('~/LG_data'))

import numpy as np
import pandas as pd
from experiment_log import log_submission, compute_cv_reliability_stats
from my_experiment_log import log_experiment
from generate_summary_report import generate_report

# 1. Log 4th submission into submission_history.json & submission_history.md
log_submission(
    sub_id="4차 제출",
    date_str="2026-08-07",
    model_name="LGBM(60%) + CatBoost(40%) Ensemble",
    changes_summary="Proven Anchor LGBM(60%, shift=-0.007) + CatBoost(40%, shift=-0.008) 가중 앙상블",
    raw_brier=0.247556,
    skill_score=842.40,
    public_lb_score=837.1995054497,
    notes="LGBM+CatBoost 앙상블 성공! LB 837.20점 달성 (+40.36점 경신). 로컬-실전 오차 -5.20점으로 역대 최고 정합도 입증."
)

stats = compute_cv_reliability_stats()

# Update submission_history.md with full markdown table
history_json_path = "~/LG_data/outputs/submission_history.json"
history_md_path = "~/LG_data/outputs/submission_history.md"

with open(history_json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

with open(history_md_path, "w", encoding="utf-8") as f:
    f.write("# DACON Aimers 9기 제출 이력 및 로컬-실전 오차 통계 보고서\n\n")
    f.write("## 1. 제출 이력 종합 기록표\n\n")
    f.write("| 제출 횟수 | 날짜 | 제출 모델 명칭 | 주요 변경 사항 | 로컬 Raw Brier | 로컬 Skill Score | **실제 리더보드 (Public LB)** | **오차 ($\text{Public LB} - \text{Local CV}$)** | 시사점 및 메모 |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for r in records:
        lb_val = f"**{r['public_lb_score']:.2f}**" if r.get('public_lb_score') is not None else "-"
        err_val = f"**{r['error']:+.2f}**" if r.get('error') is not None else "-"
        f.write(f"| {r['sub_id']} | {r['date']} | {r['model_name']} | {r['changes_summary']} | {r['raw_brier']:.6f} | {r['skill_score']:.2f}점 | {lb_val} | {err_val} | {r['notes']} |\n")
    
    f.write("\n---\n\n")
    f.write("## 2. 로컬 CV vs Public LB 신뢰성 오차 통계\n\n")
    f.write(f"- **평가된 제출 건수 ($n$)**: {stats['count']}건\n")
    f.write(f"- **평균 오차 (Public LB - CV)**: **`{stats['mean_error']:+.2f}점`**\n")
    f.write(f"- **오차 표준편차 (std)**: **`{stats['std_error']:.2f}점`**\n")
    f.write(f"- **95% 신뢰구간 (CI)**: `[{stats['ci_95'][0]:+.2f}, {stats['ci_95'][1]:+.2f}]` 점\n")


# 2. Update my_log.json / my_log.md for Exp 52-54 with actual Public LB score
my_log_path = "~/LG_data/outputs/my_log.json"
if os.path.exists(my_log_path):
    with open(my_log_path, "r", encoding="utf-8") as f:
        my_logs = json.load(f)
    
    for entry in my_logs:
        if entry["exp_id"] == "Exp 52-54 (LGBM+CatBoost Ensemble)":
            entry["public_lb"] = 837.1995054497
            entry["error"] = round(837.1995054497 - 842.40, 2)
            entry["takeaway"] = "LGBM(60%)+CatBoost(40%) 앙상블로 Public LB 837.20점 달성 (+40.36점 경신). 로컬 CV(842.40)와 오차 -5.20점으로 역대 최고 정합도 입증."
    
    with open(my_log_path, "w", encoding="utf-8") as f:
        json.dump(my_logs, f, indent=2, ensure_ascii=False)
    
    # Re-write my_log.md
    md_path = "~/LG_data/outputs/my_log.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DACON Aimers 9기 개인 실험 및 제출 히스토리 로그 (Personal Experiment Log)\n\n")
        f.write("| 실험 ID | 날짜 | 모델 명칭 | 주요 변경 사항 | CV Raw Brier | CV Skill Score | CV AUC | **Public LB** | **오차 (LB-CV)** | 핵심 요약 및 Takeaway |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for e in my_logs:
            lb_str = f"**{e['public_lb']:.2f}**" if e.get('public_lb') is not None else "-"
            err_str = f"{e['error']:+.2f}" if e.get('error') is not None else "-"
            f.write(f"| {e['exp_id']} | {e['date']} | {e['model_name']} | {e['changes_summary']} | {e['raw_brier']:.6f} | {e['skill_score']:.2f}점 | {e['auc']:.6f} | {lb_str} | {err_str} | {e['takeaway']} |\n")


# 3. Regenerate 2-week summary presentation report
generate_report()

print("Task 1 Completed successfully!")
