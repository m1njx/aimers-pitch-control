"""
update_3rd_submission_result.py — Record 3rd Submission Actual Public LB Result (796.84213)

Updates submission_history.json, submission_history.md, my_log.json, my_log.md,
and regenerates my_2week_summary_report.md.
"""

import sys, os, time, json
sys.path.insert(0, os.path.expanduser('~/LG_data'))

import config
from experiment_log import log_submission, compute_cv_reliability_stats
from my_experiment_log import log_experiment
from generate_summary_report import generate_report

print("======================================================================")
print("Updating 3rd Submission Official Public Leaderboard Result: 796.84213")
print("======================================================================")

# 1. Update submission_history.json & submission_history.md
log_submission(
    sub_id="3차 제출",
    date_str="2026-08-07",
    model_name="Candidate (c) (leaves=45, min_child=20, lr=0.05, shift=-0.007)",
    changes_summary="leaves=45 통제 변경 + Nested 검증 시계열 보정 Shift=-0.007",
    raw_brier=0.247704,
    skill_score=783.46,
    public_lb_score=796.84213,
    notes="로컬 CV 과소평가, 1차 대비 최고 실전 점수 달성. 순환검증 회피(nested-validated shift 채택)가 유효했음을 시사"
)

# 2. Update my_log.json & my_log.md
log_experiment(
    exp_id="3차 제출",
    date_str="2026-08-07",
    model_name="Candidate (c)",
    changes_summary="num_leaves=45 단일 통제 변경 + 최근성 베이스레이트 보정 Shift=-0.007",
    hypothesis="트리의 노이즈 미세 과적합만 다듬고 KBO 성공률 하락 추세에 따른 양의 편향을 Nested Shift(-0.007)로 보정",
    raw_brier=0.247704,
    skill_score=783.46,
    auc=0.549354,
    takeaway="Nested Validation으로 검증된 사후 보정이 최고 실전 점수(796.84점)를 달성함을 입증함.",
    public_lb=796.84213
)

# 3. Compute reliability stats
stats = compute_cv_reliability_stats()

# 4. Regenerate presentation report
generate_report()

print("\nAll JSON and Markdown experiment logs updated successfully!")
