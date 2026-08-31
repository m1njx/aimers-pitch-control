"""
log_catboost_to_my_log.py — Log CatBoost Experiment into my_log.json & my_log.md and update presentation report
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/LG_data'))

from my_experiment_log import log_experiment
from generate_summary_report import generate_report

log_experiment(
    exp_id="Exp 49-51 (CatBoost)",
    date_str="2026-08-07",
    model_name="CatBoost (depth=6, l2=10, shift=-0.008)",
    changes_summary="CatBoost Native Categorical 인코딩 + L2=10 정규화 + 전용 Shift(-0.008)",
    hypothesis="Native Target Encoding으로 고차원 범주 신호를 잡고 CatBoost 전용 Shift를 재탐색하여 분산 붕괴 없이 오차 최저치 달성",
    raw_brier=0.247549,
    skill_score=845.52,
    auc=0.550250,
    takeaway="CatBoost Native 인코딩과 전용 Shift(-0.008) 조합이 Raw Brier 0.247549 및 CV Skill 845.52점으로 LightGBM을 압도함.",
    public_lb=None
)

generate_report()
print("CatBoost experiment logged into personal log and summary report regenerated!")
