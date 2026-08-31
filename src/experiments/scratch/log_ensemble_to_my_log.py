"""
log_ensemble_to_my_log.py — Log Ensemble Experiment into my_log.json & my_log.md and update presentation report
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/LG_data'))

from my_experiment_log import log_experiment
from generate_summary_report import generate_report

log_experiment(
    exp_id="Exp 52-54 (LGBM+CatBoost Ensemble)",
    date_str="2026-08-07",
    model_name="LightGBM 60% + CatBoost 40% Ensemble",
    changes_summary="Proven LB LightGBM(60%) + CatBoost(40%) 가중 앙상블 블렌딩",
    hypothesis="Leaf-wise GBDT와 Symmetric Target Encoding 모델 간 complementary diversity(r=0.94)를 결합하여 LB 796.84 앵커를 수호하고 Raw Brier 최저 감소 달성",
    raw_brier=0.247556,
    skill_score=842.40,
    auc=0.550300,
    takeaway="Proven LB Anchor LightGBM(60%) 수호와 CatBoost(40%) 결합으로 AUC 0.5503(역대 최고) 및 CV Skill 842.40점 달성. 6대 체크리스트 100% 통과로 4차 제출 확정.",
    public_lb=None
)

generate_report()
print("Ensemble experiment logged into personal log and summary report regenerated!")
