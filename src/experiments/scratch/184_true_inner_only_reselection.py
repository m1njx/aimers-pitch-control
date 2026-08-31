"""
184_true_inner_only_reselection.py
사용자 지적: recency decay=0.7 선택이 outer(2024)를 포함한 3-fold 평균으로
이뤄졌을 가능성. 이미 수집된 fold_details(169/174번)로 inner(fold1+fold2)만의
평균을 재계산해서 진짜 inner-최적 decay를 재선정하고, 그 값을 5-seed로
새로 확인(outer 포함 전체 지표는 절대 선택 기준으로 안 씀).
"""
import json

# 169/174번에서 이미 수집된 2-seed 스크리닝 fold_details (원본 그대로)
data = {
    'baseline(no weight)': [1838.6377375164798, 87.09659219746646, 598.9741218064171],
    'decay=0.95': [1857.7831491255693, 97.5137151579597, 570.8519361597108],
    'decay=0.85': [1856.7494964723542, 74.94292461622808, 620.010599328591],
    'decay=0.7': [1875.5401179696873, 57.74811692019766, 654.5056208346444],
    'decay=0.6': [1863.2626831544762, 54.83148011198313, 652.6672187648796],
    'decay=0.5': [1849.859450854119, 45.209758601483244, 647.9010156443366],
}

print("=== decay 후보별 INNER(fold1+fold2)만의 평균 vs OUTER(fold3) ===")
print(f"{'설정':<25}{'inner_mean':>12}{'outer(fold3)':>15}{'3fold_mean(기존선택기준)':>25}")
results = {}
for name, (f1, f2, f3) in data.items():
    inner_mean = (f1 + f2) / 2
    full_mean = (f1 + f2 + f3) / 3
    results[name] = {'inner_mean': inner_mean, 'outer': f3, 'full_mean': full_mean}
    print(f"{name:<25}{inner_mean:>12.2f}{f3:>15.2f}{full_mean:>25.2f}")

best_by_inner = max(results.items(), key=lambda kv: kv[1]['inner_mean'])
best_by_full = max(results.items(), key=lambda kv: kv[1]['full_mean'])
print(f"\n기존 방식(3-fold 평균 기준) 선택: {best_by_full[0]} (outer 정보가 선택에 섞여 들어감 -- 순환 위험)")
print(f"올바른 nested 방식(inner만 기준) 선택: {best_by_inner[0]}")
print(f"  -> 이 값의 outer(fold3) 성능: {best_by_inner[1]['outer']:.2f}")
print(f"  -> baseline(가중치 없음)의 outer(fold3): {results['baseline(no weight)']['outer']:.2f}")
print(f"  -> delta: {best_by_inner[1]['outer'] - results['baseline(no weight)']['outer']:+.2f}")

with open('/tmp/184_reselection_result.json', 'w') as f:
    json.dump({'all_candidates': results, 'old_selection(outer_contaminated)': best_by_full[0],
                'correct_inner_only_selection': best_by_inner[0]}, f, indent=2)
