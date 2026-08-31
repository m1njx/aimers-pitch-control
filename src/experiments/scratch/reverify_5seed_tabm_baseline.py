"""
reverify_5seed_tabm_baseline.py
TabM(154번)과 Baseline MLP(155번)를 150번 정식 42-제외 5-seed(7,123,2025,31415,8675309) 표준으로
재검증. OOF를 디스크에 저장해 최종 블렌딩 선택 단계에서 재사용.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np

from dl_common import run_dl_track, SimpleMLP, DEBIASED_SEEDS_FULL, DEVICE
from track3_model import tabm_factory


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"=== 5-seed re-verification: TabM + Baseline MLP (device={DEVICE}) ===")
log(f"Seeds: {DEBIASED_SEEDS_FULL}")

def stability_report(name, result):
    psk = result['per_seed_fold_skills']
    n_collapsed = sum(1 for r in psk if r['skill_k'] <= 0)
    log(f"[{name}] Per-seed-fold stability: {len(psk)} runs, {n_collapsed} collapsed (skill<=0)")
    for r in psk:
        flag = " ⚠️ COLLAPSE" if r['skill_k'] <= 0 else ""
        log(f"  fold{r['fold']}({r['val_season']}) seed={r['seed']}: skill={r['skill_k']:.2f}{flag}")
    return n_collapsed


t0 = time.time()
result_tabm = run_dl_track(tabm_factory, "T3-TabM-5seed", epochs=10, lr=1e-3, batch_size=4096,
                            seeds=DEBIASED_SEEDS_FULL, log_fn=log)
log(f"TabM (5-seed): Skill={result_tabm['mean_skill']:.2f}점 Brier={result_tabm['overall_brier']:.6f} "
    f"({(time.time()-t0)/60:.1f}min)")
tabm_collapsed = stability_report("TabM", result_tabm)
np.savez('/tmp/tabm_5seed_oof.npz', oof=result_tabm['oof'], val_idx_all=result_tabm['val_idx_all'],
          mean_skill=result_tabm['mean_skill'], overall_brier=result_tabm['overall_brier'],
          n_collapsed=tabm_collapsed)


def baseline_factory(num_dim, cat_cardinalities, seed):
    import torch
    torch.manual_seed(seed)
    return SimpleMLP(num_dim, cat_cardinalities, hidden=(64, 32), dropout=0.1)


t1 = time.time()
result_baseline = run_dl_track(baseline_factory, "T4-baseline-5seed", epochs=10, lr=1e-3, batch_size=8192,
                                seeds=DEBIASED_SEEDS_FULL, log_fn=log)
log(f"Baseline MLP (5-seed): Skill={result_baseline['mean_skill']:.2f}점 Brier={result_baseline['overall_brier']:.6f} "
    f"({(time.time()-t1)/60:.1f}min)")
baseline_collapsed = stability_report("Baseline MLP", result_baseline)
np.savez('/tmp/baseline_mlp_5seed_oof.npz', oof=result_baseline['oof'], val_idx_all=result_baseline['val_idx_all'],
          mean_skill=result_baseline['mean_skill'], overall_brier=result_baseline['overall_brier'],
          n_collapsed=baseline_collapsed)

log(f"\n=== DONE in {(time.time()-t0)/60:.1f} min total ===")
log(f"FINAL: TabM={result_tabm['mean_skill']:.2f} (collapsed={tabm_collapsed}) "
    f"Baseline={result_baseline['mean_skill']:.2f} (collapsed={baseline_collapsed})")
