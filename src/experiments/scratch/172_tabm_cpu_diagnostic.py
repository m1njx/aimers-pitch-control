"""
172_tabm_cpu_diagnostic.py
reverify_157이 venv311에서 TabM Step2 시작 직후 CPU 0%로 두 번 연속 멈춘 원인 진단.
GBDT는 이미 843.64로 확인됐으니 재실행 안 하고, TabM만 device를 강제로 CPU로
바꿔서 실행 - MPS(GPU) 경로가 진짜 원인인지 확인. CPU라 느리지만 완주하면
venv311(torch 2.7.1)의 MPS 백엔드 자체에 문제가 있다는 뜻.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import torch
import dl_common
dl_common.DEVICE = torch.device('cpu')  # MPS 우회

from dl_common import run_dl_track, DEBIASED_SEEDS_FULL
from track3_model import tabm_factory


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log(f"=== 172: TabM CPU-강제 진단 (device={dl_common.DEVICE}) ===")
t0 = time.time()
result = run_dl_track(tabm_factory, "T3-TabM-CPU-diag", epochs=10, lr=1e-3, batch_size=4096,
                       seeds=DEBIASED_SEEDS_FULL, log_fn=log)
log(f"TabM (CPU-강제): Skill={result['mean_skill']:.2f}점 Brier={result['overall_brier']:.6f} "
    f"({(time.time()-t0)/60:.1f}min)")
log("=== 172 DONE: CPU에서 완주 -> venv311의 MPS 백엔드가 원인일 가능성 높음 ===")
