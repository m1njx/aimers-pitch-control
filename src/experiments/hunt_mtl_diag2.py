"""hunt_mtl_diag.py — is the multi-task MLP merely LOWER-VARIANCE, or does it
represent something different?  Variance reduction is capped by Idea B's ~+25
estimation budget; a representation change is not."""
import os,sys,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import skill
B=os.path.join(LG,'harness/cache'); V=os.path.join(LG,'harness/cache_mtl10')
S=[7,123,2025,31415,8675309]
print(f'{"fold":>6} {"MLP base bag":>13} {"MLP mtl bag":>12} {"delta":>8} '
      f'{"seed-sd base":>13} {"seed-sd mtl":>12} {"corr(base,mtl)":>15} {"Cov*1e3 base":>13} {"mtl":>8}')
for y in [2021,2022,2023,2024]:
    yv=np.load(os.path.join(B,f'y_{y}.npy'))
    a=np.stack([np.load(os.path.join(B,f'pred_{y}_{s}.npz'))['mlp'] for s in S])
    b=np.stack([np.load(os.path.join(V,f'pred_{y}_{s}.npz'))['mlp'] for s in S])
    ab,bb=a.mean(0),b.mean(0)
    print(f'{y:>6} {skill(ab,yv):13.1f} {skill(bb,yv):12.1f} {skill(bb,yv)-skill(ab,yv):+8.1f} '
          f'{a.std(0).mean():13.5f} {b.std(0).mean():12.5f} {np.corrcoef(ab,bb)[0,1]:15.4f} '
          f'{np.cov(ab,yv)[0,1]*1e3:13.4f} {np.cov(bb,yv)[0,1]*1e3:8.4f}')
    print(f'       pred sd base {ab.std():.5f} mtl {bb.std():.5f} | '
          f'mean base {ab.mean():.5f} mtl {bb.mean():.5f} (actual r {yv.mean():.5f})')
