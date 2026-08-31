"""hunt_slope.py — is the oracle 'slope' headroom of Idea B transferable?
The slope term is exactly the SCALE knob. If the per-fold optimal scale is stable
across folds, moving SCALE is a realizable (not oracle) gain; if it is not, the
term is unrealizable and Idea B's +25 budget shrinks further. LOFO-validated."""
import os,sys,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD,predict,skill,CACHE
SEEDS=[7,123,2025,31415,8675309]; FOLDS=[2021,2022,2023,2024]
raw={}
for y in FOLDS:
    yv=np.load(os.path.join(CACHE,f'y_{y}.npy'))
    c=dict(PROD); c['scale']=1.0; c['shift']=0.0
    p=np.mean([predict(c,dict(np.load(os.path.join(CACHE,f'pred_{y}_{s}.npz')))) for s in SEEDS],0)
    raw[y]=(p,yv)
def sk(y,scale,shift):
    p,yv=raw[y]; return skill(np.clip(0.5+scale*(p-0.5)+shift,1e-6,1-1e-6),yv)
grid_s=np.arange(0.6,2.01,0.02); grid_h=np.arange(-0.03,0.0301,0.001)
print(f'{"fold":>6} {"prod(1.10,-0.0045)":>19} {"best scale":>11} {"best shift":>11} {"best skill":>11} {"gain":>8}')
opt={}
for y in FOLDS:
    b=max(((sk(y,s,h),s,h) for s in grid_s for h in grid_h))
    opt[y]=b
    print(f'{y:>6} {sk(y,1.10,-0.0045192086):19.1f} {b[1]:11.2f} {b[2]:+11.4f} {b[0]:11.1f} '
          f'{b[0]-sk(y,1.10,-0.0045192086):+8.1f}')
print('\nLOFO transfer (pick scale/shift on 3 folds, apply to the 4th):')
for held in FOLDS:
    sel=[y for y in FOLDS if y!=held]
    b=max(((np.mean([sk(y,s,h) for y in sel]),s,h) for s in grid_s for h in grid_h))
    d=sk(held,b[1],b[2])-sk(held,1.10,-0.0045192086)
    print(f'  held {held}: chosen scale {b[1]:.2f} shift {b[2]:+.4f} -> delta on held fold {d:+8.1f}')
print('\nInner-only LOFO (2021/2022/2023, the protocol folds):')
for held in [2021,2022,2023]:
    sel=[y for y in [2021,2022,2023] if y!=held]
    b=max(((np.mean([sk(y,s,h) for y in sel]),s,h) for s in grid_s for h in grid_h))
    d=sk(held,b[1],b[2])-sk(held,1.10,-0.0045192086)
    print(f'  held {held}: chosen scale {b[1]:.2f} shift {b[2]:+.4f} -> delta {d:+8.1f}')
