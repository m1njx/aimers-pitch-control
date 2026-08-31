"""hunt_trainshift.py — deployable version of the Idea U recentering.

The constant is fitted on season y-1 (inside fold y's training set, and out-of-sample
for the models that produced those cached predictions), then applied to fold y as a
hard-coded scalar. No fold-y quantity is used anywhere. Cache-only, no training."""
import os,sys,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD,predict,skill
B=os.path.join(LG,'harness/cache'); V=os.path.join(LG,'harness/cache_mtl10')
S=[7,123,2025,31415,8675309]
def blends(y,s):
    return (predict(dict(PROD),dict(np.load(os.path.join(B,f'pred_{y}_{s}.npz')))),
            predict(dict(PROD),dict(np.load(os.path.join(V,f'pred_{y}_{s}.npz')))))
print(f'{"fold":>6} {"fit on":>7} {"delta(train-fit)":>17} {"delta(eval-derived)":>20} '
      f'{"raw bag":>9} {"train-shift bag":>16} {"eval-match bag":>15} {"pos":>5}')
rows=[]
for y,f in [(2022,2021),(2023,2022),(2024,2023)]:
    yv=np.load(os.path.join(B,f'y_{y}.npy')); A=[];R=[];T=[];M=[];d=[];dt=[];de=[]
    for s in S:
        pa_f,pb_f=blends(f,s)
        delta=float(pa_f.mean()-pb_f.mean())          # fitted on season y-1 only
        pa,pb=blends(y,s)
        ev=float(pa.mean()-pb.mean())                 # eval-derived, for comparison
        A.append(pa); R.append(pb)
        T.append(np.clip(pb+delta,1e-6,1-1e-6)); M.append(np.clip(pb+ev,1e-6,1-1e-6))
        d.append(skill(np.clip(pb+delta,1e-6,1-1e-6),yv)-skill(pa,yv))
        dt.append(delta); de.append(ev)
    d=np.array(d)
    bag=lambda X: skill(np.mean(X,0),yv)-skill(np.mean(A,0),yv)
    print(f'{y:>6} {f:>7} {np.mean(dt):+17.6f} {np.mean(de):+20.6f} {bag(R):9.1f} '
          f'{bag(T):16.1f} {bag(M):15.1f} {(d>0).sum():>3}/5')
    rows.append((y,bag(T)))
inner=[b for y,b in rows if y in (2022,2023)]
print(f'\n  inner folds (2022,2023) train-fitted bagged: {inner[0]:+.1f}, {inner[1]:+.1f}  '
      f'mean {np.mean(inner):+.1f}')
print(f'  pre-declared pass = mean >= +12 and both positive -> '
      f'{"PASS" if np.mean(inner)>=12 and all(b>0 for b in inner) else "FAIL"}')
