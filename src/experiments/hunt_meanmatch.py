"""hunt_meanmatch.py — POST-HOC (declared as such) re-analysis of Idea U.

hunt_mtl_diag showed the multi-task MLP raises Cov(p,y) on all 3 folds (+6.3/+7.0/
+9.7 %) while its MEAN prediction sits below the baseline's by 0.00397/0.00148/
0.00046 -- a level offset, which in Brier costs (m-r)^2 and which the production
pipeline already has a dedicated global knob for (CALIBRATION_SHIFT).

Here the variant's blended prediction is shifted by the LABEL-FREE constant
(mean of baseline blend - mean of variant blend), i.e. re-centred onto the level
the production shift was calibrated for. No label of the evaluation fold is used,
so this is a legal transform -- but it was chosen AFTER seeing the result, so it is
reported as a secondary analysis and CANNOT be used to claim the pre-registered
criterion was met."""
import os,sys,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD,predict,skill
B=os.path.join(LG,'harness/cache'); V=os.path.join(LG,'harness/cache_mtl10')
S=[7,123,2025,31415,8675309]; F=[2021,2022,2023]
cells=[];print(f'{"fold":>6} {"raw bag":>9} {"mean-matched bag":>17} {"seed-mean mm":>13} {"pos":>5}')
for y in F:
    yv=np.load(os.path.join(B,f'y_{y}.npy')); A=[];Bp=[];Bm=[];d=[]
    for s in S:
        pa=predict(dict(PROD),dict(np.load(os.path.join(B,f'pred_{y}_{s}.npz'))))
        pb=predict(dict(PROD),dict(np.load(os.path.join(V,f'pred_{y}_{s}.npz'))))
        pbm=np.clip(pb+(pa.mean()-pb.mean()),1e-6,1-1e-6)   # label-free re-centring
        A.append(pa);Bp.append(pb);Bm.append(pbm)
        d.append(skill(pbm,yv)-skill(pa,yv))
    d=np.array(d); cells+=list(d)
    print(f'{y:>6} {skill(np.mean(Bp,0),yv)-skill(np.mean(A,0),yv):9.1f} '
          f'{skill(np.mean(Bm,0),yv)-skill(np.mean(A,0),yv):17.1f} {d.mean():13.1f} {(d>0).sum():>3}/5')
c=np.array(cells); se=c.std(ddof=1)/np.sqrt(len(c))
print(f'\n  15 cells mean {c.mean():+.1f}  sd {c.std(ddof=1):.1f}  SE {se:.1f}  t={c.mean()/se:.2f}  '
      f'pos {(c>0).sum()}/15')
