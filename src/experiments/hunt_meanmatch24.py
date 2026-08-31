import os,sys,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD,predict,skill
B=os.path.join(LG,'harness/cache'); V=os.path.join(LG,'harness/cache_mtl10')
S=[7,123,2025,31415,8675309]
for y in [2024]:
    yv=np.load(os.path.join(B,f'y_{y}.npy')); A=[];Bp=[];Bm=[];d=[]
    for s in S:
        pa=predict(dict(PROD),dict(np.load(os.path.join(B,f'pred_{y}_{s}.npz'))))
        pb=predict(dict(PROD),dict(np.load(os.path.join(V,f'pred_{y}_{s}.npz'))))
        pbm=np.clip(pb+(pa.mean()-pb.mean()),1e-6,1-1e-6)
        A.append(pa);Bp.append(pb);Bm.append(pbm); d.append(skill(pbm,yv)-skill(pa,yv))
    d=np.array(d)
    print(f'fold {y}: raw bagged {skill(np.mean(Bp,0),yv)-skill(np.mean(A,0),yv):+.1f}  '
          f'mean-matched bagged {skill(np.mean(Bm,0),yv)-skill(np.mean(A,0),yv):+.1f}  '
          f'mm seed-mean {d.mean():+.1f} (pos {(d>0).sum()}/5)')
