"""hunt_score_fold.py — score a variant cache on an arbitrary fold (e.g. the outer
2024 hold-out), production-identical bagging + paired per-seed cells."""
import os,sys,argparse,numpy as np
LG=os.path.expanduser('~/LG_data'); sys.path.insert(0,os.path.join(LG,'harness'))
from evaluate import PROD,predict,skill
BASE=os.path.join(LG,'harness/cache'); SEEDS=[7,123,2025,31415,8675309]
ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True)
ap.add_argument('--years',type=int,nargs='+',required=True); a=ap.parse_args()
cd=os.path.join(LG,f'harness/cache_{a.tag}')
for y in a.years:
    yv=np.load(os.path.join(BASE,f'y_{y}.npy')); A=[];B=[];d=[];da=[]
    print(f'\n=== fold {y} (tag {a.tag}) ===')
    print(f'  {"seed":>9} {"base":>9} {"var":>9} {"delta":>9} {"MLP base":>9} {"MLP var":>9}')
    for s in SEEDS:
        fa=os.path.join(BASE,f'pred_{y}_{s}.npz'); fb=os.path.join(cd,f'pred_{y}_{s}.npz')
        if not os.path.exists(fb): continue
        pa=dict(np.load(fa)); pb=dict(np.load(fb))
        a_=predict(dict(PROD),pa); b_=predict(dict(PROD),pb)
        ka,kb=skill(a_,yv),skill(b_,yv); A.append(a_);B.append(b_);d.append(kb-ka)
        ma,mb=skill(pa['mlp'],yv),skill(pb['mlp'],yv); da.append(mb-ma)
        print(f'  {s:>9} {ka:9.1f} {kb:9.1f} {kb-ka:+9.1f} {ma:9.1f} {mb:9.1f}')
    if not d: print('  no variant preds'); continue
    d=np.array(d); da=np.array(da)
    print(f'  seed-mean blend {d.mean():+.1f} (pos {(d>0).sum()}/{len(d)})   '
          f'BAGGED {skill(np.mean(B,0),yv)-skill(np.mean(A,0),yv):+.1f}   '
          f'MLP-alone seed-mean {da.mean():+.1f} (pos {(da>0).sum()}/{len(da)})')
