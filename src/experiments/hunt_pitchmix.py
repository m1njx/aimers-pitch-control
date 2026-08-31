"""Does the same differencing trick recover the per-pitch PITCH TYPE?
Reported for completeness only -- pitch type is a Trackman-derived attribute of the
CURRENT pitch, which this project has ruled out as privileged/LUPI information.
Not used in any experiment here."""
import os, numpy as np, pandas as pd
LG=os.path.expanduser('~/LG_data')
df=pd.read_csv(os.path.join(LG,'open/data/train.csv')); df.columns=[c.replace('﻿','') for c in df.columns]
g=df.groupby(['pitcher_id','season'],sort=False)
d=g['asof_pitcher_pitchmix_n'].diff()
print('asof_pitcher_pitchmix_n diff within (pitcher,season):')
print(d.value_counts(dropna=False).head(6))
n=df['asof_pitcher_pitchmix_n'].astype(float); nn=g['asof_pitcher_pitchmix_n'].shift(-1).astype(float)
L={}
for col,nm in [('asof_pitcher_fastball_rate','fb'),('asof_pitcher_breaking_rate','br'),('asof_pitcher_offspeed_rate','os')]:
    L[nm]=nn*g[col].shift(-1).astype(float)-n*df[col].astype(float)
L=pd.DataFrame(L); m=L.notna().all(1)
R=L[m].round()
print('\nrecovered per-pitch type indicator value counts:')
for c in R.columns: print(' ',c,R[c].value_counts().head(4).to_dict())
print('\nexactly-one-type rows:',(R.sum(1)==1).mean().round(5),' of',m.sum())
print('rounding residual 99.9pct:', float((L[m]-R).abs().max(1).quantile(0.999)).__round__(5))
