import os, numpy as np, pandas as pd
LG=os.path.expanduser('~/LG_data')
df = pd.read_csv(os.path.join(LG,'open/data/train.csv'))
df.columns=[c.replace('﻿','') for c in df.columns]
RATES=[('asof_pitcher_success_rate','succ'),('asof_pitcher_reverse_rate','rev'),
       ('asof_pitcher_middle_rate','mid'),('asof_pitcher_ball_rate','ball'),
       ('asof_pitcher_strike_rate','strike')]
n = df['asof_pitcher_n'].astype(float)
key = df['pitcher_id'].astype(str)+'_'+df['season'].astype(str)
grp = df.groupby(['pitcher_id','season'])
out={}
for col,name in RATES:
    cnt = (n*df[col].astype(float))
    nxt = grp[col].shift(-1)
    nnext = grp['asof_pitcher_n'].shift(-1)
    cnt_next = nnext.astype(float)*nxt.astype(float)
    lab = cnt_next - cnt
    out[name]=lab
L=pd.DataFrame(out)
print('recovered label value distribution (rounded):')
for c in L.columns:
    v=L[c].round(3)
    print(f'  {c}: ', v.value_counts(dropna=False).head(5).to_dict())
# validate succ against control_success
m = L['succ'].notna()
rec = L.loc[m,'succ'].round().astype(int)
true = df.loc[m,'control_success'].astype(int)
print(f'\nsucc recon vs control_success: match {(rec==true).mean():.6f} on {m.sum():,} rows')
resid = (L.loc[m,'succ']-true).abs()
print('  max abs resid', resid.max(), ' 99.9pct', resid.quantile(0.999))
