import os, numpy as np, pandas as pd
LG=os.path.expanduser('~/LG_data')
df = pd.read_csv(os.path.join(LG,'open/data/train.csv'))
df.columns=[c.replace('﻿','') for c in df.columns]
RATES=[('asof_pitcher_success_rate','succ'),('asof_pitcher_reverse_rate','rev'),
       ('asof_pitcher_middle_rate','mid'),('asof_pitcher_ball_rate','ball'),
       ('asof_pitcher_strike_rate','strike')]
grp = df.groupby(['pitcher_id','season'])
n = df['asof_pitcher_n'].astype(float)
L={}
for col,name in RATES:
    cnt = n*df[col].astype(float)
    cnt_next = grp['asof_pitcher_n'].shift(-1).astype(float)*grp[col].shift(-1).astype(float)
    L[name]=(cnt_next-cnt)
L=pd.DataFrame(L)
m=L.notna().all(1)
Lr=L[m].round().astype(int); Lr['season']=df.loc[m,'season'].values
print('per-label rate:'); print(Lr[['succ','rev','mid','ball','strike']].mean())
print('\nsucc/rev/mid crosstab (sum):')
print(pd.crosstab([Lr.succ,Lr.rev],Lr['mid']))
print('\nball/strike crosstab:')
print(pd.crosstab(Lr.ball,Lr.strike))
print('\nzone-class x succ:')
zc = Lr.ball*1+Lr.strike*2
print(pd.crosstab(zc,Lr.succ,normalize='index'))
print('\ncount by zone class:'); print(zc.value_counts())
# joint fine class
fine = Lr.succ.astype(str)+Lr.rev.astype(str)+Lr.mid.astype(str)+'_'+zc.astype(str)
print('\nfine class distribution:'); print(fine.value_counts().head(24))
print('\nper-season aux rates:'); print(Lr.groupby('season')[['succ','rev','mid','ball','strike']].mean())
