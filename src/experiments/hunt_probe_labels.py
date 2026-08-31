import os, sys, numpy as np, pandas as pd
LG=os.path.expanduser('~/LG_data')
df = pd.read_csv(os.path.join(LG,'open/data/train.csv'))
df.columns=[c.replace('﻿','') for c in df.columns]
print(df.shape)
print(df.groupby('season')['control_success'].agg(['mean','size']))
# is the file chronologically ordered per pitcher-season?
d = df[['pitcher_id','season','asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_middle_rate','asof_pitcher_reverse_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','control_success']].copy()
g = d.groupby(['pitcher_id','season'])['asof_pitcher_n']
inc = g.diff()
print('\nasof_pitcher_n diff within (pitcher,season): value_counts head')
print(inc.value_counts(dropna=False).head(10))
# check rates sum
print('\nsuccess+reverse+middle sample:')
s = d['asof_pitcher_success_rate']+d['asof_pitcher_reverse_rate']+d['asof_pitcher_middle_rate']
print(s.describe())
s2 = d['asof_pitcher_ball_rate']+d['asof_pitcher_strike_rate']
print('ball+strike:'); print(s2.describe())
