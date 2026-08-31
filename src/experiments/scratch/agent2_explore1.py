import pandas as pd, numpy as np, sys
sys.path.insert(0,os.path.expanduser('~/LG_data'))
import config

df = pd.read_csv(config.TRAIN_PATH, usecols=['row_id','season','game_month','game_dayofweek','inning','top_bottom',
    'balls_before','strikes_before','outs_before','run_top_before','run_bot_before','run_total_before',
    'pitcher_id','batter_id','pitcher_team_id','batter_team_id','pitcher_hand','batter_hand'])
tm = pd.read_csv(config.TRACKMAN_PATH)

print("=== TRAIN team-season presence ===")
p = df.groupby(['season','pitcher_team_id']).size().unstack(fill_value=0)
print(p.to_string())
print("\n=== TKM pitcher_team season presence ===")
q = tm.groupby(['season','pitcher_team']).size().unstack(fill_value=0)
print(q.to_string())
