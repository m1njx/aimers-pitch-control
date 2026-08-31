import sys, os, time
import pandas as pd
print("1. Loading Data...")
sys.stdout.flush()
sys.path[:0] = ["~/LG_data/scratch", os.path.expanduser("~/LG_data")]
import config
from agent2_recover_labels import recover

df = pd.read_csv(config.TRAIN_PATH)
print(f"Data loaded: {df.shape}. Recovering labels...")
sys.stdout.flush()
L = recover(df)
print(f"Labels recovered: {L.shape}. Done.")
sys.stdout.flush()
