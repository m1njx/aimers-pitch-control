"""
compare_v8_v9_tabm_predictions.py
v8(메인환경, numpy2.x, torch2.10.0)와 v9(venv311, numpy1.26.4, torch2.7.1)의
TabM 모델이 같은 입력에 대해 얼마나 다른 예측을 내는지 직접 비교.
"""
import sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/LG_data'))
sys.path.insert(0, '~/LG_data/scratch')
import numpy as np
import pandas as pd
import torch

import config
from preprocessing import PitchPreprocessor
from tabm_inference_model import TabM

df_sample = pd.read_csv(config.TRAIN_PATH).sample(n=2000, random_state=0)
SEEDS = [7, 123, 2025, 31415, 8675309]

results = {}
for version, model_dir in [('v8', 'work/submit_v8/model'), ('v9', 'work/submit_v9/model')]:
    print(f"\n=== {version} ===")
    with open(f'{model_dir}/dl_preprocessing_artifacts.pkl', 'rb') as f:
        dl_art = pickle.load(f)

    prep = PitchPreprocessor()
    prep.fit(pd.read_csv(config.TRAIN_PATH), as_of_season=None, is_final=True)
    X = prep.transform(df_sample)

    base_str = ((df_sample['runner_on_1b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_sample['runner_on_2b'].fillna(0) > 0).astype(int).astype(str) + '_' +
                (df_sample['runner_on_3b'].fillna(0) > 0).astype(int).astype(str))
    cc_str = (df_sample['balls_before'].fillna(0).astype(int).astype(str) + '_' +
              df_sample['strikes_before'].fillna(0).astype(int).astype(str))
    count_x_base_raw = (cc_str + '_' + base_str)
    X['count_x_base'] = count_x_base_raw.map(dl_art['count_x_base_map']).fillna(-1).astype(int)

    num_cols, cat_cols = dl_art['num_cols'], dl_art['cat_cols']
    num_mean, num_std = dl_art['num_mean'], dl_art['num_std']
    cat_vocabs, cat_cardinalities = dl_art['cat_vocabs'], dl_art['cat_cardinalities']

    num_arr = X[num_cols].astype(np.float32).values
    num_z = np.nan_to_num((num_arr - num_mean) / num_std, nan=0.0)
    cat_arrs = []
    for c, vocab in zip(cat_cols, cat_vocabs):
        vals = X[c].astype(str)
        unk_idx = len(vocab)
        cat_arrs.append(vals.map(vocab).fillna(unk_idx).astype(np.int64).values)
    cat_arr = np.stack(cat_arrs, axis=1)

    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    num_t = torch.tensor(num_z, dtype=torch.float32)
    cat_t = torch.tensor(cat_arr, dtype=torch.int64)

    import json
    with open(f'{model_dir}/tabm_shifts.json') as f:
        shifts = json.load(f)

    p_sum = np.zeros(len(df_sample))
    for seed in SEEDS:
        model = TabM(num_t.shape[1], cat_cardinalities, seed=seed)
        state = torch.load(f'{model_dir}/tabm_model_seed{seed}.pt', map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(num_t.to(DEVICE), cat_t.to(DEVICE))).cpu().numpy()
        shift = shifts.get(str(seed), shifts.get(seed, 0.0))
        p_sum += np.clip(p + shift, 1e-6, 1 - 1e-6)

    p_tabm = np.clip(p_sum / len(SEEDS), 1e-6, 1 - 1e-6)
    results[version] = p_tabm
    print(f"  TabM pred: mean={p_tabm.mean():.6f} std={p_tabm.std():.6f} "
          f"min={p_tabm.min():.6f} max={p_tabm.max():.6f}")

print("\n=== DIRECT COMPARISON (same 2000-row sample) ===")
diff = np.abs(results['v8'] - results['v9'])
print(f"TabM abs diff: mean={diff.mean():.6f} max={diff.max():.6f} "
      f"corr={np.corrcoef(results['v8'], results['v9'])[0,1]:.6f}")
print(f"v8 target-ish rate: {results['v8'].mean():.4f}, v9 target-ish rate: {results['v9'].mean():.4f}")
actual_rate = df_sample[config.TARGET_COL].mean()
print(f"Actual control_success rate in this sample: {actual_rate:.4f}")
