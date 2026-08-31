"""
Final submission join validation: TrackmanFeatureBuilder in final mode (as_of_season=None)
transforms the 5-row test.csv sample (season=2025) and validates match rates & coverage.
Also compares 7-key combinations in 2025 sample vs historical coverage.
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/LG_data'))

import pandas as pd
import numpy as np
import config
from trackman_features import TrackmanFeatureBuilder

print("=== Final Join Validation: test.csv 5-row sample ===\n")

# 1. Build final-mode TrackmanFeatureBuilder (7-key, all 2019-2024 data)
print("Step 1: Fitting TrackmanFeatureBuilder (final mode, as_of_season=None) ...")
builder = TrackmanFeatureBuilder()
builder.fit(as_of_season=None)

agg_df = builder.artifacts["agg_df"]
join_keys = config.TRACKMAN_JOIN_KEYS
tkm_cols = builder.artifacts["tkm_feature_cols"]
global_means = builder.artifacts["global_means"]

print(f"  Aggregation table shape: {agg_df.shape}")
print(f"  Unique 7-key situations in trackman (2019-2024): {len(agg_df):,}")
print()

# 2. Load test.csv sample (5 rows)
print("Step 2: Loading test.csv sample ...")
df_test = pd.read_csv(config.TEST_PATH)
print(f"  Shape: {df_test.shape}")
print(f"  Columns: {list(df_test.columns)[:10]} ...")
print(f"\n  Season values in test sample: {df_test['season'].unique().tolist()}")
print()

# 3. Check which join_keys exist in test
missing_keys = [k for k in join_keys if k not in df_test.columns]
print(f"  Join keys in test: {join_keys}")
print(f"  Missing join keys from test: {missing_keys if missing_keys else 'None (all present)'}")
print()

# 4. Transform test.csv through final builder
print("Step 3: Transforming test.csv sample ...")
df_test_out = builder.transform(df_test)

print(f"\n  Output shape: {df_test_out.shape}")
print(f"  tkm_match values: {df_test_out['tkm_match'].tolist()}")
match_rate = df_test_out['tkm_match'].mean() * 100
print(f"  Match rate: {match_rate:.1f}%")
print()

# 5. Show tkm feature values for each test row
print("Step 4: Trackman feature values in transformed test.csv:")
show_cols = ["season", "game_month", "inning", "balls_before", "strikes_before",
             "tkm_match", "tkm_rel_speed_mean", "tkm_spin_rate_mean",
             "tkm_induced_vert_break_mean", "tkm_horz_break_mean", "tkm_n_pitches"]
available_show = [c for c in show_cols if c in df_test_out.columns]
print(df_test_out[available_show].to_string(index=True))
print()

# 6. For unmatched rows (if any), show the unmatched join key combinations
unmatched = df_test_out[df_test_out["tkm_match"] == 0]
if len(unmatched) > 0:
    print(f"Step 5: {len(unmatched)} unmatched rows — their join key values:")
    print(unmatched[join_keys].to_string())
    
    # Check if those combinations ever existed in trackman
    print("\n  Checking if these 7-key combos appear in trackman history...")
    df_track = pd.read_csv(config.TRACKMAN_PATH,
                           usecols=["season", "game_month", "game_dayofweek",
                                    "inning", "top_bottom", "balls_before",
                                    "strikes_before", "outs_before"])
    df_track["top_bottom"] = df_track["top_bottom"].map({"Top": "T", "Bottom": "B"})
    for _, row in unmatched[join_keys].iterrows():
        mask = pd.Series([True] * len(df_track))
        for k in join_keys:
            mask &= (df_track[k] == row[k])
        cnt = mask.sum()
        print(f"  {dict(row)} → found {cnt} matching rows in trackman")
else:
    print("Step 5: All 5 test rows matched! No unmatched situations.")
print()

# 7. Estimate coverage for the full 245,789 test rows
# Based on the 7-key unique situations in trackman vs possible combos
# game_month: 3-10 (8), game_dayofweek: 0-6 (7), inning: 1-15 (~12),
# top_bottom: 2, balls: 0-3 (4), strikes: 0-2 (3), outs: 0-2 (3)
# Theoretical max: 8*7*12*2*4*3*3 = 48,384
theoretical_max = 8 * 7 * 12 * 2 * 4 * 3 * 3
pct_covered = len(agg_df) / theoretical_max * 100
print(f"Step 6: Situation coverage estimate")
print(f"  Theoretical max 7-key combos: {theoretical_max:,}")
print(f"  Covered in trackman (2019-2024): {len(agg_df):,} ({pct_covered:.1f}%)")
print()

# 8. Verify test rows are NOT in the training season range in tkm
print("Step 7: Verifying 2025 rows get correct (2019-2024 historical) priors ...")
print("  test.csv seasons:", df_test['season'].unique().tolist())
print("  Trackman history seasons:", sorted(df_track['season'].unique().tolist()) 
      if 'df_track' in dir() else "see above")
print()
print("  → Since join key does NOT include 'season', 2025 val rows correctly receive")
print("    situation-level priors derived from 2019-2024 historical data.")
print("  → This is the INTENDED design (season-agnostic physical priors).")
print()

# Summary
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"  Test sample rows: {len(df_test)}")
print(f"  Matched rows: {df_test_out['tkm_match'].sum()}/{len(df_test)} ({match_rate:.0f}%)")
print(f"  Global situations covered: {len(agg_df):,}/{theoretical_max:,} ({pct_covered:.1f}%)")
print(f"  Conclusion: Trackman priors are {'VALID' if match_rate >= 80 else 'PARTIALLY VALID'} for 2025 test data")

# Save full output
df_test_out.to_csv("~/LG_data/outputs/15_test_transform_output.csv", index=False)
print(f"\nFull output saved to outputs/15_test_transform_output.csv")
print("DONE")
