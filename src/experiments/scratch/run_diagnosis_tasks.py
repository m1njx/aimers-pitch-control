"""
run_diagnosis_tasks.py — Comprehensive script to diagnose model reliability issues:
  Task 1: season feature drift & 2025 generalization risk
  Task 2: Fold 1 (2023) low AUC anomaly investigation
  Task 3: game_type feature detailed analysis & proxy check
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.expanduser('~/LG_data'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import mutual_info_score
import lightgbm as lgb

import config
from cv_utils import get_cv_folds
from trackman_features import TrackmanFeatureBuilder
from preprocessing import PitchPreprocessor

# ── TASK 1: Season Drift & Generalization Analysis ─────────────────────────────
def run_task1(df):
    print("\n=======================================================")
    print("TASK 1: SEASON DRIFT & 2025 GENERALIZATION RISK")
    print("=======================================================")

    # 1.1 Season-level target distribution & chi-square test
    season_target = df.groupby('season')['control_success'].agg(
        total='count',
        success='sum',
        rate='mean'
    ).reset_index()
    season_target['failure'] = season_target['total'] - season_target['success']

    contingency_matrix = season_target[['success', 'failure']].values
    chi2, p_val, dof, _ = chi2_contingency(contingency_matrix)

    print("\n1.1 Target Distribution per Season:")
    print(season_target[['season', 'total', 'success', 'rate']].to_string(index=False))
    print(f"\nChi-Square Test: chi2={chi2:.2f}, p-value={p_val:.4e}, dof={dof}")
    print(f"P-value < 0.05: {p_val < 0.05} (Target baseline success rate varies significantly across seasons)")

    # 1.2 CV Experiment: 3-fold CV WITHOUT 'season' feature vs WITH 'season'
    folds = get_cv_folds(df, strategy="time")

    def run_cv_without_season(df_in):
        results = []
        for fi, fold in enumerate(folds):
            df_tr = df_in.iloc[fold.train_idx].reset_index(drop=True)
            df_va = df_in.iloc[fold.val_idx].reset_index(drop=True)

            prep = PitchPreprocessor()
            prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

            X_tr = prep.transform(df_tr)
            X_va = prep.transform(df_va)

            # Drop 'season' column from X_tr and X_va
            if 'season' in X_tr.columns:
                X_tr = X_tr.drop(columns=['season'])
            if 'season' in X_va.columns:
                X_va = X_va.drop(columns=['season'])

            y_tr = df_tr[config.TARGET_COL].values
            y_va = df_va[config.TARGET_COL].values

            # Cat features after drop
            cat_cols = [c for c in X_tr.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]

            model = lgb.LGBMClassifier(
                n_estimators=300, num_leaves=63, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                random_state=42, verbosity=-1, n_jobs=-1
            )
            model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
            preds = model.predict_proba(X_va)[:, 1]

            auc = roc_auc_score(y_va, preds)
            ll = log_loss(y_va, preds)
            results.append({'fold': fi, 'val_season': fold.val_season, 'auc_no_season': auc, 'll_no_season': ll})
            print(f"  Fold {fi} (val={fold.val_season}) WITHOUT season: AUC={auc:.6f}, LogLoss={ll:.6f}")

        return pd.DataFrame(results)

    print("\n1.2 Running 3-Fold CV WITHOUT 'season' feature...")
    res_no_season = run_cv_without_season(df)

    # Load previous baseline results with season (from ablation study: c_with_tkm_aware)
    baseline_aucs = [0.57526, 0.52468, 0.54522]
    res_no_season['auc_with_season'] = baseline_aucs
    res_no_season['auc_diff'] = res_no_season['auc_no_season'] - res_no_season['auc_with_season']

    print("\nComparison: WITH season vs WITHOUT season:")
    print(res_no_season[['fold', 'val_season', 'auc_with_season', 'auc_no_season', 'auc_diff']].to_string(index=False))
    print(f"Mean AUC with season:    {np.mean(baseline_aucs):.6f}")
    print(f"Mean AUC without season: {res_no_season['auc_no_season'].mean():.6f}")
    print(f"Mean ΔAUC (no_season - with_season): {res_no_season['auc_diff'].mean():+.6f}")

    # 1.3 Inspect LightGBM Tree splits on 'season'
    print("\n1.3 Inspecting LightGBM tree splits on 'season'...")
    df_tr0 = df.iloc[folds[0].train_idx].reset_index(drop=True)
    prep0 = PitchPreprocessor()
    prep0.fit(df_tr0, as_of_season=folds[0].fold_max_season, is_final=False)
    X_tr0 = prep0.transform(df_tr0)
    y_tr0 = df_tr0[config.TARGET_COL].values

    cat_cols0 = [c for c in X_tr0.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
    m0 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
    m0.fit(X_tr0, y_tr0, categorical_feature=[X_tr0.columns.get_loc(c) for c in cat_cols0 if c in X_tr0.columns])

    tree_df = m0.booster_.dump_model()['tree_info']
    season_splits = []
    for tree in tree_df:
        def find_splits(node):
            if 'split_feature' in node:
                feat_name = m0.booster_.feature_name()[node['split_feature']]
                if feat_name == 'season':
                    season_splits.append({'threshold': node.get('threshold'), 'decision_type': node.get('decision_type')})
                if 'left_child' in node:
                    find_splits(node['left_child'])
                if 'right_child' in node:
                    find_splits(node['right_child'])
        find_splits(tree['tree_structure'])

    splits_df = pd.DataFrame(season_splits)
    print(f"Found {len(splits_df)} splits on 'season' across trees.")
    if not splits_df.empty:
        print("Sample season split thresholds:")
        print(splits_df.head(10).to_string(index=False))

    return season_target, res_no_season, splits_df


# ── TASK 2: Fold 1 (2023) Anomaly Investigation ────────────────────────────────
def run_task2(df):
    print("\n=======================================================")
    print("TASK 2: FOLD 1 (2023) LOW AUC ANOMALY INVESTIGATION")
    print("=======================================================")

    seasons = sorted(df['season'].unique())

    # 2.1 Basic distribution per season
    season_stats = []
    asof_cols = [c for c in df.columns if c.startswith('asof_')]

    for yr in seasons:
        sub = df[df['season'] == yr]
        row = {
            'season': yr,
            'count': len(sub),
            'control_success_rate': round(sub['control_success'].mean(), 4),
            'balls_mean': round(sub['balls_before'].mean(), 4),
            'strikes_mean': round(sub['strikes_before'].mean(), 4),
            'outs_mean': round(sub['outs_before'].mean(), 4),
            'game_type_unique': sub['game_type'].nunique(),
            'game_type_top_val': sub['game_type'].value_counts().index[0],
            'game_type_top_pct': round(sub['game_type'].value_counts(normalize=True).iloc[0] * 100, 2),
            'null_count': sub.isnull().sum().sum(),
            'asof_pitcher_success_mean': round(sub['asof_pitcher_success_rate'].mean(), 4),
            'asof_pitcher_n_mean': round(sub['asof_pitcher_n'].mean(), 1),
            'asof_batter_success_mean': round(sub['asof_batter_success_rate'].mean(), 4),
            'asof_batter_n_mean': round(sub['asof_batter_n'].mean(), 1),
        }
        season_stats.append(row)

    stats_df = pd.DataFrame(season_stats)
    print("\n2.1 Per-Season Distribution Comparison Table:")
    print(stats_df[['season', 'count', 'control_success_rate', 'balls_mean', 'strikes_mean',
                    'game_type_top_val', 'game_type_top_pct', 'asof_pitcher_success_mean',
                    'asof_batter_success_mean']].to_string(index=False))

    # 2.2 Deep dive into 2023 game_type breakdown vs other seasons
    print("\n2.2 game_type distribution per season (% of rows):")
    ct_gt = pd.crosstab(df['season'], df['game_type'], normalize='index') * 100
    print(ct_gt.round(2))

    print("\n2.3 Target (control_success) rate by game_type per season:")
    ct_target = df.groupby(['season', 'game_type'])['control_success'].mean().unstack()
    print(ct_target.round(4))

    # 2.4 Test 2023 model trained on ONLY 2022 vs trained on 2019-2022
    print("\n2.4 Train data window effect on 2023 Validation AUC:")
    df_2023_val = df[df['season'] == 2023].reset_index(drop=True)

    # Train A: 2019-2022
    df_tr_a = df[df['season'].isin([2019, 2020, 2021, 2022])].reset_index(drop=True)
    # Train B: 2022 only
    df_tr_b = df[df['season'] == 2022].reset_index(drop=True)
    # Train C: 2021-2022
    df_tr_c = df[df['season'].isin([2021, 2022])].reset_index(drop=True)

    for name, df_tr in [("2019-2022 (4 yrs)", df_tr_a), ("2021-2022 (2 yrs)", df_tr_c), ("2022 (1 yr)", df_tr_b)]:
        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=df_tr['season'].max(), is_final=False)
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_2023_val)
        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_2023_val[config.TARGET_COL].values

        cat_cols = [c for c in X_tr.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        model = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
        model.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
        auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        print(f"  Train: {name:<20s} → 2023 Val AUC: {auc:.6f}")

    return stats_df, ct_gt, ct_target


# ── TASK 3: Game Type Analysis & Proxy Check ────────────────────────────────────
def run_task3(df):
    print("\n=======================================================")
    print("TASK 3: GAME_TYPE FEATURE DETAILED ANALYSIS & PROXY CHECK")
    print("=======================================================")

    # 3.1 Frequency, target mean, season distribution for game_type
    gt_stats = df.groupby('game_type').agg(
        count=('control_success', 'count'),
        pct=('control_success', lambda x: len(x) / len(df) * 100),
        target_mean=('control_success', 'mean')
    ).reset_index()

    print("\n3.1 game_type Overall Statistics:")
    print(gt_stats.to_string(index=False))

    print("\n3.2 game_type Crosstab with Season (Row Counts):")
    ct_season = pd.crosstab(df['game_type'], df['season'], margins=True)
    print(ct_season)

    print("\n3.3 game_type Crosstab with Season (Column Proportions %):")
    ct_prop = pd.crosstab(df['game_type'], df['season'], normalize='columns') * 100
    print(ct_prop.round(2))

    # 3.4 Correlation / Mutual Information between game_type and season
    mi = mutual_info_score(df['game_type'].astype(str), df['season'].astype(str))
    print(f"\n3.4 Mutual Information between game_type and season: {mi:.6f}")

    chi2_gt, p_val_gt, dof_gt, _ = chi2_contingency(pd.crosstab(df['game_type'], df['season']))
    print(f"Chi-Square Test (game_type vs season): chi2={chi2_gt:.2f}, p-value={p_val_gt:.4e}")

    # 3.5 Check test.csv sample game_type values
    df_test = pd.read_csv(config.TEST_PATH)
    test_gt_unique = df_test['game_type'].unique().tolist()
    train_gt_unique = df['game_type'].unique().tolist()
    print(f"\n3.5 test.csv sample game_type unique values: {test_gt_unique}")
    print(f"train.csv game_type unique values: {train_gt_unique}")
    print(f"All test.csv game_type values exist in train.csv: {set(test_gt_unique).issubset(set(train_gt_unique))}")

    # 3.6 CV experiment: WITHOUT game_type vs WITH game_type
    folds = get_cv_folds(df, strategy="time")
    results_gt = []

    for fi, fold in enumerate(folds):
        df_tr = df.iloc[fold.train_idx].reset_index(drop=True)
        df_va = df.iloc[fold.val_idx].reset_index(drop=True)

        prep = PitchPreprocessor()
        prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        y_tr = df_tr[config.TARGET_COL].values
        y_va = df_va[config.TARGET_COL].values

        # Model A: WITH game_type
        cat_cols = [c for c in X_tr.columns if c in config.CATEGORICAL_COLS + config.DERIVED_CATEGORICAL_COLS or c == config.TRACKMAN_MATCH_FLAG_COL]
        mA = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
        mA.fit(X_tr, y_tr, categorical_feature=[X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns])
        auc_with_gt = roc_auc_score(y_va, mA.predict_proba(X_va)[:, 1])

        # Model B: WITHOUT game_type
        X_tr_no_gt = X_tr.drop(columns=['game_type']) if 'game_type' in X_tr.columns else X_tr
        X_va_no_gt = X_va.drop(columns=['game_type']) if 'game_type' in X_va.columns else X_va
        cat_cols_no_gt = [c for c in cat_cols if c != 'game_type']

        mB = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, random_state=42, verbosity=-1, n_jobs=-1)
        mB.fit(X_tr_no_gt, y_tr, categorical_feature=[X_tr_no_gt.columns.get_loc(c) for c in cat_cols_no_gt if c in X_tr_no_gt.columns])
        auc_no_gt = roc_auc_score(y_va, mB.predict_proba(X_va_no_gt)[:, 1])

        results_gt.append({
            'fold': fi,
            'val_season': fold.val_season,
            'auc_with_gt': auc_with_gt,
            'auc_no_gt': auc_no_gt,
            'auc_diff': auc_no_gt - auc_with_gt
        })
        print(f"  Fold {fi} (val={fold.val_season}): WITH game_type AUC={auc_with_gt:.6f} | WITHOUT game_type AUC={auc_no_gt:.6f} | ΔAUC={auc_no_gt - auc_with_gt:+.6f}")

    res_gt_df = pd.DataFrame(results_gt)
    print("\nComparison: WITH game_type vs WITHOUT game_type:")
    print(res_gt_df.to_string(index=False))
    print(f"Mean AUC WITH game_type:    {res_gt_df['auc_with_gt'].mean():.6f}")
    print(f"Mean AUC WITHOUT game_type: {res_gt_df['auc_no_gt'].mean():.6f}")
    print(f"Mean ΔAUC (no_gt - with_gt): {res_gt_df['auc_diff'].mean():+.6f}")

    return gt_stats, ct_season, ct_prop, res_gt_df


# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Loading train.csv ...")
    df = pd.read_csv(config.TRAIN_PATH)
    print(f"Loaded {len(df):,} rows.")

    season_target, res_no_season, splits_df = run_task1(df)
    stats_df, ct_gt, ct_target = run_task2(df)
    gt_stats, ct_season, ct_prop, res_gt_df = run_task3(df)

    t_end = time.perf_counter()
    print(f"\n=======================================================")
    print(f"ALL DIAGNOSIS TASKS COMPLETED in {t_end - t_start:.2f}s")
    print(f"=======================================================")
