"""
cv_utils.py — Cross-Validation utilities for DACON Aimers 9th competition.

Implements the confirmed CV strategy (CV_STRATEGY = "time") based on:
  - train seasons: 2019–2024
  - test season: 2025 (confirmed from test.csv sample, 11_test_split_strategy.md)
  - Correct CV approach: season-based expanding window (temporal split)

Also provides GroupKFold alternative for comparison experiments.
Both can be called via get_cv_folds(df, strategy="time"|"group").

Each time fold now returns a FoldInfo named tuple with:
  (train_idx, val_idx, fold_max_season, val_season, fold_notes)
This enables automatic fold-aware as_of_season wiring for TrackmanFeatureBuilder.
"""

from typing import List, NamedTuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config


class FoldInfo(NamedTuple):
    """Container for a single CV fold's metadata alongside index arrays.

    Attributes:
        train_idx      : np.ndarray of integer row positions for train split.
        val_idx        : np.ndarray of integer row positions for val split.
        fold_max_season: Maximum season in the train split.
                         Use this as as_of_season in TrackmanFeatureBuilder.fit().
        val_season     : The validation season (or None for GroupKFold folds).
        notes          : Human-readable description of this fold.
    """
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold_max_season: Optional[int]
    val_season: Optional[int]
    notes: str


# ── COVID-season flag ──────────────────────────────────────────────────────────
# 2020 was a COVID-shortened season (144 games vs ~144, but started in May rather
# than March/April). Folds that include 2020 in their train set may have slightly
# different distributions (fewer early-season pitches, fewer innings pitched per game).
# This is recorded in FoldInfo.notes so experiment logs can flag it automatically.
_COVID_SEASON = 2020
_COVID_NOTE = (
    "⚠️  train includes 2020 COVID-shortened season (May-Oct only, no March/April). "
    "Results may differ vs full-season folds."
)


def get_time_folds(
    df: pd.DataFrame,
    min_train_seasons: int = None,
    season_col: str = None,
) -> List[FoldInfo]:
    """Season-based expanding window CV folds (recommended strategy).

    Each fold uses all seasons up to val_season-1 as train, val_season as validation.
    Minimum `min_train_seasons` seasons required before starting validation.

    Args:
        df (pd.DataFrame): Full training DataFrame containing season_col.
        min_train_seasons (int): Minimum train seasons before first val fold.
                                 Defaults to config.CV_MIN_TRAIN_SEASONS.
        season_col (str): Season column name. Defaults to config.CV_SEASON_COL.

    Returns:
        List[FoldInfo]: Each FoldInfo contains (train_idx, val_idx,
                        fold_max_season, val_season, notes).

    Example folds with CV_SEASONS=[2019..2024], CV_MIN_TRAIN_SEASONS=3:
        Fold 0: train=2019-2021 (fold_max_season=2021), val=2022
        Fold 1: train=2019-2022 (fold_max_season=2022), val=2023
        Fold 2: train=2019-2023 (fold_max_season=2023), val=2024
    """
    if min_train_seasons is None:
        min_train_seasons = config.CV_MIN_TRAIN_SEASONS
    if season_col is None:
        season_col = config.CV_SEASON_COL

    seasons = config.CV_SEASONS  # [2019, 2020, 2021, 2022, 2023, 2024]
    folds: List[FoldInfo] = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        fold_max_season = max(train_seasons)  # = seasons[i-1]

        train_mask = df[season_col].isin(train_seasons)
        val_mask = df[season_col] == val_season

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Build fold notes (flag COVID season if present in train)
        notes_parts = [
            f"train={train_seasons} → val={val_season}",
            f"tkm_as_of_season={fold_max_season}",
        ]
        if _COVID_SEASON in train_seasons:
            notes_parts.append(_COVID_NOTE)

        notes = " | ".join(notes_parts)

        print(f"[get_time_folds] Fold {len(folds)}: "
              f"train seasons={train_seasons} ({len(train_idx):,} rows) | "
              f"val season={val_season} ({len(val_idx):,} rows) | "
              f"fold_max_season={fold_max_season}")

        folds.append(FoldInfo(
            train_idx=train_idx,
            val_idx=val_idx,
            fold_max_season=fold_max_season,
            val_season=val_season,
            notes=notes,
        ))

    return folds


def get_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = None,
) -> List[FoldInfo]:
    """GroupKFold CV on pitcher_id (alternative strategy for comparison).

    WARNING: Does NOT respect temporal order. Use for comparison only — NOT
    as the primary CV strategy. fold_max_season is set to None (undefined).

    Args:
        df (pd.DataFrame): Full training DataFrame.
        n_splits (int): Number of CV folds. Defaults to 5.
        group_col (str): Group column. Defaults to config.CV_GROUP_COL.

    Returns:
        List[FoldInfo]: Each FoldInfo contains (train_idx, val_idx,
                        fold_max_season=None, val_season=None, notes).
    """
    if group_col is None:
        group_col = config.CV_GROUP_COL

    print(f"[get_group_folds] WARNING: GroupKFold on '{group_col}' does not preserve "
          f"temporal order. Use for comparison only. fold_max_season=None.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    folds: List[FoldInfo] = []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        n_tr = len(np.unique(groups[train_idx]))
        n_va = len(np.unique(groups[val_idx]))
        notes = (
            f"GroupKFold fold {fold_i} on '{group_col}' | "
            f"train={len(train_idx):,} rows ({n_tr} groups) | "
            f"val={len(val_idx):,} rows ({n_va} groups) | "
            f"⚠️  temporal order NOT preserved"
        )
        print(f"[get_group_folds] Fold {fold_i}: "
              f"train={len(train_idx):,} rows ({n_tr} pitchers) | "
              f"val={len(val_idx):,} rows ({n_va} pitchers)")
        folds.append(FoldInfo(
            train_idx=train_idx,
            val_idx=val_idx,
            fold_max_season=None,
            val_season=None,
            notes=notes,
        ))

    return folds


def get_cv_folds(
    df: pd.DataFrame,
    strategy: str = None,
    **kwargs,
) -> List[FoldInfo]:
    """Dispatch to the correct CV fold generator based on strategy.

    Args:
        df (pd.DataFrame): Training DataFrame.
        strategy (str): "time" (default) or "group". Defaults to config.CV_STRATEGY.
        **kwargs: Forwarded to get_time_folds or get_group_folds.

    Returns:
        List[FoldInfo]: Each element has (train_idx, val_idx, fold_max_season,
                        val_season, notes).
    """
    if strategy is None:
        strategy = config.CV_STRATEGY

    if strategy == "time":
        return get_time_folds(df, **kwargs)
    elif strategy == "group":
        return get_group_folds(df, **kwargs)
    else:
        raise ValueError(f"Unknown CV strategy: '{strategy}'. Choose 'time' or 'group'.")


if __name__ == "__main__":
    print("=== CV Utils Smoke Test ===\n")
    df = pd.read_csv(config.TRAIN_PATH, usecols=[config.CV_SEASON_COL, config.CV_GROUP_COL])
    print(f"Loaded {len(df):,} rows. Seasons: {sorted(df[config.CV_SEASON_COL].unique())}\n")

    print("── Strategy: TIME (recommended) ──")
    time_folds = get_cv_folds(df, strategy="time")
    print(f"Total folds: {len(time_folds)}")
    for fi, fold in enumerate(time_folds):
        print(f"  Fold {fi}: fold_max_season={fold.fold_max_season}, "
              f"val_season={fold.val_season}")
        print(f"    Notes: {fold.notes}")

    print("\n── Strategy: GROUP (comparison) ──")
    group_folds = get_cv_folds(df, strategy="group", n_splits=5)
    print(f"Total folds: {len(group_folds)}")

    print("\n=== Smoke Test Passed ===")

    # Usage pattern for downstream scripts:
    print("\n── Usage pattern example ──")
    print("""
from cv_utils import get_cv_folds
from preprocessing import PitchPreprocessor

folds = get_cv_folds(df)  # returns List[FoldInfo]

for fold_i, fold in enumerate(folds):
    df_tr = df.iloc[fold.train_idx]
    df_va = df.iloc[fold.val_idx]

    prep = PitchPreprocessor()
    # fold.fold_max_season ensures TrackmanFeatureBuilder only uses
    # trackman data from seasons up to the fold's train boundary:
    prep.fit(df_tr, as_of_season=fold.fold_max_season, is_final=False)

    X_tr = prep.transform(df_tr)
    X_va = prep.transform(df_va)
    # model.fit(X_tr, y_tr) → score on X_va ...
""")
