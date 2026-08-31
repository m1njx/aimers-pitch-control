"""
game_theory_features.py — 게임 이론 기반 피처 엔지니어링

게임 이론(Game Theory) 관점에서 투수-타자 대결 상황의 전략적 균형점을 수치화합니다.

구현 피처:
1. gt_pressure_perf: 고압 상황에서의 투수 퍼포먼스 (Pressure-Performance Interaction)
   = log1p(li) × closeness × pitcher_success
   - 해석: 레버리지가 높고(li↑), 점수가 근소하며(closeness↑), 투수가 잘하고 있을 때(pitcher_success↑)
            투수의 제구 성공 확률이 높아지는 현상 (투수 집중력 + 경험 효과)

2. gt_critical_eq: 임계 균형점 (Critical Equilibrium under Maximum Pressure)
   = (outs==2) × scoring_pos × log1p(li) × pitcher_success
   - 해석: 2아웃 + 득점권 + 높은 레버리지 = 게임 이론에서의 "최대 긴장 상태"
            이 상황에서 투수의 누적 성공률이 높을수록 제구 성공 가능성이 증가
            (Mixed Strategy Nash Equilibrium에서 고능력 투수가 최적 전략을 선택할 확률↑)

설계 원칙:
- 미래 정보 사용 없음 (all features are past-only or current-state)
- NaN 처리: pitcher_success가 NaN인 경우 전체 평균(0.5)으로 대체
- closeness는 현재 점수 상태에서 산출 (score_diff_pitcher_team 기반)
- CFA 파이프라인 이후에 적용 가능 (독립적 모듈)

Dependencies:
- numpy, pandas (standard)
- 입력 DataFrame에 필요한 컬럼: li, score_diff_pitcher_team (또는 score_diff_home),
  outs_before, runner_on_2b, runner_on_3b, asof_pitcher_success_rate

Usage:
    from game_theory_features import GameTheoryFeatureBuilder, add_game_theory_features

    # 방법 1: 함수형 (간단)
    df = add_game_theory_features(df)

    # 방법 2: 클래스형 (파이프라인 통합)
    builder = GameTheoryFeatureBuilder()
    df = builder.transform(df)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ===========================================================================
# Constants
# ===========================================================================
PITCHER_SUCCESS_COL = "asof_pitcher_success_rate"
PITCHER_SUCCESS_FALLBACK = 0.5  # NaN 대체값 (prior = 50%)

# 점수차 컬럼 우선순위 (pitcher_team 기준이 더 정확)
SCORE_DIFF_COLS = ["score_diff_pitcher_team", "score_diff_home"]

# 출력 피처 이름
FEATURE_PRESSURE_PERF = "gt_pressure_perf"
FEATURE_CRITICAL_EQ = "gt_critical_eq"

GT_FEATURE_COLS = [FEATURE_PRESSURE_PERF, FEATURE_CRITICAL_EQ]


# ===========================================================================
# Core Feature Computation Functions
# ===========================================================================
def compute_closeness(score_diff: pd.Series) -> pd.Series:
    """점수 근소 지표 (Closeness) 계산.

    closeness = 1 / (1 + |score_diff|)

    Properties:
    - score_diff == 0 (동점) → closeness = 1.0
    - |score_diff| == 1 → closeness = 0.5
    - |score_diff| == 2 → closeness = 0.333
    - |score_diff| → ∞ → closeness → 0

    Args:
        score_diff: 점수차 시리즈 (양수 = 리드, 음수 = 뒤짐)

    Returns:
        closeness: [0, 1] 범위의 근소 지표
    """
    return 1.0 / (1.0 + score_diff.abs())


def compute_scoring_position(df: pd.DataFrame) -> pd.Series:
    """득점권 여부 계산 (2루 또는 3루 주자 있음).

    Args:
        df: runner_on_2b, runner_on_3b 컬럼 포함 DataFrame

    Returns:
        scoring_pos: 0/1 이진 시리즈
    """
    runner_2b = df.get("runner_on_2b", pd.Series(0, index=df.index))
    runner_3b = df.get("runner_on_3b", pd.Series(0, index=df.index))
    return ((runner_2b == 1) | (runner_3b == 1)).astype(np.float64)


def compute_gt_pressure_perf(
    li: pd.Series,
    closeness: pd.Series,
    pitcher_success: pd.Series
) -> pd.Series:
    """게임 이론 피처 1: Pressure-Performance Interaction.

    gt_pressure_perf = log1p(li) × closeness × pitcher_success

    Args:
        li: Leverage Index
        closeness: 점수 근소 지표 [0, 1]
        pitcher_success: 투수 누적 제구 성공률 [0, 1]

    Returns:
        gt_pressure_perf feature
    """
    return np.log1p(li) * closeness * pitcher_success


def compute_gt_critical_eq(
    outs_before: pd.Series,
    scoring_pos: pd.Series,
    li: pd.Series,
    pitcher_success: pd.Series
) -> pd.Series:
    """게임 이론 피처 2: Critical Equilibrium.

    gt_critical_eq = (outs==2) × scoring_pos × log1p(li) × pitcher_success

    Args:
        outs_before: 현재 아웃 카운트 (0, 1, 2)
        scoring_pos: 득점권 여부 (0/1)
        li: Leverage Index
        pitcher_success: 투수 누적 제구 성공률 [0, 1]

    Returns:
        gt_critical_eq feature
    """
    is_two_outs = (outs_before == 2).astype(np.float64)
    return is_two_outs * scoring_pos * np.log1p(li) * pitcher_success


# ===========================================================================
# Class-based Builder (Pipeline Integration)
# ===========================================================================
class GameTheoryFeatureBuilder:
    """게임 이론 기반 피처 빌더.

    Stateless transformer — fit()이 필요 없는 순수 변환기.
    모든 계산은 현재 행의 상태값만 사용 (no leakage).
    """

    def __init__(
        self,
        pitcher_success_col: str = PITCHER_SUCCESS_COL,
        pitcher_success_fallback: float = PITCHER_SUCCESS_FALLBACK,
        score_diff_cols: Optional[list] = None
    ):
        """
        Args:
            pitcher_success_col: 투수 성공률 컬럼명
            pitcher_success_fallback: NaN 대체값
            score_diff_cols: 점수차 컬럼 후보 리스트 (우선순위 순)
        """
        self.pitcher_success_col = pitcher_success_col
        self.pitcher_success_fallback = pitcher_success_fallback
        self.score_diff_cols = score_diff_cols or SCORE_DIFF_COLS

    def _get_score_diff(self, df: pd.DataFrame) -> pd.Series:
        """점수차 컬럼을 우선순위에 따라 선택."""
        for col in self.score_diff_cols:
            if col in df.columns:
                return df[col].fillna(0)
        raise KeyError(
            f"점수차 컬럼을 찾을 수 없습니다. "
            f"필요한 컬럼 중 하나: {self.score_diff_cols}"
        )

    def _get_pitcher_success(self, df: pd.DataFrame) -> pd.Series:
        """투수 성공률 추출 (NaN → fallback)."""
        if self.pitcher_success_col in df.columns:
            return df[self.pitcher_success_col].fillna(self.pitcher_success_fallback)
        else:
            print(f"  [WARNING] '{self.pitcher_success_col}' not found. Using fallback={self.pitcher_success_fallback}")
            return pd.Series(self.pitcher_success_fallback, index=df.index)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """게임 이론 피처를 계산하여 DataFrame에 추가.

        Args:
            df: 입력 DataFrame
            inplace: True이면 원본 수정, False이면 복사본 반환

        Returns:
            게임 이론 피처가 추가된 DataFrame
        """
        if not inplace:
            df = df.copy()

        print("[GameTheoryFeatureBuilder] Computing game theory features ...")

        # 1. 필요 변수 추출
        li = df["li"].fillna(0) if "li" in df.columns else pd.Series(0, index=df.index)
        score_diff = self._get_score_diff(df)
        pitcher_success = self._get_pitcher_success(df)
        outs_before = df["outs_before"] if "outs_before" in df.columns else pd.Series(0, index=df.index)

        # 2. 중간 변수 계산
        closeness = compute_closeness(score_diff)
        scoring_pos = compute_scoring_position(df)

        # 3. 게임 이론 피처 계산
        df[FEATURE_PRESSURE_PERF] = compute_gt_pressure_perf(li, closeness, pitcher_success)
        df[FEATURE_CRITICAL_EQ] = compute_gt_critical_eq(outs_before, scoring_pos, li, pitcher_success)

        # 4. 통계 출력
        print(f"  {FEATURE_PRESSURE_PERF}: mean={df[FEATURE_PRESSURE_PERF].mean():.4f}, "
              f"std={df[FEATURE_PRESSURE_PERF].std():.4f}, "
              f"nonzero_ratio={( df[FEATURE_PRESSURE_PERF] > 0).mean():.4f}")
        print(f"  {FEATURE_CRITICAL_EQ}: mean={df[FEATURE_CRITICAL_EQ].mean():.4f}, "
              f"std={df[FEATURE_CRITICAL_EQ].std():.4f}, "
              f"nonzero_ratio={(df[FEATURE_CRITICAL_EQ] > 0).mean():.4f}")

        return df

    def get_feature_names(self) -> list:
        """생성되는 피처 이름 리스트 반환."""
        return list(GT_FEATURE_COLS)


# ===========================================================================
# Functional API (Simple Usage)
# ===========================================================================
def add_game_theory_features(
    df: pd.DataFrame,
    pitcher_success_col: str = PITCHER_SUCCESS_COL,
    pitcher_success_fallback: float = PITCHER_SUCCESS_FALLBACK,
    inplace: bool = False
) -> pd.DataFrame:
    """게임 이론 피처를 DataFrame에 추가하는 편의 함수.

    Args:
        df: 입력 DataFrame
        pitcher_success_col: 투수 성공률 컬럼명
        pitcher_success_fallback: NaN 대체값
        inplace: True면 원본 수정

    Returns:
        게임 이론 피처가 추가된 DataFrame
    """
    builder = GameTheoryFeatureBuilder(
        pitcher_success_col=pitcher_success_col,
        pitcher_success_fallback=pitcher_success_fallback
    )
    return builder.transform(df, inplace=inplace)


# ===========================================================================
# Pipeline Integration Helper
# ===========================================================================
def run_game_theory_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pitcher_success_col: str = PITCHER_SUCCESS_COL,
    pitcher_success_fallback: float = PITCHER_SUCCESS_FALLBACK
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test 모두에 게임 이론 피처를 적용하는 파이프라인 함수.

    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        pitcher_success_col: 투수 성공률 컬럼명
        pitcher_success_fallback: NaN 대체값

    Returns:
        (train_df_with_gt, test_df_with_gt): 피처 추가된 DataFrame 튜플
    """
    builder = GameTheoryFeatureBuilder(
        pitcher_success_col=pitcher_success_col,
        pitcher_success_fallback=pitcher_success_fallback
    )

    print("=" * 60)
    print("[Game Theory Pipeline] Processing TRAIN data")
    print("=" * 60)
    train_out = builder.transform(train_df)

    print()
    print("=" * 60)
    print("[Game Theory Pipeline] Processing TEST data")
    print("=" * 60)
    test_out = builder.transform(test_df)

    return train_out, test_out


# ===========================================================================
# Standalone Execution & Verification
# ===========================================================================
if __name__ == "__main__":
    import os
    import sys
    import time as _time

    # Setup path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config

    print("=" * 70)
    print("GAME THEORY FEATURE ENGINEERING — VERIFICATION")
    print("=" * 70)

    t_start = _time.perf_counter()

    # 1. Load data
    print("\n[1/5] Loading data ...")
    # Resolve data paths: config.TRAIN_PATH uses BASE_DIR which is this script's dir
    # Actual data lives at ~/LG_data/open/data/
    lg_data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(lg_data_root, "open", "data", "train.csv")
    test_path = os.path.join(lg_data_root, "open", "data", "test.csv")

    # Fallback: try config paths
    if not os.path.exists(train_path):
        train_path = config.TRAIN_PATH
        test_path = config.TEST_PATH

    # Second fallback: absolute known path
    if not os.path.exists(train_path):
        train_path = "~/LG_data/open/data/train.csv"
        test_path = "~/LG_data/open/data/test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"  Train: {df_train.shape}")
    print(f"  Test:  {df_test.shape}")

    # 2. Apply game theory features
    print("\n[2/5] Computing game theory features ...")
    train_out, test_out = run_game_theory_pipeline(df_train, df_test)

    # 3. Verify output
    print("\n[3/5] Verification ...")
    for feat in GT_FEATURE_COLS:
        assert feat in train_out.columns, f"MISSING: {feat} in train"
        assert feat in test_out.columns, f"MISSING: {feat} in test"
        assert train_out[feat].isnull().sum() == 0, f"NaN found in train.{feat}"
        assert test_out[feat].isnull().sum() == 0, f"NaN found in test.{feat}"
    print("  ✅ All features present, no NaN values")

    # 4. Correlation with target
    print("\n[4/5] Target correlation analysis ...")
    target = train_out["control_success"]
    for feat in GT_FEATURE_COLS:
        corr = train_out[feat].corr(target)
        print(f"  {feat} ↔ control_success: r = {corr:.4f}")

    # 5. Save augmented data
    print("\n[5/5] Saving augmented datasets ...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)

    gt_train_path = os.path.join(output_dir, "gt_features_train.csv")
    gt_test_path = os.path.join(output_dir, "gt_features_test.csv")

    # Save only the GT feature columns (for merge later)
    train_out[["row_id"] + GT_FEATURE_COLS].to_csv(gt_train_path, index=False)
    test_out[["row_id"] + GT_FEATURE_COLS].to_csv(gt_test_path, index=False)
    print(f"  Saved: {gt_train_path}")
    print(f"  Saved: {gt_test_path}")

    t_end = _time.perf_counter()
    print(f"\n{'='*70}")
    print(f"DONE — Total time: {t_end - t_start:.2f}s")
    print(f"  Train columns: {len(train_out.columns)} (original {len(df_train.columns)} + {len(GT_FEATURE_COLS)} GT features)")
    print(f"  New features: {GT_FEATURE_COLS}")
    print(f"{'='*70}")
