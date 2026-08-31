"""
cfa_latent_features.py — CFA(Confirmatory Factor Analysis) 기반 숨겨진 변수 추정

목적:
  48개 관측 변수에서 이론적으로 설계된 4개 잠재 변수(latent factors)를 추출하여
  원래 피처에 factor scores로 추가합니다.

잠재 변수 설계 (도메인 지식 기반):
  F1: Pitcher_Skill (투수 역량) — 투수의 누적 성적/능력 지표
  F2: Game_Pressure (경기 압박) — 경기 상황 압박 정도
  F3: Pitcher_Physics (투구 물리량) — Trackman 물리적 측정치
  F4: Count_State (카운트/상황) — 볼카운트 및 이닝 상태

접근 방식:
  - CFA는 각 관측 변수가 특정 잠재 변수에만 로딩되도록 사전 지정합니다.
  - sklearn.decomposition.FactorAnalysis를 사용하여 각 잠재 요인별
    할당된 변수 서브셋에 대해 1-factor 모델을 적합하고 factor score를 추출합니다.
  - 이론 기반 변수 할당 + Maximum Likelihood 추정 + Bartlett factor scoring

라이브러리: scikit-learn (FactorAnalysis), pandas, numpy, scipy
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scipy import stats as scipy_stats

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ============================================================
# 1. CFA 모델 설계: 4개 잠재 변수와 관측 변수 매핑
# ============================================================

# 잠재 변수별 관측 변수 할당 (도메인 이론 기반)
CFA_MODEL_SPEC: Dict[str, List[str]] = {
    # F1: 투수 역량 (Pitcher Skill/Ability)
    # 투수의 누적 성적 및 투구 배합 능력을 반영하는 잠재 변수
    "Pitcher_Skill": [
        "asof_pitcher_success_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "asof_pitcher_n",
        "asof_pitcher_pitchmix_n",
    ],

    # F2: 경기 압박 (Game Pressure)
    # 경기 상황에서 발생하는 심리적/전술적 압박을 반영
    "Game_Pressure": [
        "li",                         # Leverage Index (압박 지수)
        "home_win_expectancy",
        "away_win_expectancy",
        "score_diff_pitcher_team",
        "run_top_before",
        "run_bot_before",
        "run_total_before",
        "score_diff_home",
        "num_runners_on",
        "runner_on_1b",
        "runner_on_2b",
        "runner_on_3b",
    ],

    # F3: 투구 물리량 (Pitcher Physics / Trackman Metrics)
    # 투구의 물리적 특성 (구속, 회전수, 무브먼트)
    "Pitcher_Physics": [
        "tkm_rel_speed_mean",
        "tkm_rel_speed_std",
        "tkm_spin_rate_mean",
        "tkm_spin_rate_std",
        "tkm_induced_vert_break_mean",
        "tkm_induced_vert_break_std",
        "tkm_horz_break_mean",
        "tkm_horz_break_std",
        "tkm_extension_mean",
        "tkm_extension_std",
        "tkm_rel_height_mean",
        "tkm_rel_height_std",
        "tkm_rel_side_mean",
        "tkm_rel_side_std",
        "tkm_zone_speed_mean",
        "tkm_zone_speed_std",
        "tkm_n_pitches",
    ],

    # F4: 카운트/이닝 상태 (Count & Inning State)
    # 투구 시점의 카운트 및 이닝 정보로 구성된 상황 변수
    "Count_State": [
        "balls_before",
        "strikes_before",
        "outs_before",
        "inning",
        "game_month",
        "game_dayofweek",
        "asof_pitcher_middle_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
    ],
}


# ============================================================
# 2. Factorability Tests (Bartlett's Sphericity, KMO)
# ============================================================

def bartlett_sphericity_test(X: np.ndarray) -> Tuple[float, float]:
    """
    Bartlett's Test of Sphericity.
    H0: 상관행렬이 단위행렬이다 (변수 간 상관 없음)
    → p < 0.05이면 요인분석 적합
    """
    n, p = X.shape
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    # Handle singular correlation matrices
    det = np.linalg.det(corr_matrix)
    if det <= 0:
        return np.inf, 0.0  # Singular → definitely not identity → significant
    
    chi_sq = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = scipy_stats.chi2.sf(chi_sq, df)
    return chi_sq, p_value


def calculate_kmo(X: np.ndarray) -> float:
    """
    Kaiser-Meyer-Olkin (KMO) 측정.
    KMO > 0.6: 적합, > 0.8: 매우 적합
    """
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    # Partial correlation matrix
    try:
        inv_corr = np.linalg.pinv(corr_matrix)
    except np.linalg.LinAlgError:
        return np.nan
    
    # Diagonal scaling for partial correlations
    diag_inv = np.diag(inv_corr)
    diag_sqrt = np.sqrt(np.abs(diag_inv))
    
    # Avoid division by zero
    diag_sqrt[diag_sqrt == 0] = 1e-10
    
    partial_corr = -inv_corr / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(partial_corr, 1.0)
    
    # KMO formula
    corr_sq_sum = np.sum(corr_matrix ** 2) - np.sum(np.diag(corr_matrix ** 2))
    partial_sq_sum = np.sum(partial_corr ** 2) - np.sum(np.diag(partial_corr ** 2))
    
    denominator = corr_sq_sum + partial_sq_sum
    if denominator == 0:
        return np.nan
    
    kmo = corr_sq_sum / denominator
    return kmo


# ============================================================
# 3. CFA Factor Score Extractor 클래스
# ============================================================

class CFALatentFeatureExtractor:
    """
    Confirmatory Factor Analysis 기반 잠재 변수 추출기.
    
    각 잠재 요인에 할당된 관측 변수 서브셋에 대해:
    1. 표준화 (StandardScaler)
    2. Factor Analysis (sklearn, n_components=1, ML estimation)
    3. Factor Score 추출 (Bartlett regression method)
    
    결과: 최대 4개의 factor score 컬럼이 원본 DataFrame에 추가됨.
    """

    def __init__(
        self,
        model_spec: Dict[str, List[str]] = None,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.model_spec = model_spec or CFA_MODEL_SPEC
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Fitted artifacts
        self.scalers: Dict[str, StandardScaler] = {}
        self.fa_models: Dict[str, FactorAnalysis] = {}
        self.fit_stats: Dict[str, Dict] = {}
        self.fitted_cols: Dict[str, List[str]] = {}
        self.is_fitted: bool = False

    def _validate_columns(self, df: pd.DataFrame, factor_name: str, cols: List[str]) -> List[str]:
        """DataFrame에 존재하는 컬럼만 필터링."""
        available = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  [CFA] {factor_name}: {len(missing)} columns missing (skipped): "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        if len(available) < 3:
            print(f"  [CFA WARNING] {factor_name}: Only {len(available)} variables. "
                  f"Min 3 required. Skipping.")
            return []
        return available

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "CFALatentFeatureExtractor":
        """
        CFA 모델을 학습합니다.
        
        Parameters:
            df: 학습 데이터 (raw 또는 preprocessed DataFrame)
            verbose: 진행 상황 출력 여부
        
        Returns:
            self (fitted)
        """
        if verbose:
            print("=" * 60)
            print("CFA (Confirmatory Factor Analysis) Model Fitting")
            print("=" * 60)
            print(f"  Input shape: {df.shape}")
            print(f"  Factors to extract: {list(self.model_spec.keys())}")
            print(f"  Estimation: Maximum Likelihood (sklearn FactorAnalysis)")
            print()

        for factor_name, indicator_cols in self.model_spec.items():
            if verbose:
                print(f"--- Factor: {factor_name} ---")

            # 1. Validate columns
            available_cols = self._validate_columns(df, factor_name, indicator_cols)
            if not available_cols:
                continue

            # 2. Extract subset and handle missing values
            X_raw = df[available_cols].copy()
            
            # Impute missing with median (FA requires complete data)
            medians = X_raw.median()
            X_imputed = X_raw.fillna(medians)
            
            # 3. Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # 4. Check factorability
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bart_chi2, bart_p = bartlett_sphericity_test(X_scaled)
                kmo_value = calculate_kmo(X_scaled)
            
            if verbose:
                print(f"  Variables: {len(available_cols)}")
                print(f"  KMO: {kmo_value:.4f} "
                      f"({'adequate' if kmo_value >= 0.5 else 'POOR'})")
                bart_sig = "significant" if bart_p < 0.05 else "NOT significant"
                print(f"  Bartlett's p-value: {bart_p:.2e} ({bart_sig})")

            # 5. Fit Factor Analysis (1 component = 1 latent factor per CFA block)
            fa = FactorAnalysis(
                n_components=1,
                max_iter=self.max_iter,
                random_state=self.random_state,
                svd_method="lapack"
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fa.fit(X_scaled)
            except Exception as e:
                print(f"  [CFA ERROR] {factor_name}: FA fitting failed: {e}")
                continue

            # 6. Store artifacts
            self.scalers[factor_name] = scaler
            self.fa_models[factor_name] = fa
            self.fitted_cols[factor_name] = available_cols
            
            # Compute fit statistics
            loadings = fa.components_[0]  # shape: (n_features,)
            noise_variance = fa.noise_variance_
            communalities = 1.0 - noise_variance
            
            # Eigenvalue equivalent (sum of squared loadings)
            eigenvalue = np.sum(loadings ** 2)
            variance_explained = eigenvalue / len(available_cols)
            
            factor_stats = {
                "n_variables": len(available_cols),
                "variables": available_cols,
                "loadings": dict(zip(available_cols, loadings.tolist())),
                "eigenvalue": eigenvalue,
                "variance_explained_ratio": variance_explained,
                "communalities": dict(zip(available_cols, communalities.tolist())),
                "kmo": kmo_value,
                "bartlett_chi2": bart_chi2,
                "bartlett_p": bart_p,
                "noise_variance": dict(zip(available_cols, noise_variance.tolist())),
                "log_likelihood": fa.score(X_scaled),
            }
            self.fit_stats[factor_name] = factor_stats

            if verbose:
                print(f"  Eigenvalue (sum λ²): {eigenvalue:.4f}")
                print(f"  Variance explained: {variance_explained*100:.2f}%")
                print(f"  Log-likelihood: {factor_stats['log_likelihood']:.4f}")
                # Top 5 loadings by absolute value
                sorted_loadings = sorted(
                    zip(available_cols, loadings), key=lambda x: abs(x[1]), reverse=True
                )
                print(f"  Top-5 loadings (absolute):")
                for col, load in sorted_loadings[:5]:
                    print(f"    {col}: {load:+.4f}")
                print()

        self.is_fitted = True
        if verbose:
            n_fitted = len(self.fa_models)
            print(f"✓ CFA fitting complete. {n_fitted} factor(s) successfully fitted.")
            print("=" * 60)
        
        return self

    def transform(self, df: pd.DataFrame, prefix: str = "cfa_") -> pd.DataFrame:
        """
        학습된 CFA 모델로 factor scores를 추출하여 DataFrame에 추가합니다.
        
        Factor scoring 방식: sklearn FactorAnalysis.transform()은
        Bartlett-method equivalent scoring을 수행합니다:
          score = X @ Λ @ (ΛΛ' + Ψ)^{-1}
        
        Parameters:
            df: 변환할 DataFrame
            prefix: factor score 컬럼 접두사
        
        Returns:
            factor scores가 추가된 DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("CFALatentFeatureExtractor is not fitted. Call .fit() first.")

        df_out = df.copy()
        
        for factor_name, fa in self.fa_models.items():
            available_cols = self.fitted_cols[factor_name]
            scaler = self.scalers[factor_name]
            
            # Extract and impute
            X_raw = df_out[available_cols].copy()
            X_imputed = X_raw.fillna(X_raw.median())
            
            # Standardize with fitted scaler
            X_scaled = scaler.transform(X_imputed)
            
            # Extract factor scores
            scores = fa.transform(X_scaled)  # shape: (n_samples, 1)
            
            # Add as new column
            col_name = f"{prefix}{factor_name}"
            df_out[col_name] = scores[:, 0]
        
        return df_out

    def fit_transform(self, df: pd.DataFrame, prefix: str = "cfa_", verbose: bool = True) -> pd.DataFrame:
        """fit + transform을 한번에 수행합니다."""
        self.fit(df, verbose=verbose)
        return self.transform(df, prefix=prefix)

    def get_factor_score_columns(self, prefix: str = "cfa_") -> List[str]:
        """추출된 factor score 컬럼명 리스트를 반환합니다."""
        return [f"{prefix}{name}" for name in self.fa_models.keys()]

    def summary(self) -> pd.DataFrame:
        """모든 팩터의 적합 통계를 DataFrame으로 요약합니다."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted yet.")
        
        rows = []
        for name, stats in self.fit_stats.items():
            rows.append({
                "Factor": name,
                "N_Variables": stats["n_variables"],
                "KMO": stats.get("kmo", None),
                "Bartlett_p": stats.get("bartlett_p", None),
                "Eigenvalue": stats.get("eigenvalue", None),
                "Var_Explained_%": (stats.get("variance_explained_ratio", 0) or 0) * 100,
                "Log_Likelihood": stats.get("log_likelihood", None),
            })
        return pd.DataFrame(rows)

    def get_loadings_matrix(self) -> pd.DataFrame:
        """
        전체 로딩 행렬을 DataFrame으로 반환합니다.
        CFA 구조: 각 변수는 하나의 팩터에만 로딩됩니다.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted yet.")
        
        all_vars = []
        all_loadings = {}
        
        for factor_name, stats in self.fit_stats.items():
            for var, loading in stats["loadings"].items():
                if var not in all_vars:
                    all_vars.append(var)
                all_loadings.setdefault(var, {})[factor_name] = loading
        
        # Build DataFrame
        factor_names = list(self.fa_models.keys())
        matrix = pd.DataFrame(0.0, index=all_vars, columns=factor_names)
        for var, loadings in all_loadings.items():
            for factor, val in loadings.items():
                matrix.loc[var, factor] = val
        
        return matrix


# ============================================================
# 4. 데이터 로드 함수
# ============================================================

def load_v33_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    v33 데이터를 로드합니다.
    
    Returns:
        (train_df, test_df): Raw DataFrame 쌍
    """
    # config.TRAIN_PATH uses BASE_DIR=submit_v33, data is in LG_data/open/data/
    data_dir = os.path.join(os.path.dirname(config.BASE_DIR), "..", "open", "data")
    data_dir = os.path.normpath(data_dir)
    
    # Try config path first, then fallback
    train_path = config.TRAIN_PATH
    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        train_path = "~/LG_data/open/data/train.csv"
    
    test_path = config.TEST_PATH
    if not os.path.exists(test_path):
        test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        test_path = "~/LG_data/open/data/test.csv"
    
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Train shape: {train_df.shape}")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Test shape: {test_df.shape}")
    
    return train_df, test_df


# ============================================================
# 5. CFA 파이프라인 실행 함수
# ============================================================

def run_cfa_pipeline(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    include_trackman: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], CFALatentFeatureExtractor]:
    """
    CFA 파이프라인을 실행합니다.
    
    Parameters:
        train_df: 학습 데이터
        test_df: 테스트 데이터 (None이면 train만 처리)
        include_trackman: Trackman 피처 포함 여부
        verbose: 상세 출력
    
    Returns:
        (train_with_factors, test_with_factors, extractor)
    """
    # Model spec 조정 (trackman 미포함 시)
    model_spec = dict(CFA_MODEL_SPEC)
    if not include_trackman or "Pitcher_Physics" in model_spec:
        # Check if trackman cols exist
        tkm_cols = model_spec.get("Pitcher_Physics", [])
        available_tkm = [c for c in tkm_cols if c in train_df.columns]
        if len(available_tkm) < 3:
            model_spec.pop("Pitcher_Physics", None)
            if verbose:
                print("[CFA] Pitcher_Physics factor removed (Trackman columns not available)")
    
    # Initialize extractor
    extractor = CFALatentFeatureExtractor(
        model_spec=model_spec,
        max_iter=1000,
        random_state=42
    )
    
    # Fit on train
    extractor.fit(train_df, verbose=verbose)
    
    # Transform train
    train_out = extractor.transform(train_df)
    
    # Transform test
    test_out = None
    if test_df is not None:
        test_out = extractor.transform(test_df)
    
    return train_out, test_out, extractor


# ============================================================
# 6. 메인 실행
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CFA-Based Latent Variable Estimation for v33 Pipeline")
    print("=" * 70 + "\n")
    
    # Step 1: Load v33 data (48 variables)
    train_df, test_df = load_v33_data()
    
    # Step 2: CFA 수행 (raw 데이터에서; Trackman은 preprocessing 후 사용)
    print("\n[Phase 1] CFA on Raw Features (without Trackman)")
    print("-" * 50)
    
    train_with_factors, test_with_factors, extractor = run_cfa_pipeline(
        train_df=train_df,
        test_df=test_df,
        include_trackman=False,
        verbose=True
    )
    
    # Step 3: 결과 요약
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    
    factor_cols = extractor.get_factor_score_columns()
    print(f"\n[Factor Score Columns Added]: {factor_cols}")
    print(f"[Original columns]: {train_df.shape[1]}")
    print(f"[After CFA columns]: {train_with_factors.shape[1]}")
    print(f"[New factor features]: {train_with_factors.shape[1] - train_df.shape[1]}")
    
    # Summary table
    print("\n[CFA Fit Summary]:")
    summary_df = extractor.summary()
    print(summary_df.to_string(index=False))
    
    # Loadings matrix
    print("\n[CFA Loadings Matrix (confirmatory structure)]:")
    loadings_df = extractor.get_loadings_matrix()
    print(loadings_df.round(4).to_string())
    
    # Factor score statistics
    print("\n[Factor Score Statistics (Train)]:")
    if factor_cols:
        print(train_with_factors[factor_cols].describe().round(4).to_string())
    
    # Inter-factor correlations
    if len(factor_cols) >= 2:
        print("\n[Inter-Factor Correlations]:")
        print(train_with_factors[factor_cols].corr().round(4).to_string())
    
    # Target correlation (important for predictive utility)
    if config.TARGET_COL in train_with_factors.columns and factor_cols:
        print("\n[Factor-Target Correlations]:")
        for col in factor_cols:
            corr = train_with_factors[col].corr(train_with_factors[config.TARGET_COL])
            print(f"  {col} ↔ {config.TARGET_COL}: {corr:.6f}")
    
    # Step 4: Save output
    output_dir = os.path.join(config.BASE_DIR, "model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save factor scores
    factor_output_path = os.path.join(output_dir, "cfa_factor_scores_train.csv")
    save_cols = [config.ID_COL] + factor_cols if config.ID_COL in train_with_factors.columns else factor_cols
    train_with_factors[save_cols].to_csv(factor_output_path, index=False)
    print(f"\n[Saved] Factor scores (train) → {factor_output_path}")
    
    if test_with_factors is not None:
        factor_test_path = os.path.join(output_dir, "cfa_factor_scores_test.csv")
        save_cols_test = [config.ID_COL] + factor_cols if config.ID_COL in test_with_factors.columns else factor_cols
        test_with_factors[save_cols_test].to_csv(factor_test_path, index=False)
        print(f"[Saved] Factor scores (test) → {factor_test_path}")
    
    # Save extractor
    import joblib
    extractor_path = os.path.join(output_dir, "cfa_extractor.pkl")
    joblib.dump(extractor, extractor_path)
    print(f"[Saved] CFA extractor → {extractor_path}")
    
    print("\n" + "=" * 60)
    print(f"✓ CFA Latent Feature Extraction Complete!")
    print(f"  → {len(factor_cols)} latent variables added to DataFrame")
    print(f"  → Final train shape: {train_with_factors.shape}")
    if test_with_factors is not None:
        print(f"  → Final test shape: {test_with_factors.shape}")
    print("=" * 60)
