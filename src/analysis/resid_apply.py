"""resid — 체인 잔차 후처리 (2026-08-30).

체인(A/B 블렌드 → U 오프셋 → in-game → pcxh → ctr) 이 남긴 잔차를 test.csv 원본 43컬럼으로
예측해 확률에 직접 더한다. 표·모델은 전부 학습 때 만들어 resid_tables/ 에 저장했고
여기서는 predict 만 한다.

규정(2-4) 안전: apply_resid 는 **각 행 자신의 43개 컬럼**만 입력으로 받는다. LightGBM 은
행마다 독립적으로 트리를 타므로 프레임 통계·행 순서·다른 test 행을 일절 참조하지 않는다.
`check_row_locality_resid` 가 단일행/셔플/부분집합 == 전체프레임 을 기계 검증한다.

검증(정직한 3분할): 학습 2021+2022 → β 는 2023 에서 적합 → **2024 에서 +6.65 폴드 실현**.
배포본은 같은 기하를 한 해 밀었다: 학습 2022+2023 → β 는 2024 → 2025 예측(격차 2년 동일).
⚠️ β 를 학습에 포함된 해에서 적합하면 β 가 폭주해 −5000 이 난다(실측). 반드시 분리할 것.
"""
import json
import os

import numpy as np
import pandas as pd

__all__ = ["load_resid", "resid_delta", "apply_resid", "check_row_locality_resid"]
EPS = 1e-6


def load_resid(tables_dir):
    import lightgbm as lgb
    p = json.load(open(os.path.join(tables_dir, "resid_params.json"), encoding="utf-8"))
    return {"model": lgb.Booster(model_file=os.path.join(tables_dir, "resid_model.txt")),
            "params": p}


def resid_delta(df, T):
    """행별 확률 보정량 (float64). 컬럼이 없으면 NaN 으로 채워 모델에 맡긴다."""
    cols = T["params"]["feats"]
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df.columns
                      else np.nan for c in cols}, index=df.index)
    return float(T["params"]["beta"]) * T["model"].predict(X.to_numpy(np.float64))


def apply_resid(test_df, tables_dir, p):
    T = load_resid(tables_dir) if isinstance(tables_dir, str) else tables_dir
    p = np.asarray(p, dtype=np.float64).ravel()
    if len(p) != len(test_df):
        raise ValueError(f"apply_resid: p has {len(p)} rows, test_df has {len(test_df)}")
    out = np.clip(p + resid_delta(test_df, T), EPS, 1 - EPS)
    if not np.all(np.isfinite(out)):
        raise AssertionError("apply_resid produced non-finite probabilities")
    return out


def check_row_locality_resid(df, tables_dir, n_rows=32, seed=0):
    T = load_resid(tables_dir) if isinstance(tables_dir, str) else tables_dir
    full = resid_delta(df, T)
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(df), size=min(n_rows, len(df)), replace=False):
        one = resid_delta(df.iloc[[i]], T)
        assert abs(one[0] - full[i]) < 1e-9, f"row {i}: {one[0]} != {full[i]}"
    perm = rng.permutation(len(df))
    assert np.allclose(resid_delta(df.iloc[perm], T), full[perm], atol=1e-9), "shuffled differs"
    half = perm[: len(df) // 2]
    assert np.allclose(resid_delta(df.iloc[half], T), full[half], atol=1e-9), "subset differs"
    return True
