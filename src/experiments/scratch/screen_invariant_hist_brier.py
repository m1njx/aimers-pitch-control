"""Low-variance, era-invariant Brier regressor; honest forward-season screen."""
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.expanduser("~/LG_data")
TRAIN = os.path.join(ROOT, "open/data/train.csv")
OUT = os.path.join(ROOT, "scratch/invariant_hist_brier_results.json")

RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
COUNT_COLS = ["balls_before", "strikes_before", "outs_before", "inning", "li",
              "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
              "pitcher_hand", "batter_hand", "top_bottom"]
N_COLS = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]


def features(d):
    x = d[RATE_COLS + COUNT_COLS + N_COLS].copy()
    x["top_bottom"] = d["top_bottom"].astype(str).map({"B": 0, "T": 1})
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in N_COLS:
        x[c] = np.log1p(pd.to_numeric(x[c], errors="coerce").clip(lower=0))
    # Compact invariant state encodings; no IDs, dates, teams or test-batch statistics.
    x["count_state"] = d.balls_before.clip(0, 3) * 3 + d.strikes_before.clip(0, 2)
    x["base_state_num"] = ((d.runner_on_1b.fillna(0) > 0).astype(int) +
                           2 * (d.runner_on_2b.fillna(0) > 0).astype(int) +
                           4 * (d.runner_on_3b.fillna(0) > 0).astype(int))
    x["platoon"] = (d.pitcher_hand.astype(str) == d.batter_hand.astype(str)).astype(int)
    return x.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def skill(y, p):
    r = y.mean(); b = np.mean((np.clip(p, 1e-6, 1-1e-6) - y) ** 2)
    return 100000 * (1 - b / (r * (1-r)))


def basepred(vs):
    c = np.load(os.path.join(ROOT, "scratch/cache_final", f"final_val{vs}.npz"))
    m = np.load(os.path.join(ROOT, "scratch/multitask_aux_preds", f"val{vs}.npz"))["p"]
    g = .15*(c["p_lgb"]-.007) + .75*(c["p_cb"]-.008) + .10*(c["p_xgb"]-.006)
    return np.clip(.5 + 1.1*((.68*g + .32*m)-.5) - .0045192086, 1e-6, 1-1e-6)


def main():
    df = pd.read_csv(TRAIN)
    y = df.control_success.to_numpy(np.float32)
    X = features(df)
    result = {}
    for vs in (2022, 2023, 2024):
        mt = (df.season < vs).to_numpy(); mv = (df.season == vs).to_numpy()
        model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=.05,
            max_iter=180, max_leaf_nodes=15, max_depth=5, min_samples_leaf=800,
            l2_regularization=30, random_state=123, early_stopping=True,
            validation_fraction=.08, n_iter_no_change=20)
        model.fit(X.loc[mt], y[mt])
        ph = np.clip(model.predict(X.loc[mv]), 1e-6, 1-1e-6)
        pb = basepred(vs); yy = y[mv]
        grid = {}
        for w in (0, .025, .05, .075, .10, .15, .20, .30, .40):
            pp = (1-w)*pb + w*ph
            grid[str(w)] = {"skill": float(skill(yy, pp)),
                            "gain": float(skill(yy, pp)-skill(yy, pb))}
        result[str(vs)] = {"hist": float(skill(yy, ph)), "base": float(skill(yy, pb)),
                           "iterations": int(model.n_iter_), "grid": grid}
        with open(OUT, "w") as f: json.dump(result, f, indent=2)
        print(vs, result[str(vs)], flush=True)


if __name__ == "__main__":
    main()
