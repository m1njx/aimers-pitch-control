"""agent3_calib.py — calibration schemes applied to era-neutralized deviation predictions."""
import numpy as np, pandas as pd

SEASON_R = {2019: 0.564670, 2020: 0.532712, 2021: 0.532762, 2022: 0.528920,
            2023: 0.499957, 2024: 0.486105}
FTS = {0: [2019, 2020, 2021], 1: [2019, 2020, 2021, 2022], 2: [2019, 2020, 2021, 2022, 2023]}


def r_hat(k, mode):
    ys = np.array([SEASON_R[s] for s in FTS[k]])
    if mode == 'last':
        return ys[-1]
    if mode == 'last_plus_meandelta':
        return ys[-1] + np.diff(ys).mean()
    if mode == 'last_plus_halfdelta':
        return ys[-1] + 0.5 * np.diff(ys).mean()
    if mode == 'last_plus_lastdelta':
        return ys[-1] + (ys[-1] - ys[-2])
    raise ValueError(mode)


def seg_keys(sd, kind):
    b = sd['balls_before'].clip(0, 3).values
    s = sd['strikes_before'].clip(0, 2).values
    if kind == 'count':
        return b * 3 + s
    if kind == 'balls':
        return b
    if kind == 'count_platoon':
        return (b * 3 + s) * 4 + (sd['pitcher_hand'].values - 1) * 2 + (sd['batter_hand'].values - 1)
    raise ValueError(kind)


def seg_rhat(sd_tr, y_tr, k, kind, mode, min_n=300):
    g = seg_keys(sd_tr, kind)
    seas = sd_tr['season'].values
    d = pd.DataFrame({'g': g, 's': seas, 'y': y_tr})
    piv = d.pivot_table(index='g', columns='s', values='y', aggfunc='mean')
    cnt = d.pivot_table(index='g', columns='s', values='y', aggfunc='size')
    last = FTS[k][-1]
    if mode == 'last':
        r = piv[last]
    elif mode == 'last_plus_meandelta':
        r = piv[last] + piv[FTS[k]].diff(axis=1).mean(axis=1)
    elif mode == 'last_plus_halfdelta':
        r = piv[last] + 0.5 * piv[FTS[k]].diff(axis=1).mean(axis=1)
    else:
        raise ValueError(mode)
    gl = float(np.average(piv[last].values, weights=cnt[last].values))
    lam = cnt[last] / (cnt[last] + min_n)
    r = lam * r + (1 - lam) * (gl + (r_hat(k, mode) - gl))
    return r


def calibrate(raw, sd_va, sd_tr, y_tr, k, scheme, mode='last', kind='count', alpha=1.0):
    """raw = era-deviation predictions (mean ~ 0)."""
    R = r_hat(k, mode)
    if scheme == 'add':                       # EXP07 winner: just add r_hat
        return np.clip(raw + R, 1e-6, 1 - 1e-6)
    if scheme == 'force_global':
        return np.clip(raw - raw.mean() + R, 1e-6, 1 - 1e-6)
    if scheme == 'force_seg':
        g = seg_keys(sd_va, kind)
        rs = seg_rhat(sd_tr, y_tr, k, kind, mode)
        rt = pd.Series(g).map(rs).fillna(R).values
        mu = pd.DataFrame({'g': g, 'p': raw}).groupby('g')['p'].transform('mean').values
        return np.clip(raw - mu + rt, 1e-6, 1 - 1e-6)
    if scheme == 'seg_relative':
        # keep the model's own global offset, correct only the *relative* segment structure
        g = seg_keys(sd_va, kind)
        rs = seg_rhat(sd_tr, y_tr, k, kind, mode)
        rt = pd.Series(g).map(rs).fillna(R).values
        mu = pd.DataFrame({'g': g, 'p': raw}).groupby('g')['p'].transform('mean').values
        corr = (rt - R) - (mu - raw.mean())
        return np.clip(raw + R + alpha * corr, 1e-6, 1 - 1e-6)
    raise ValueError(scheme)
