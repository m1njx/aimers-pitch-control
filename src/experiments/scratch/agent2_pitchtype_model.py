"""
agent2_pitchtype_model.py — AUXILIARY MODEL: predict WHICH PITCH is about to be
thrown, from pre-pitch state only, then hand the predicted pitch-type
distribution (and its expected control effect) to the main model as features.

Diagnostic that motivates this (agent2_pitchtype_ceiling.py, 1.12M train rows
matched 1:1 to trackman rows):

    control_success by tagged_pitch_type
      Curveball 0.4532 | Splitter 0.4943 | Slider 0.5008 | ChangeUp 0.5171
      Fastball  0.5380 | Sinker   0.5387 | Cutter 0.5410      (overall 0.5175)
    Var explained by pitch type alone = 7.34e-4  ->  294 skill points if known.

We cannot know the actual pitch (rule 6 forbids the current pitch's type), but
pitch selection is highly patterned, and trackman gives 1.79M labelled examples
of (pre-pitch state -> pitch type). Training on trackman and scoring the main
rows is legal: it uses only historical trackman data plus the row's own
pre-pitch state.

Fold safety: trackman is filtered to season <= as_of by the caller, and the
per-type control effect e_t is estimated from the fold's TRAIN seasons only.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from agent2_link_teams import TB_MAP
from agent2_tkm_profile import load_pitcher_map

CLASSES = ['Fastball', 'Sinker', 'Cutter', 'Slider', 'Curveball', 'ChangeUp', 'Splitter', 'Other']
FEATS = ['ptid', 'btid', 'pitcher_hand_n', 'batter_hand_n', 'balls_before',
         'strikes_before', 'outs_before', 'inning', 'top_bottom_n', 'game_month']
CATS = ['ptid', 'btid']


def load_batter_map(min_margin=0.15, min_n=20):
    m = pd.read_csv('~/LG_data/scratch/agent2_map_batter.csv')
    ok = (m.margin >= min_margin) & (m.n_a >= min_n) & m.mutual
    hand_ok = ((m.hand_a == 1) & (m.hand_b == 'Left')) | ((m.hand_a == 2) & (m.hand_b == 'Right'))
    return dict(zip(m[ok & hand_ok].a_id.astype(int), m[ok & hand_ok].b_id.astype(int)))


def _tm_frame(tm, pid_codes, bid_codes):
    X = pd.DataFrame(index=tm.index)
    X['ptid'] = tm['pitcher_trackman_id'].map(pid_codes).fillna(-1).astype(np.int32)
    X['btid'] = tm['batter_trackman_id'].map(bid_codes).fillna(-1).astype(np.int32)
    X['pitcher_hand_n'] = (tm['pitcher_hand'] == 'Left').astype(np.int8)
    X['batter_hand_n'] = (tm['batter_hand'] == 'Left').astype(np.int8)
    X['balls_before'] = tm['balls_before'].astype(np.int8)
    X['strikes_before'] = tm['strikes_before'].astype(np.int8)
    X['outs_before'] = tm['outs_before'].astype(np.int8)
    X['inning'] = tm['inning'].clip(upper=15).astype(np.int8)
    X['top_bottom_n'] = (tm['top_bottom'].map(TB_MAP).fillna(tm['top_bottom']) == 'T').astype(np.int8)
    X['game_month'] = tm['game_month'].astype(np.int8)
    return X[FEATS]


class PitchTypePredictor:
    def __init__(self, n_estimators=120, num_leaves=96, learning_rate=0.12, seed=7):
        self.params = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                           learning_rate=learning_rate, objective='multiclass',
                           num_class=len(CLASSES), random_state=seed, verbosity=-1,
                           n_jobs=-1, min_child_samples=50, colsample_bytree=0.9,
                           subsample=0.9, subsample_freq=1)
        self.pmap = load_pitcher_map()
        self.bmap = load_batter_map()

    def fit(self, tm: pd.DataFrame, df_hist_labelled: pd.DataFrame = None):
        tm = tm.copy()
        self.pid_codes = {v: i for i, v in enumerate(sorted(tm['pitcher_trackman_id'].unique()))}
        self.bid_codes = {v: i for i, v in enumerate(sorted(tm['batter_trackman_id'].unique()))}
        lab = tm['tagged_pitch_type'].where(tm['tagged_pitch_type'].isin(CLASSES[:-1]), 'Other')
        y = lab.map({c: i for i, c in enumerate(CLASSES)}).values
        X = _tm_frame(tm, self.pid_codes, self.bid_codes)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, categorical_feature=[FEATS.index(c) for c in CATS])
        self.prior_ = np.bincount(y, minlength=len(CLASSES)) / len(y)
        self.effect_ = None
        if df_hist_labelled is not None:
            self.effect_ = self._fit_effects(tm, df_hist_labelled)
        return self

    def _fit_effects(self, tm, df_hist):
        """E[control_success | pitch type] from 1:1-matched rows of the TRAIN split."""
        from agent2_pitchtype_ceiling import KEYS
        d = df_hist.copy()
        d['ptid'] = d['pitcher_id'].map(self.pmap)
        d['btid'] = d['batter_id'].map(self.bmap)
        d = d.dropna(subset=['ptid', 'btid'])
        d['ptid'] = d['ptid'].astype(np.int64); d['btid'] = d['btid'].astype(np.int64)
        t = tm.rename(columns={'pitcher_trackman_id': 'ptid', 'batter_trackman_id': 'btid'}).copy()
        t['top_bottom'] = t['top_bottom'].map(TB_MAP).fillna(t['top_bottom'])
        a = d.groupby(KEYS).size().rename('na'); b = t.groupby(KEYS).size().rename('nb')
        j = pd.concat([a, b], axis=1).dropna()
        uniq = j[(j.na == 1) & (j.nb == 1)].index
        dm = d.set_index(KEYS).loc[uniq].reset_index()
        tmm = t.set_index(KEYS).loc[uniq].reset_index()
        lab = tmm['tagged_pitch_type'].where(tmm['tagged_pitch_type'].isin(CLASSES[:-1]), 'Other')
        g = pd.DataFrame({'lab': lab.values, 'y': dm[config.TARGET_COL].values})
        mu = g.y.mean()
        eff = g.groupby('lab').y.agg(['size', 'mean'])
        e = np.array([eff['mean'].get(c, mu) - mu for c in CLASSES])
        print(f"  [effects] matched={len(g):,} mu={mu:.4f} e={np.round(e, 4).tolist()}")
        return e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X['ptid'] = df['pitcher_id'].map(self.pmap).map(self.pid_codes).fillna(-1).astype(np.int32)
        X['btid'] = df['batter_id'].map(self.bmap).map(self.bid_codes).fillna(-1).astype(np.int32)
        X['pitcher_hand_n'] = (df['pitcher_hand'] == 1).astype(np.int8)
        X['batter_hand_n'] = (df['batter_hand'] == 1).astype(np.int8)
        X['balls_before'] = df['balls_before'].fillna(0).astype(np.int8)
        X['strikes_before'] = df['strikes_before'].fillna(0).astype(np.int8)
        X['outs_before'] = df['outs_before'].fillna(0).astype(np.int8)
        X['inning'] = df['inning'].clip(upper=15).astype(np.int8)
        X['top_bottom_n'] = (df['top_bottom'] == 'T').astype(np.int8)
        X['game_month'] = df['game_month'].astype(np.int8)
        P = self.model.predict_proba(X[FEATS])
        out = pd.DataFrame(P, columns=[f'pt_{c.lower()}' for c in CLASSES], index=df.index)
        if self.effect_ is not None:
            out['pt_expected_effect'] = P @ self.effect_
        out['pt_entropy'] = -(np.clip(P, 1e-9, 1) * np.log(np.clip(P, 1e-9, 1))).sum(axis=1)
        return out.astype(np.float32)


if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_PATH)
    tm_full = pd.read_csv(config.TRACKMAN_PATH)
    for vs in [2022, 2024]:
        print(f"\n===== as_of {vs-1}, serving val {vs} =====")
        tm = tm_full[tm_full.season <= vs - 1]
        hist = df[df.season < vs]
        m = PitchTypePredictor().fit(tm, hist)
        val = df[df.season == vs]
        F = m.transform(val)
        e = F['pt_expected_effect'].values
        print(f"  Var(expected pitch-type effect) on val = {np.var(e):.6e}"
              f"  -> max recoverable skill = {1e5*np.var(e)/0.25:.0f} points")
        print(f"  (perfect knowledge would be 7.34e-04 -> 294 points)")
        print(f"  corr(expected_effect, y) = {np.corrcoef(e, val[config.TARGET_COL])[0,1]:.4f}")
        from sklearn.metrics import roc_auc_score
        print(f"  AUC(expected_effect) = {roc_auc_score(val[config.TARGET_COL], e):.4f}")
        print(F.describe().T[['mean', 'std', 'min', 'max']].to_string())
