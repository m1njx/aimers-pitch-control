"""
agent2_link_teams.py — Entity resolution step 1: map train `pitcher_team_id`
(int) to trackman `pitcher_team` (string) using situation-block co-occurrence.

Idea: both tables log the SAME underlying pitches (trackman is a superset).
A block key K = (season, month, dayofweek, inning, top_bottom, balls, strikes, outs)
is shared by both tables. Two entities that are the same real-world entity will
have near-identical count profiles over K. Cosine similarity over the sparse
count matrix recovers the mapping.
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
from scipy import sparse
import config

TB_MAP = {'Top': 'T', 'Bottom': 'B'}


def block_key(df, tb_col='top_bottom'):
    return (df['season'].astype(np.int64) * 10 ** 10
            + df['game_month'].astype(np.int64) * 10 ** 9
            + df['game_dayofweek'].astype(np.int64) * 10 ** 8
            + df['inning'].clip(upper=15).astype(np.int64) * 10 ** 6
            + (df[tb_col] == 'T').astype(np.int64) * 10 ** 5
            + df['balls_before'].fillna(0).astype(np.int64) * 10 ** 3
            + df['strikes_before'].fillna(0).astype(np.int64) * 10 ** 2
            + df['outs_before'].fillna(0).astype(np.int64) * 10)


def cosine_map(a_keys, a_ent, b_keys, b_ent):
    """Return (ents_a, ents_b, similarity matrix)."""
    all_keys = pd.Index(np.union1d(np.unique(a_keys), np.unique(b_keys)))
    ea = pd.Index(np.unique(a_ent)); eb = pd.Index(np.unique(b_ent))
    A = sparse.coo_matrix((np.ones(len(a_keys)),
                           (ea.get_indexer(a_ent), all_keys.get_indexer(a_keys))),
                          shape=(len(ea), len(all_keys))).tocsr()
    B = sparse.coo_matrix((np.ones(len(b_keys)),
                           (eb.get_indexer(b_ent), all_keys.get_indexer(b_keys))),
                          shape=(len(eb), len(all_keys))).tocsr()
    na = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()
    nb = np.sqrt(B.multiply(B).sum(axis=1)).A.ravel()
    S = (A @ B.T).toarray() / (na[:, None] * nb[None, :] + 1e-12)
    return ea, eb, S


if __name__ == '__main__':
    print("loading...")
    df = pd.read_csv(config.TRAIN_PATH, usecols=['season', 'game_month', 'game_dayofweek', 'inning',
                                                 'top_bottom', 'balls_before', 'strikes_before',
                                                 'outs_before', 'pitcher_team_id', 'batter_team_id',
                                                 'pitcher_id', 'pitcher_hand'])
    tm = pd.read_csv(config.TRACKMAN_PATH, usecols=['season', 'game_month', 'game_dayofweek', 'inning',
                                                    'top_bottom', 'balls_before', 'strikes_before',
                                                    'outs_before', 'pitcher_team', 'batter_team',
                                                    'pitcher_trackman_id', 'pitcher_hand'])
    tm['top_bottom'] = tm['top_bottom'].map(TB_MAP)
    ka = block_key(df); kb = block_key(tm)
    print(f"train blocks={ka.nunique():,}  tkm blocks={kb.nunique():,}")

    ea, eb, S = cosine_map(ka.values, df['pitcher_team_id'].values,
                           kb.values, tm['pitcher_team'].values)
    print("\n=== best trackman team for each train pitcher_team_id ===")
    for i, t in enumerate(ea):
        order = np.argsort(-S[i])
        top = [(eb[j], round(S[i, j], 4)) for j in order[:3]]
        print(f"  team_id {t:>3}: {top}")

    print("\n=== reverse: best train id for each trackman team ===")
    for j, t in enumerate(eb):
        order = np.argsort(-S[:, j])
        top = [(int(ea[i]), round(S[i, j], 4)) for i in order[:2]]
        print(f"  {t:<9}: {top}")

    np.save('~/LG_data/scratch/agent2_team_S.npy', S)
    pd.Series(ea).to_csv('~/LG_data/scratch/agent2_team_ea.csv', index=False)
    pd.Series(eb).to_csv('~/LG_data/scratch/agent2_team_eb.csv', index=False)
