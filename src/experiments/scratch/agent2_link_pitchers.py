"""
agent2_link_pitchers.py — Entity resolution step 2: map train `pitcher_id` ->
trackman `pitcher_trackman_id` via situation-block count-profile cosine similarity.

Also does batter_id -> batter_trackman_id.
Outputs a mapping csv with confidence diagnostics (top1 sim, margin over top2,
hand agreement).
"""
import sys
sys.path.insert(0, os.path.expanduser('~/LG_data'))
import numpy as np
import pandas as pd
from scipy import sparse
import config
from agent2_link_teams import block_key, TB_MAP


def build_matrix(keys, ents, all_keys, ent_index):
    return sparse.coo_matrix((np.ones(len(keys)),
                              (ent_index.get_indexer(ents), all_keys.get_indexer(keys))),
                             shape=(len(ent_index), len(all_keys))).tocsr()


def resolve(df, tm, a_ent_col, b_ent_col, a_hand_col, b_hand_col, label, min_n=30):
    ka = block_key(df).values
    kb = block_key(tm).values
    all_keys = pd.Index(np.union1d(np.unique(ka), np.unique(kb)))
    ea = pd.Index(np.unique(df[a_ent_col].values))
    eb = pd.Index(np.unique(tm[b_ent_col].values))
    A = build_matrix(ka, df[a_ent_col].values, all_keys, ea)
    B = build_matrix(kb, tm[b_ent_col].values, all_keys, eb)
    na = np.sqrt(A.multiply(A).sum(axis=1)).A.ravel()
    nb = np.sqrt(B.multiply(B).sum(axis=1)).A.ravel()
    print(f"[{label}] A={A.shape} B={B.shape}, computing similarity...")
    S = (A @ B.T).toarray() / (na[:, None] * nb[None, :] + 1e-12)

    cnt_a = np.asarray(A.sum(axis=1)).ravel()
    cnt_b = np.asarray(B.sum(axis=1)).ravel()

    # hand profiles (majority hand per entity)
    ha = df.groupby(a_ent_col)[a_hand_col].agg(lambda s: s.value_counts().index[0]).reindex(ea)
    hb = tm.groupby(b_ent_col)[b_hand_col].agg(lambda s: s.value_counts().index[0]).reindex(eb)

    order = np.argsort(-S, axis=1)
    top1 = order[:, 0]; top2 = order[:, 1]
    rows = []
    for i in range(len(ea)):
        rows.append(dict(
            a_id=ea[i], n_a=int(cnt_a[i]),
            b_id=eb[top1[i]], n_b=int(cnt_b[top1[i]]),
            sim1=S[i, top1[i]], sim2=S[i, top2[i]],
            margin=S[i, top1[i]] - S[i, top2[i]],
            hand_a=ha.iloc[i], hand_b=hb.iloc[top1[i]],
        ))
    res = pd.DataFrame(rows)
    # mutual-best check
    rev_best = np.argmax(S, axis=0)
    res['mutual'] = [rev_best[top1[i]] == i for i in range(len(ea))]
    print(f"[{label}] n={len(res)}  mutual-best={res.mutual.mean():.3f}  "
          f"sim1 median={res.sim1.median():.4f}  margin median={res.margin.median():.4f}")
    big = res[res.n_a >= min_n]
    print(f"[{label}] n_a>={min_n}: {len(big)} entities, mutual={big.mutual.mean():.3f}, "
          f"sim1>0.8: {(big.sim1>0.8).mean():.3f}, sim1>0.9: {(big.sim1>0.9).mean():.3f}")
    print(f"[{label}] hand agreement table:")
    print(pd.crosstab(big.hand_a, big.hand_b))
    return res, S, ea, eb


if __name__ == '__main__':
    cols = ['season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
            'balls_before', 'strikes_before', 'outs_before']
    df = pd.read_csv(config.TRAIN_PATH, usecols=cols + ['pitcher_id', 'batter_id',
                                                        'pitcher_hand', 'batter_hand'])
    tm = pd.read_csv(config.TRACKMAN_PATH, usecols=cols + ['pitcher_trackman_id', 'batter_trackman_id',
                                                           'pitcher_hand', 'batter_hand'])
    tm['top_bottom'] = tm['top_bottom'].map(TB_MAP)

    resP, SP, eaP, ebP = resolve(df, tm, 'pitcher_id', 'pitcher_trackman_id',
                                 'pitcher_hand', 'pitcher_hand', 'PITCHER')
    resP.to_csv('~/LG_data/scratch/agent2_map_pitcher.csv', index=False)
    print(resP.sort_values('n_a', ascending=False).head(25).to_string())

    resB, SB, eaB, ebB = resolve(df, tm, 'batter_id', 'batter_trackman_id',
                                 'batter_hand', 'batter_hand', 'BATTER')
    resB.to_csv('~/LG_data/scratch/agent2_map_batter.csv', index=False)
