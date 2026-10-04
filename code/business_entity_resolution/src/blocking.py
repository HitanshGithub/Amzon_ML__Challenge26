"""Candidate generation.

Matches never cross countries in the training data, so blocking is done inside each
country value. Country is treated as an open set of strings: whatever labels appear in
the test files (including France, unseen in training) get their own block.

Within a block every record is embedded as ONE sparse vector
    [ sqrt(w) * char-3gram TF-IDF(name) , sqrt(1-w) * word TF-IDF(address) ]
so a dot product equals  w * name_cos + (1-w) * addr_cos  and top-k is taken over the whole
pool on the combined score. Names in this data are highly repetitive, so name-only top-k is
crowded by look-alikes; on full-size pools the joint search raised the recall ceiling from
92.5%/96.0% (name top-30 + address top-10) to ~96.7%/97.7% (India/US) with fewer candidates.
A small name-only top-k is added for records whose address is empty. Chosen on 3k-5k
queries per country against the full training pools: w=0.4, top-40 joint + top-10 name,
char 3-grams in >5% of names dropped (same recall, ~20% faster): 97.8% (India) / 98.4% (US).
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _topn(Q, P_T, k, threshold, chunk, n_threads):
    rows, cols = [], []
    for s in range(0, Q.shape[0], chunk):
        C = sp_matmul_topn(Q[s:s + chunk], P_T, top_n=k, threshold=threshold, n_threads=n_threads).tocoo()
        rows.append(C.row.astype(np.int64) + s)
        cols.append(C.col.astype(np.int64))
    return np.concatenate(rows), np.concatenate(cols)


def _rowdot(A, B, qi, pi, step=2_000_000):
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), step):
        out[s:s + step] = np.asarray(A[qi[s:s + step]].multiply(B[pi[s:s + step]]).sum(axis=1)).ravel()
    return out


def block(s1: pd.DataFrame, pool: pd.DataFrame, cfg, countries=None) -> pd.DataFrame:
    n_threads = cfg.n_jobs if cfg.n_jobs > 1 else None
    w = cfg.joint_w
    frames = []
    for country, q_idx in s1.groupby("country").indices.items():
        if countries is not None and country not in countries:
            continue
        t = time.time()
        p_idx = np.flatnonzero(pool["country"].values == country)
        if len(p_idx) == 0:
            print(f"[block] {country!r}: no pool records", flush=True)
            continue
        qn, pn = s1["name_key"].values[q_idx], pool["name_key"].values[p_idx]
        qa, pa = s1["addr_norm"].values[q_idx], pool["addr_norm"].values[p_idx]

        vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=cfg.name_max_df,
                             sublinear_tf=True, dtype=np.float32)
        vn.fit(np.concatenate([qn, pn]))
        Qn, Pn = vn.transform(qn), vn.transform(pn)
        va = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, max_df=0.3,
                             sublinear_tf=True, dtype=np.float32)
        va.fit(np.concatenate([qa, pa]))
        Qa, Pa = va.transform(qa), va.transform(pa)

        Jq = sparse.hstack([Qn * np.sqrt(w), Qa * np.sqrt(1 - w)]).tocsr()
        JpT = sparse.hstack([Pn * np.sqrt(w), Pa * np.sqrt(1 - w)]).tocsr().T.tocsr()
        r, c = _topn(Jq, JpT, cfg.k_joint, None, cfg.query_chunk, n_threads)
        del Jq, JpT
        if cfg.k_name_extra > 0:
            r2, c2 = _topn(Qn, Pn.T.tocsr(), cfg.k_name_extra, cfg.name_min_sim, cfg.query_chunk, n_threads)
            r, c = np.concatenate([r, r2]), np.concatenate([c, c2])
        pairs = np.unique(r * len(p_idx) + c)
        qi, pi = pairs // len(p_idx), pairs % len(p_idx)

        name_cos = _rowdot(Qn, Pn, qi, pi)
        addr_cos = _rowdot(Qa, Pa, qi, pi)
        df = pd.DataFrame({"q": q_idx[qi], "p": p_idx[pi], "name_cos": name_cos, "addr_cos": addr_cos})
        df["comb"] = (w * df["name_cos"] + (1 - w) * df["addr_cos"]).astype(np.float32)
        frames.append(df)
        print(f"[block] {country!r}: q={len(q_idx):,} pool={len(p_idx):,} pairs={len(df):,} "
              f"({len(df) / len(q_idx):.1f}/q) in {time.time() - t:.0f}s", flush=True)
    cands = pd.concat(frames, ignore_index=True)
    cands["q"] = cands["q"].astype(np.int32)
    cands["p"] = cands["p"].astype(np.int32)
    return cands


def rescue_empty_address(s1: pd.DataFrame, pool: pd.DataFrame, cfg) -> pd.DataFrame:
    """Name-only top-k restricted to pool records with an EMPTY address.

    On the full training data 77% of the pairs the joint search missed had an empty address on
    the S2/S3 side: without address tokens their joint score is low and generic names get crowded
    out. Searching names only among empty-address records (3-4% of the pool) is cheap and lifted
    the recall ceiling from 98.1% to ~99.0% (India 97.8 -> 98.8, US 98.3 -> 99.3, top-20)."""
    n_threads = cfg.n_jobs if cfg.n_jobs > 1 else None
    frames = []
    for country, q_idx in s1.groupby("country").indices.items():
        t = time.time()
        p_idx = np.flatnonzero((pool["country"].values == country) & (pool["addr_empty"].values == 1))
        if len(p_idx) == 0:
            continue
        all_p = np.flatnonzero(pool["country"].values == country)
        vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        vn.fit(np.concatenate([s1["name_key"].values[q_idx], pool["name_key"].values[all_p]]))
        Qn = vn.transform(s1["name_key"].values[q_idx])
        Pn = vn.transform(pool["name_key"].values[p_idx])
        r, c = _topn(Qn, Pn.T.tocsr(), cfg.rescue_k, cfg.rescue_min_sim, cfg.query_chunk, n_threads)
        frames.append(pd.DataFrame({"q": q_idx[r], "p": p_idx[c], "name_cos": _rowdot(Qn, Pn, r, c),
                                    "addr_cos": np.zeros(len(r), np.float32)}))
        print(f"[rescue] {country!r}: q={len(q_idx):,} empty-address pool={len(p_idx):,} pairs={len(r):,} "
              f"in {time.time() - t:.0f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["comb"] = (cfg.joint_w * out["name_cos"]).astype(np.float32)
    return out


def blocking_recall(cands, s1, pool, gt_long):
    """Share of true pairs present in the candidate set (the recall ceiling)."""
    s1_pos = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    p_pos = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    g = gt_long[gt_long["source1_entity_id"].isin(s1_pos.index) & gt_long["entity_id"].isin(p_pos.index)]
    true_keys = s1_pos[g["source1_entity_id"]].values.astype(np.int64) * len(pool) + p_pos[g["entity_id"]].values
    cand_keys = cands["q"].values.astype(np.int64) * len(pool) + cands["p"].values
    return np.isin(true_keys, cand_keys).mean(), len(true_keys)
