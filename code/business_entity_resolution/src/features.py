"""Pair features for (Source 1 record, candidate) pairs.

Two groups:
  * context features from blocking scores - cheap, computed on the full candidate table so
    that "how does this pair rank among competitors" is identical in train and test;
  * string features (rapidfuzz) - computed only on the rows we actually score.
Country is deliberately NOT a feature so the model transfers to countries unseen in training.
"""
import os
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

S1_COLS = ["name_norm", "name_core", "name_key", "name_sq", "legal", "addr_norm", "nums",
           "addr_empty", "is_domain", "non_ascii"]


def context_features(c: pd.DataFrame) -> pd.DataFrame:
    """Rank / gap features w.r.t. competing candidates of the same S1 row (q) and pool row (p)."""
    gq, gp = c.groupby("q"), c.groupby("p")
    c["q_n"] = gq["p"].transform("size").astype(np.int16)
    c["p_n"] = gp["q"].transform("size").astype(np.int16)
    for col in ("name_cos", "addr_cos", "comb"):
        c[f"q_rank_{col}"] = gq[col].rank(ascending=False, method="min").astype(np.float32)
        c[f"q_gap_{col}"] = (gq[col].transform("max") - c[col]).astype(np.float32)
        c[f"p_rank_{col}"] = gp[col].rank(ascending=False, method="min").astype(np.float32)
        c[f"p_gap_{col}"] = (gp[col].transform("max") - c[col]).astype(np.float32)
    # second best competitor for the pool record: is this pair a clear winner?
    cs = c[["p", "comb"]].sort_values(["p", "comb"], ascending=[True, False])
    second = cs[cs.groupby("p").cumcount() == 1].set_index("p")["comb"]
    c["p_second_comb"] = c["p"].map(second).fillna(0).astype(np.float32)
    return c


def _nums(s):
    return s.split() if s else []


def _num_feats(na, nb):
    a, b = _nums(na), _nums(nb)
    if not a or not b:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    sa, sb = set(a), set(b)
    inter = sa & sb
    jac = len(inter) / len(sa | sb)
    first_eq = float(a[0] == b[0])
    # truncated / padded house numbers: 2454 vs 245, 1014 vs 014
    fix = 0.0
    for x in sa:
        for y in sb:
            if x != y and min(len(x), len(y)) >= 2 and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)):
                fix = 1.0
    la = {x for x in sa if len(x) >= 5}
    lb = {x for x in sb if len(x) >= 5}
    post = np.nan if not la or not lb else float(bool(la & lb))
    return jac, first_eq, fix, post, float(len(inter))


def _house_diff(na, nb):
    """Smallest absolute difference between short numbers (house / unit numbers) of the two addresses."""
    a = [int(x) for x in _nums(na) if len(x) <= 6]
    b = [int(x) for x in _nums(nb) if len(x) <= 6]
    if not a or not b:
        return np.nan
    return float(np.log1p(min(abs(x - y) for x in a for y in b)))


def _tok_match(t, others):
    """t matches if equal, an abbreviation (prefix: 'pr'~'private', 'gr'~'group'), or a close typo."""
    for o in others:
        if t == o or (len(t) < len(o) and o.startswith(t)) or (len(o) < len(t) and t.startswith(o)):
            return True
        if len(t) >= 4 and len(o) >= 4 and fuzz.ratio(t, o) >= 80:
            return True
    return False


def _abbr_feats(ka, kb):
    ta, tb = ka.split(), kb.split()
    if not ta or not tb:
        return np.nan, np.nan, np.nan, np.nan
    mb = sum(_tok_match(t, ta) for t in tb)
    ma = sum(_tok_match(t, tb) for t in ta)
    return mb / len(tb), ma / len(ta), float(ma == len(ta)), float(len(tb) - mb)


def _jacc(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def string_features_chunk(args):
    A, B = args
    n = len(A["name_norm"])
    out = np.full((n, 27), np.nan, dtype=np.float32)
    for i in range(n):
        nn_a, nn_b = A["name_norm"][i], B["name_norm"][i]
        co_a, co_b = A["name_core"][i], B["name_core"][i]
        ky_a, ky_b = A["name_key"][i], B["name_key"][i]
        sq_a, sq_b = A["name_sq"][i], B["name_sq"][i]
        lg_a, lg_b = A["legal"][i], B["legal"][i]
        ad_a, ad_b = A["addr_norm"][i], B["addr_norm"][i]
        r = out[i]
        r[0] = fuzz.ratio(nn_a, nn_b)
        r[1] = fuzz.token_sort_ratio(nn_a, nn_b)
        r[2] = fuzz.token_set_ratio(co_a, co_b)
        r[3] = fuzz.partial_ratio(ky_a, ky_b)
        r[4] = fuzz.ratio(co_a, co_b)
        r[5] = JaroWinkler.normalized_similarity(ky_a, ky_b)
        r[6] = fuzz.ratio(sq_a, sq_b)
        r[7] = fuzz.partial_ratio(sq_a, sq_b) if sq_a and sq_b else np.nan
        r[8] = float(ky_a.split(" ", 1)[0] == ky_b.split(" ", 1)[0])
        r[9] = _jacc(ky_a, ky_b)
        r[10] = float(lg_a == lg_b)
        r[11] = _jacc(lg_a, lg_b) if lg_a and lg_b else np.nan
        r[12] = len(ky_b) - len(ky_a)
        if ad_a and ad_b:
            r[13] = fuzz.ratio(ad_a, ad_b)
            r[14] = fuzz.token_set_ratio(ad_a, ad_b)
            r[15] = fuzz.token_sort_ratio(ad_a, ad_b)
            r[16] = _jacc(ad_a, ad_b)
        r[17:22] = _num_feats(A["nums"][i], B["nums"][i])
        r[22:26] = _abbr_feats(ky_a, ky_b)
        r[26] = _house_diff(A["nums"][i], B["nums"][i])
    return out


# ---------------------------------------------------------------- record-level features (v5)
# Training-set facts: 2% of true matches carry a synthetic replacement name whose words never
# occur in any Source-1 name (88% of them fully out-of-vocabulary vs 1.6% of decoys); decoys
# practically never share an exact normalised address with a Source-1 entity (<0.05% vs 27% of
# true copies) and practically never have an empty address (0.3% vs 4.4%). Per-record vocabulary
# and uniqueness counts let the model weigh name evidence against address evidence. Everything is
# computed from the same split's records only (no labels).
REC_COLS = ["b_oov_frac", "b_oov_all", "b_n_tok", "b_addr_s1_cnt", "b_name_s1_cnt", "b_addr_pool_cnt",
            "b_name_pool_cnt", "a_addr_s1_cnt", "a_name_s1_cnt", "a_addr_pool_cnt", "a_name_pool_cnt"]


def record_features(s1: pd.DataFrame, pool: pd.DataFrame):
    vocab = {}
    for c, g in s1.groupby("country")["name_key"]:
        vocab[c] = set(t for n in g.values for t in n.split())
    oov = np.full(len(pool), np.nan, np.float32)
    ntok = np.zeros(len(pool), np.float32)
    for i, (n, c) in enumerate(zip(pool["name_key"].values, pool["country"].values)):
        toks = n.split()
        ntok[i] = len(toks)
        if toks:
            v = vocab.get(c, set())
            oov[i] = sum(t not in v for t in toks) / len(toks)

    def counts(keys_ref, keys_query):
        vc = pd.Series(keys_ref).value_counts()
        return pd.Series(keys_query).map(vc).fillna(0).to_numpy(np.float32)

    ak = s1["country"].values.astype(object) + "|" + s1["addr_norm"].values.astype(object)
    nk = s1["country"].values.astype(object) + "|" + s1["name_key"].values.astype(object)
    pak = pool["country"].values.astype(object) + "|" + pool["addr_norm"].values.astype(object)
    pnk = pool["country"].values.astype(object) + "|" + pool["name_key"].values.astype(object)
    empty_a = s1["addr_norm"].values == ""
    empty_p = pool["addr_norm"].values == ""
    b = np.column_stack([oov, (oov >= 1).astype(np.float32), ntok,
                         np.where(empty_p, np.nan, counts(ak, pak)), counts(nk, pnk),
                         np.where(empty_p, np.nan, counts(pak, pak)), counts(pnk, pnk)])
    a = np.column_stack([np.where(empty_a, np.nan, counts(ak, ak)), counts(nk, nk),
                         np.where(empty_a, np.nan, counts(pak, ak)), counts(pnk, nk)])
    return a.astype(np.float32), b.astype(np.float32)


# ER_REC_FEATS=0 disables the record-level features (v3/v4 feature set) - see stage4/v6 notes:
# they shifted owner assignment among co-located French entities and lowered the board score.
USE_REC = os.environ.get("ER_REC_FEATS", "1") == "1"
_REC_CACHE = {}


def record_features_cached(s1, pool):
    key = (id(s1), id(pool), len(s1), len(pool))
    if key not in _REC_CACHE:
        _REC_CACHE.clear()
        _REC_CACHE[key] = record_features(s1, pool)
    return _REC_CACHE[key]


STRING_COLS = ["n_ratio", "n_tsort", "n_tset_core", "n_partial_key", "n_ratio_core", "n_jw_key",
               "n_ratio_sq", "n_partial_sq", "n_first_eq", "n_jacc_key", "legal_eq", "legal_jacc",
               "n_len_diff", "a_ratio", "a_tset", "a_tsort", "a_jacc",
               "num_jacc", "num_first_eq", "num_fix", "post_eq", "num_common",
               "n_abbr_b", "n_abbr_a", "n_a_in_b", "n_extra_b", "num_house_diff"]


def string_features(c: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame, n_jobs: int, chunk=50_000) -> pd.DataFrame:
    t = time.time()
    q, p = c["q"].values, c["p"].values
    text = ["name_norm", "name_core", "name_key", "name_sq", "legal", "addr_norm", "nums"]
    A = {k: s1[k].values[q] for k in text}
    B = {k: pool[k].values[p] for k in text}
    jobs = [({k: v[s:s + chunk].tolist() for k, v in A.items()},
             {k: v[s:s + chunk].tolist() for k, v in B.items()}) for s in range(0, len(c), chunk)]
    if n_jobs > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as pl:
            res = pl.map(string_features_chunk, jobs, chunksize=1)
    else:
        res = [string_features_chunk(j) for j in jobs]
    F = pd.DataFrame(np.vstack(res) if res else np.zeros((0, len(STRING_COLS)), np.float32),
                     columns=STRING_COLS, index=c.index)
    F["a_empty_a"] = s1["addr_empty"].values[q]
    F["a_empty_b"] = pool["addr_empty"].values[p]
    F["b_domain"] = pool["is_domain"].values[p]
    F["b_non_ascii"] = pool["non_ascii"].values[p]
    F["b_src"] = pool["src"].values[p]
    F["b_legal_empty"] = (pool["legal"].values[p] == "").astype(np.int8)
    F["b_alias"] = pool["has_alias"].values[p]
    if USE_REC:
        ra, rb = record_features_cached(s1, pool)
        for i, col in enumerate(REC_COLS[:7]):
            F[col] = rb[p, i]
        for i, col in enumerate(REC_COLS[7:]):
            F[col] = ra[q, i]
    print(f"[features] {len(c):,} pairs in {time.time() - t:.0f}s", flush=True)
    return F


CONTEXT_COLS = ["name_cos", "addr_cos", "comb", "q_n", "p_n", "p_second_comb"] + [
    f"{s}_{k}_{col}" for col in ("name_cos", "addr_cos", "comb") for s in ("q", "p") for k in ("rank", "gap")]
EXTRA_COLS = ["a_empty_a", "a_empty_b", "b_domain", "b_non_ascii", "b_src", "b_legal_empty", "b_alias"] + (REC_COLS if USE_REC else [])
FEATURES = CONTEXT_COLS + STRING_COLS + EXTRA_COLS


def build(c, s1, pool, n_jobs):
    F = string_features(c, s1, pool, n_jobs)
    return pd.concat([c[["q", "p"] + CONTEXT_COLS].reset_index(drop=True), F.reset_index(drop=True)], axis=1)
