"""Reading the raw TSVs and caching normalised records as parquet."""
import csv
import json
import os
import re
import time
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from normalize import _REPEAT, normalize_parallel, set_translit, words


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_filter=False)


def raw_paths(cfg, split):
    d = os.path.join(cfg.data_dir, split)
    return {k: os.path.join(d, f"{split}_{k}.tsv") for k in ("source1", "source2", "source3")}


def load_gt(cfg) -> pd.DataFrame:
    """Ground truth as a long table of (source1_entity_id, entity_id) pairs."""
    gt = read_tsv(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"))
    gt["entity_id"] = gt["matched_entity_ids"].str.split(",")
    long = gt[["source1_entity_id", "entity_id"]].explode("entity_id")
    long = long[long["entity_id"].notna() & (long["entity_id"] != "")]
    return long.reset_index(drop=True)


_toks = words


def learn_translit(cfg, min_count=3):
    """Learn a non-Latin-token -> English-token dictionary from the *training* ground truth.

    Some S2/S3 records spell names/states in Devanagari while the matching S1 record uses
    English. For every such true pair we count (foreign token, English token) co-occurrences
    and keep, per foreign token, the English token with the best
        P(english | foreign) * (0.5 + phonetic similarity(unidecode(foreign), english)).
    Uses only the provided training data.
    """
    t = time.time()
    p = raw_paths(cfg, "train")
    s1 = read_tsv(p["source1"]).set_index("entity_id")
    pool = pd.concat([read_tsv(p["source2"]), read_tsv(p["source3"])], ignore_index=True)
    na = pool["business_name"].str.contains(r"[^\x00-\x7f]") | pool["business_address"].str.contains(r"[^\x00-\x7f]")
    pool = pool[na].set_index("entity_id")
    gt = load_gt(cfg)
    gt = gt[gt["entity_id"].isin(pool.index)]
    a_name, a_addr = s1.loc[gt["source1_entity_id"], "business_name"].values, s1.loc[gt["source1_entity_id"], "business_address"].values
    b_name, b_addr = pool.loc[gt["entity_id"], "business_name"].values, pool.loc[gt["entity_id"], "business_address"].values

    pair_c, tok_c = Counter(), Counter()
    for an, aa, bn, ba in zip(a_name, a_addr, b_name, b_addr):
        eng = set(_toks(an))
        for d in set(x for x in _toks(bn) if not x.isascii()):
            tok_c[d] += 1
            pair_c.update((d, e) for e in eng)
        foreign_addr = set(x for x in _toks(ba) if not x.isascii())
        if foreign_addr:
            eng_a = set(x for c in aa.split(",")[-3:] for x in _toks(c))
            for d in foreign_addr:
                tok_c[d] += 1
                pair_c.update((d, e) for e in eng_a)

    best = {}
    for (d, e), c in pair_c.items():
        if tok_c[d] < min_count or e.isdigit():
            continue
        sim = fuzz.ratio(_REPEAT.sub(r"\1", unidecode(d).lower()), _REPEAT.sub(r"\1", e)) / 100
        score = c / tok_c[d] * (0.5 + sim)
        if score > best.get(d, (0, ""))[0]:
            best[d] = (score, e)
    table = {d: e for d, (s, e) in best.items() if s >= 0.5}
    with open(cfg.path("translit.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    print(f"[translit] {len(table):,} tokens learned from {len(gt):,} pairs in {time.time() - t:.0f}s", flush=True)
    return table


def load_translit(cfg):
    path = cfg.path("translit.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            set_translit(json.load(f))


def prepare(cfg, split):
    """Normalise S1 and the pooled S2+S3 records of a split and cache them."""
    t = time.time()
    load_translit(cfg)
    p = raw_paths(cfg, split)
    s1 = read_tsv(p["source1"])
    pool = pd.concat([read_tsv(p["source2"]), read_tsv(p["source3"])], ignore_index=True)
    s1n = normalize_parallel(s1, cfg.n_jobs)
    pooln = normalize_parallel(pool, cfg.n_jobs)
    pooln["src"] = pooln["entity_id"].str[1].astype("int8")          # 2 or 3
    s1n.to_parquet(cfg.path(f"{split}_s1.parquet"), index=False)
    pooln.to_parquet(cfg.path(f"{split}_pool.parquet"), index=False)
    print(f"[prepare:{split}] s1={len(s1n):,} pool={len(pooln):,} in {time.time() - t:.0f}s", flush=True)


def load_prepared(cfg, split):
    return (pd.read_parquet(cfg.path(f"{split}_s1.parquet")),
            pd.read_parquet(cfg.path(f"{split}_pool.parquet")))
