"""Shared-address post-filter for countries whose address structure differs from training.

Finding (test set): in France 13% of Source-1 entities share their exact normalised address with
another Source-1 entity and 47% of pool records sit at an address that belongs to some Source-1
entity (US/India: 4-5% and 11-21%). The matcher was trained where an exact address match is
near-conclusive, so in France it merges co-located but different businesses. This filter keeps a
matched pair at a shared address only if the names agree.

  python postfilter.py --work-dir work --in output_v3/matching_results.tsv --out output_v6/matching_results.tsv \
      --country france --min-name-sim 60 [--scored work/test_scored_v5ens.parquet --shared-thr 0.9]

Rules (applied only to the given country):
  * S1 address shared by >= 2 S1 entities, or the pool record's address matches >= 2 S1 entities
    -> keep the pair only if token_set_ratio(name_key_s1, name_key_pool) >= min_name_sim
       (and, if --scored is given, probability >= shared_thr).
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config import Config
from data import load_prepared, read_tsv
from features import record_features
from matching import write_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--split", default="test")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--country", required=True, help="comma list of countries to filter (lower-case as in the data)")
    ap.add_argument("--min-name-sim", type=float, default=60)
    ap.add_argument("--scored", default="", help="optional q,p,prob parquet; shared-address pairs also need prob >= --shared-thr")
    ap.add_argument("--shared-thr", type=float, default=0.9)
    ap.add_argument("--mode", default="either", choices=["either", "s1", "pool"], help="which sharing condition triggers the filter")
    a = ap.parse_args()
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)
    s1, pool = load_prepared(cfg, a.split)
    ra, rb = record_features(s1, pool)
    sp = pd.Series(np.arange(len(s1)), index=s1["entity_id"].values)
    pp = pd.Series(np.arange(len(pool)), index=pool["entity_id"].values)
    r = read_tsv(a.inp)
    r["entity_id"] = r["matched_entity_ids"].str.split(",")
    r = r.explode("entity_id")
    r = r[r["entity_id"].notna() & (r["entity_id"] != "")]
    d = pd.DataFrame({"q": sp[r["source1_entity_id"]].values, "p": pp[r["entity_id"]].values})
    d["country"] = s1["country"].values[d["q"].values]
    countries = set(c.strip().lower() for c in a.country.split(","))
    tgt = d["country"].isin(countries).to_numpy()
    s1_shared = ra[d["q"].values, 0] > 1
    pool_multi = rb[d["p"].values, 3] > 1
    cond = {"either": s1_shared | pool_multi, "s1": s1_shared, "pool": pool_multi}[a.mode] & tgt
    nsim = np.full(len(d), 101.0)
    idx = np.flatnonzero(cond)
    nsim[idx] = [fuzz.token_set_ratio(s1["name_key"].values[q], pool["name_key"].values[p])
                 for q, p in zip(d["q"].values[idx], d["p"].values[idx])]
    drop = cond & (nsim < a.min_name_sim)
    if a.scored:
        sc = pd.read_parquet(a.scored)
        d = d.merge(sc, on=["q", "p"], how="left")
        drop = drop | (cond & (d["prob"].fillna(0).to_numpy() < a.shared_thr))
    keep = d[~drop]
    lists = {}
    for q, p in zip(keep["q"].values, keep["p"].values):
        lists.setdefault(s1["entity_id"].values[q], set()).add(pool["entity_id"].values[p])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_lists(a.out, s1["entity_id"].values, lists, "matched_entity_ids")
    for c in sorted(countries):
        m = d["country"].to_numpy() == c
        print(f"[postfilter] {c}: {int(m.sum()):,} matched pairs, {int((cond & m).sum()):,} at shared addresses "
              f"({(cond & m).sum() / max(1, m.sum()):.1%}), dropped {int((drop & m).sum()):,} ({(drop & m).sum() / max(1, m.sum()):.1%})", flush=True)


if __name__ == "__main__":
    main()
