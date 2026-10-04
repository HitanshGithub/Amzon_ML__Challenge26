"""Synthetic labelled world for a country without training labels (France), used to train the final stage.

S1 = the real test S1 records of the country. Their true copies are generated with synth_fr's noise
(noisy_name / noisy_addr), with the (n_S2, n_S3) copy counts drawn from the training ground truth.
Decoys = real test pool records of the country that the current model is confident match nothing
(max final probability < --decoy-max-prob): test decoys are unrelated businesses, not copies of S1
entities, so the world keeps the country's real crowding, names and decoys. A --drop-s1-frac share
of S1 entities is removed after generating their copies (look-alike decoys, as in the training-side
distractor simulation).

Writes <out>/test/test_source{1,2,3}.tsv (the world, laid out like a test split) and <out>/syn_gt.tsv.
"""
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd

from synth_fr import noisy_addr, noisy_name


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work", help="prepared test pool the scored file indexes")
    ap.add_argument("--scored", default="work6/test_scored_v6ens_src.parquet")
    ap.add_argument("--out", default="dataset_syn")
    ap.add_argument("--country", default="France")
    ap.add_argument("--decoy-max-prob", type=float, default=0.02)
    ap.add_argument("--decoy-share", type=float, default=0.4, help="target share of decoys in the pool (<=0: all)")
    ap.add_argument("--drop-s1-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    r, rng = random.Random(a.seed), np.random.default_rng(a.seed)
    td = os.path.join(a.data_dir, "test")

    s1 = read(os.path.join(td, "test_source1.tsv"))
    s1 = s1[s1["country"] == a.country].reset_index(drop=True)
    gt = read(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
    ids = gt["matched_entity_ids"].str.split(",")
    counts = np.stack([ids.map(lambda l: sum(x.startswith("S2-") for x in l)).to_numpy(),
                       ids.map(lambda l: sum(x.startswith("S3-") for x in l)).to_numpy()], 1)
    pick = counts[rng.integers(0, len(counts), len(s1))]

    copies, gt_rows, nid = {2: [], 3: []}, [], 1_000_000_000
    for i in range(len(s1)):
        name, addr = s1["business_name"].iat[i], s1["business_address"].iat[i]
        matched = []
        for src, n in ((2, pick[i, 0]), (3, pick[i, 1])):
            for _ in range(n):
                nm, heavy = noisy_name(name, r, src)
                pid = f"S{src}-{nid}"
                nid += 1
                copies[src].append((pid, nm, noisy_addr(addr, r, src, allow_empty=not heavy), a.country))
                matched.append(pid)
        gt_rows.append((f"S1-{1_000_000_000 + i}", ",".join(matched)))
    keep = rng.random(len(s1)) >= a.drop_s1_frac

    # decoys: the country's real pool records whose best final probability is tiny
    raw = pd.concat([read(os.path.join(td, "test_source2.tsv")), read(os.path.join(td, "test_source3.tsv"))], ignore_index=True)
    prep = pd.read_parquet(os.path.join(a.work_dir, "test_pool.parquet"), columns=["entity_id"])
    assert len(prep) == len(raw) and (prep["entity_id"].values == raw["entity_id"].values).all(), "pool order mismatch"
    sc = pd.read_parquet(a.scored, columns=["p", "prob"])
    mx = sc.groupby("p")["prob"].max()
    maxp = np.zeros(len(raw), dtype=np.float32)
    maxp[mx.index.values] = mx.values
    is_c = (raw["country"] == a.country).to_numpy()
    dec = raw[is_c & (maxp < a.decoy_max_prob)]
    n_true = len(copies[2]) + len(copies[3])
    if a.decoy_share > 0:
        n_dec = min(len(dec), int(n_true * a.decoy_share / (1 - a.decoy_share)))
        dec = dec.sample(n=n_dec, random_state=a.seed)

    out = os.path.join(a.out, "test")
    os.makedirs(out, exist_ok=True)
    cols = ["entity_id", "business_name", "business_address", "country"]
    def w(df, f):                       # plain join: the readers use QUOTE_NONE without an escape char
        with open(os.path.join(out, f), "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(cols) + "\n")
            fh.writelines("\t".join(row) + "\n" for row in df[cols].itertuples(index=False, name=None))
    s1o = s1.assign(entity_id=[f"S1-{1_000_000_000 + i}" for i in range(len(s1))])[keep]
    w(s1o, "test_source1.tsv")
    for src in (2, 3):
        cp = pd.DataFrame(copies[src], columns=cols)
        d = dec[dec["entity_id"].str.startswith(f"S{src}-")]
        w(pd.concat([cp, d[cols]], ignore_index=True).sample(frac=1.0, random_state=a.seed + src), f"test_source{src}.tsv")
    out, cols = a.out, ["source1_entity_id", "matched_entity_ids"]
    w(pd.DataFrame(gt_rows, columns=cols)[keep], "syn_gt.tsv")

    real_ratio = is_c.sum() / len(s1)
    pool_n = n_true + len(dec)
    empty = np.mean([c[2] == "" for src in (2, 3) for c in copies[src]])
    print(f"[syn] {a.country}: S1 {int(keep.sum()):,} kept of {len(s1):,}; true copies {n_true:,} "
          f"({n_true / len(s1):.2f}/S1, empty address {empty:.3f}); decoys {len(dec):,} "
          f"(candidates {int((is_c & (maxp < a.decoy_max_prob)).sum()):,}); pool {pool_n:,} = {pool_n / keep.sum():.2f}/S1 "
          f"(real {a.country} test pool {real_ratio:.2f}/S1)", flush=True)


if __name__ == "__main__":
    main()
