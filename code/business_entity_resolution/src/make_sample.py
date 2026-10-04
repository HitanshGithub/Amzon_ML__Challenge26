"""Build a small, density-preserving copy of the dataset for local development.

train: keep a fraction of S1 entities, all of their true matches, and the same fraction of
       unmatched S2/S3 records (so distractor density per entity matches the full data).
test:  keep the same fraction of every file at random.
"""
import argparse
import csv
import os

import numpy as np

from data import read_tsv


def write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="sample_dataset")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    tr = os.path.join(a.data_dir, "train")
    s1 = read_tsv(os.path.join(tr, "train_source1.tsv"))
    gt = read_tsv(os.path.join(tr, "train_ground_truth.tsv"))
    keep = s1[rng.random(len(s1)) < a.frac]
    gt_keep = gt[gt["source1_entity_id"].isin(set(keep["entity_id"]))]
    all_matched = set(x for l in gt["matched_entity_ids"].str.split(",") for x in l if x)
    kept_matched = set(x for l in gt_keep["matched_entity_ids"].str.split(",") for x in l if x)
    write(keep, os.path.join(a.out_dir, "train", "train_source1.tsv"))
    write(gt_keep, os.path.join(a.out_dir, "train", "train_ground_truth.tsv"))
    for k in ("source2", "source3"):
        df = read_tsv(os.path.join(tr, f"train_{k}.tsv"))
        unmatched = ~df["entity_id"].isin(all_matched)
        sel = df["entity_id"].isin(kept_matched) | (unmatched & (rng.random(len(df)) < a.frac))
        write(df[sel], os.path.join(a.out_dir, "train", f"train_{k}.tsv"))

    te = os.path.join(a.data_dir, "test")
    for k in ("source1", "source2", "source3"):
        df = read_tsv(os.path.join(te, f"test_{k}.tsv"))
        write(df[rng.random(len(df)) < a.frac], os.path.join(a.out_dir, "test", f"test_{k}.tsv"))
    print("sample written to", a.out_dir)


if __name__ == "__main__":
    main()
