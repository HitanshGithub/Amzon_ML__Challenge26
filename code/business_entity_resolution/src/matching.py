"""Turning pair probabilities into per-entity match lists, and writing submission files."""
import csv

import numpy as np
import pandas as pd

from evaluate import macro_f05


def assign(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Each S2/S3 record belongs to at most one S1 entity (true in 100% of training pairs):
    keep only its highest-probability S1, then apply the probability threshold."""
    best = pairs.sort_values("prob", ascending=False).drop_duplicates("p")
    return best[best["prob"] >= threshold]


def to_lists(sel: pd.DataFrame, s1_ids, pool_ids) -> dict:
    d = {}
    for q, p in zip(sel["q"].values, sel["p"].values):
        d.setdefault(s1_ids[q], set()).add(pool_ids[p])
    return d


def tune_threshold(pairs, s1_ids, pool_ids, true: dict, eval_ids, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.96, 0.025), 3)
    scores = []
    for t in grid:
        pred = to_lists(assign(pairs, t), s1_ids, pool_ids)
        scores.append(macro_f05(pred, true, eval_ids))
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i]), list(zip(grid.tolist(), scores))


def write_lists(path, s1_ids_all, lists: dict, col: str):
    """One row per S1 entity (empty list allowed), tab separated, comma-joined IDs, no quoting."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        w.writerow(["source1_entity_id", col])
        for sid in s1_ids_all:
            w.writerow([sid, ",".join(sorted(lists.get(sid, ())))])
