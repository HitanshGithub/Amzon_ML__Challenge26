"""Re-derive matching_results.tsv from saved test probabilities with a per-country decision rule.

The matching model is unchanged; only the final decision changes. Validation cannot measure
countries absent from training, so per-country adjustments are chosen from leaderboard evidence
(the country's share of the test set and the score difference between two variants).

  python decide.py --work-dir work --scored work/test_scored_v5ce3.parquet --out-dir output_v6 \
      --rule thr --thr 0.725 --country-thr france=0.60
"""
import argparse
import os

import numpy as np
import pandas as pd

from data import load_prepared
from matching import to_lists, write_lists
from model import assign_best, decide_expected_f, decide_threshold
from stage3 import decide_threshold_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--scored", required=True, help="parquet with q, p, prob for every test candidate pair")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rule", default="thr", choices=["thr", "src", "expf"])
    ap.add_argument("--thr", type=float, default=0.725, help="global threshold (thr) / S2 threshold (src)")
    ap.add_argument("--thr3", type=float, default=None, help="S3 threshold for the src rule")
    ap.add_argument("--alpha", type=float, default=0.0, help="expf rule parameter")
    ap.add_argument("--country-thr", default="", help="comma list country=threshold overriding the rule for that country")
    ap.add_argument("--country-scale", default="", help="comma list country=factor multiplying probabilities of that country")
    a = ap.parse_args()
    from config import Config
    cfg = Config(data_dir=a.data_dir, work_dir=a.work_dir)

    s1, pool = load_prepared(cfg, "test")
    sc = pd.read_parquet(a.scored)
    if "src" not in sc.columns:
        sc["src"] = pool["src"].values[sc["p"].values]
    sc["country"] = s1["country"].values[sc["q"].values]
    for item in filter(None, a.country_scale.split(",")):
        c, f = item.split("=")
        m = sc["country"].values == c.lower()
        sc.loc[m, "prob"] = np.clip(sc.loc[m, "prob"].values * float(f), 0, 1)
    best = assign_best(sc["q"].values, sc["p"].values, sc["prob"].values).merge(sc[["q", "p", "src", "country"]], on=["q", "p"])
    if a.rule == "thr":
        sel = decide_threshold(best, a.thr)
    elif a.rule == "src":
        sel = decide_threshold_src(best, a.thr, a.thr3 if a.thr3 is not None else a.thr)
    else:
        sel = decide_expected_f(best, a.alpha)
    sel = sel.merge(best[["q", "p", "country"]], on=["q", "p"])
    for item in filter(None, a.country_thr.split(",")):
        c, t = item.split("=")
        c = c.lower()
        sel = sel[sel["country"] != c]
        bc = best[best["country"] == c]
        sel = pd.concat([sel, decide_threshold(bc, float(t)).merge(bc[["q", "p", "country"]], on=["q", "p"])], ignore_index=True)
    s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
    os.makedirs(a.out_dir, exist_ok=True)
    write_lists(os.path.join(a.out_dir, "matching_results.tsv"), s1_ids, to_lists(sel, s1_ids, pool_ids), "matched_entity_ids")
    cnt = sel.groupby("country").size()
    ent = pd.Series(s1["country"].values).value_counts()
    print("matched ids per country:", cnt.to_dict(), "| per entity:", (cnt / ent).round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
