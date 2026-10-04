"""Final decision on a scored test table: expected-F0.5 prefix cut or per-source/per-country
thresholds, one owner per record, then the two submission files + validator-ready output."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "code/src")
from config import Config
from data import load_prepared
from matching import to_lists, write_lists
from model import assign_best, decide_expected_f
ap = argparse.ArgumentParser()
ap.add_argument("--work-dir", default="work7")
ap.add_argument("--scored", default="work7/test_scored_v8.parquet")
ap.add_argument("--out-dir", default="output_v8_expf")
ap.add_argument("--rule", default="expf", choices=["expf", "thr"])
ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--t2", type=float, default=0.75); ap.add_argument("--t3", type=float, default=0.75)
ap.add_argument("--fr-scale", type=float, default=1.0, help="multiply French thresholds by this (thr rule)")
ap.add_argument("--n-jobs", type=int, default=31)
a = ap.parse_args(); t = time.time()
cfg = Config(data_dir="dataset", work_dir=a.work_dir, out_dir=a.out_dir, n_jobs=a.n_jobs)
s1, pool = load_prepared(cfg, "test")
d = pd.read_parquet(a.scored)
meta = pd.read_parquet(cfg.path("s4_test_meta.parquet"), columns=["q", "p", "src"])
best = assign_best(d["q"].to_numpy(), d["p"].to_numpy(), d["prob"].to_numpy()).merge(meta, on=["q", "p"], how="left")
if a.rule == "expf":
    sel = decide_expected_f(best, a.alpha)
else:
    fr = (s1["country"].values[best["q"].to_numpy()] == "France")
    thr = np.where(best["src"].to_numpy() == 2, a.t2, a.t3) * np.where(fr, a.fr_scale, 1.0)
    sel = best[best["prob"].to_numpy() >= thr]
os.makedirs(cfg.out_dir, exist_ok=True)
s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
write_lists(os.path.join(cfg.out_dir, "matching_results.tsv"), s1_ids, to_lists(sel, s1_ids, pool_ids), "matched_entity_ids")
share = pd.Series(s1["country"].values[np.unique(sel["q"].to_numpy())]).value_counts() / pd.Series(s1["country"].values).value_counts()
per = pd.Series(s1["country"].values[sel["q"].to_numpy()]).value_counts() / pd.Series(s1["country"].values).value_counts()
print(f"[assemble] {a.rule} alpha={a.alpha} thr=({a.t2},{a.t3}) fr_scale={a.fr_scale}: {len(sel):,} ids; "
      f"share-with-matches {share.round(4).to_dict()}; ids/entity {per.round(3).to_dict()}; {time.time()-t:.0f}s", flush=True)
print("ASSEMBLE_DONE", flush=True)
