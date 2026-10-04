"""Test-side probability ensemble: w * other_run + (1-w) * main_run on shared pairs, then the
usual one-owner assignment and per-source thresholds; writes matching_results.tsv."""
import argparse, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0,"code/src")
from config import Config
from data import load_prepared
from matching import to_lists, write_lists
from model import assign_best, decide_expected_f
ap=argparse.ArgumentParser()
ap.add_argument("--main", default="work7/test_scored_v8.parquet")
ap.add_argument("--other", default="work6/test_scored_v6ens_src.parquet")
ap.add_argument("--w", type=float, default=0.5)
ap.add_argument("--t2", type=float, default=0.75); ap.add_argument("--t3", type=float, default=0.75)
ap.add_argument("--rule", default="thr", choices=["thr","expf"]); ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--out-dir", default="output_v8ens")
a=ap.parse_args(); t=time.time()
cfg=Config(data_dir="dataset", work_dir="work7", out_dir=a.out_dir, n_jobs=31)
s1,pool=load_prepared(cfg,"test")
m=pd.read_parquet(a.main)[["q","p","prob"]].rename(columns={"prob":"pm"})
o=pd.read_parquet(a.other)[["q","p","prob"]].rename(columns={"prob":"po"})
d=m.merge(o,on=["q","p"],how="left"); cov=d["po"].notna().mean(); d["po"]=d["po"].fillna(d["pm"])
pv=(a.w*d["po"].to_numpy()+(1-a.w)*d["pm"].to_numpy()).astype(np.float32)
meta=pd.read_parquet(cfg.path("s4_test_meta.parquet"), columns=["q","p","src"])
best=assign_best(d["q"].to_numpy(), d["p"].to_numpy(), pv).merge(meta,on=["q","p"],how="left")
if a.rule=="expf": sel=decide_expected_f(best, a.alpha)
else:
    thr=np.where(best["src"].to_numpy()==2,a.t2,a.t3); sel=best[best["prob"].to_numpy()>=thr]
os.makedirs(cfg.out_dir, exist_ok=True)
s1_ids,pool_ids=s1["entity_id"].values,pool["entity_id"].values
write_lists(os.path.join(cfg.out_dir,"matching_results.tsv"), s1_ids, to_lists(sel,s1_ids,pool_ids), "matched_entity_ids")
per=pd.Series(s1["country"].values[sel["q"].to_numpy()]).value_counts()/pd.Series(s1["country"].values).value_counts()
print(f"[enstest] w={a.w} overlap {cov:.4f} rule {a.rule} thr ({a.t2},{a.t3}) -> {len(sel):,} ids; ids/entity {per.round(3).to_dict()}; {time.time()-t:.0f}s", flush=True)
print("ENSTEST_DONE", flush=True)
